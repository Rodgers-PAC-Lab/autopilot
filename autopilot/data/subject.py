"""
Abstraction layer around subject data storage files
"""
import os
import threading
import datetime
import json
import uuid
import warnings
import typing
from typing import Union, Optional
from contextlib import contextmanager
from pathlib import Path
import shutil
import queue

# watchtower
import urllib3
import requests

import pandas as pd
import numpy as np
import tables
from tables.tableextension import Row
from tables.nodes import filenode

import autopilot
from autopilot import prefs
from autopilot.data.modeling.base import Table
from autopilot.data.models.subject import Subject_Structure, Protocol_Status, Hashes, History, Weights
from autopilot.data.models.biography import Biography
from autopilot.data.models.protocol import Protocol_Group
from autopilot.utils.loggers import init_logger

if typing.TYPE_CHECKING:
    from autopilot.tasks.graduation import Graduation

# suppress pytables natural name warnings
warnings.simplefilter('ignore', category=tables.NaturalNameWarning)

import pandas


class Subject(object):
    """
    Class for managing one subject's data and protocol.

    Creates a :mod:`tables` hdf5 file in `prefs.get('DATADIR')` with the general structure::

        / root
        |--- current (tables.filenode) storing the current task as serialized JSON
        |--- data (group)
        |    |--- task_name  (group)
        |         |--- S##_step_name
        |         |    |--- trial_data
        |         |    |--- continuous_data
        |         |--- ...
        |--- history (group)
        |    |--- hashes - history of git commit hashes
        |    |--- history - history of changes: protocols assigned, params changed, etc.
        |    |--- weights - history of pre and post-task weights
        |    |--- past_protocols (group) - stash past protocol params on reassign
        |         |--- date_protocol_name - tables.filenode of a previous protocol's params.
        |         |--- ...
        |--- info - group with biographical information as attributes

    Attributes:
        name (str): Subject ID
        file (str): Path to hdf5 file - usually `{prefs.get('DATADIR')}/{self.name}.h5`
        logger (:class:`logging.Logger`): from :func:`~.utils.loggers.init_logger`
        running (bool): Flag that signals whether the subject is currently running a task or not.
        data_queue (:class:`queue.Queue`): Queue to dump data while running task
    """
    _VERSION = 1

    def __init__(self, name: str=None):
        """
        Args:
            name (str): subject ID
        """
        # Save name of subject
        self.name = name
        
        # Create logger
        self.logger = init_logger(self)

        # Is the subject currently running (ie. we expect data to be incoming)
        # Used to keep the subject object alive, 
        # otherwise we close the file whenever we don't need it
        self.running = False

        # We use a threading queue to dump data into a kept-alive h5f file
        self.data_queue = None
        self._thread = None

    def prepare_run(self, pilot:str) -> dict:
        """
        Prepares the Subject object to receive data while running the task.

        Gets information about current task, trial number,
        spawns :class:`~.tasks.graduation.Graduation` object,
        spawns :attr:`~.Subject.data_queue` and calls :meth:`~.Subject._data_thread`.

        Returns:
            Dict: the parameters for the current step, with subject id, step number,
                current trial, and session number included.
        """
        # This subject must have a :class:`Protocol_Status` assigned to the
        # attribute `self.protocol`, so that we know what task to run. 
        # Otherwise, raise an error.
        if self.protocol is None:
            e = RuntimeError(
                "No task assigned to subject, can't prepare_run. "
                'use Subject.assign_protocol or protocol reassignment wizard '
                'in the terminal GUI')
            self.logger.exception(f"{e}")
            raise e

        # Get current task parameters
        # I think this is essentially just the contents of the protocol JSON
        task_params = self.protocol.protocol[self.step]
        task_class_name = task_params['task_type']
        
        """
        print('self.protocol: {}'.format(str(self.protocol)))
        print('self.protocol.protocol: {}'.format(str(self.protocol.protocol)))
        print('self.step: {}'.format(self.step))
        
        self.protocol: dict-like, has keys 'assigned', 'current_trial', 
            'pilot', 'protocol', 'protocol_name', 'session', 'step'
        
        self.protocol.protocol: list-like of dict, each dict is 
            like the protocol JSON
        
        self.step: int, indexing self.protocol.protocol
        """
        
        ## Calculate reward amount
        # Box-specific reward amount
        if pilot == 'rpi_parent01':
            box_reward = [85, 100, 120]
        elif pilot == 'rpi_parent02':
            box_reward = [55, 65, 80]
        elif pilot == 'rpiparent03':
            box_reward = [55, 65, 80]
        elif pilot == 'rpiparent04':
            box_reward = [55, 65, 80]
        else:
            self.logger.debug(
                'unknown reward amount for {}, using 55'.format(pilot))
            box_reward = [55, 65, 80]
        
        # Mouse-specific reward amount
        high_reward_l = []
        med_reward_l = []
        low_reward_l = []
        if self.name in high_reward_l:
            mouse_reward = box_reward[2]
        elif self.name in med_reward_l:
            mouse_reward = box_reward[1]
        elif self.name in low_reward_l:
            mouse_reward = box_reward[0]
        else:
            self.logger.debug(
                'unknown reward amount for {}, using highest'.format(self.name))
            mouse_reward = box_reward[1]
        
        # Store this in task_params, and it will be sent to the pilot to
        # start the task
        # Note that this is overwriting the value loaded from the protocol
        task_params['reward'] = mouse_reward
        
        
        ## other session stuff
        # increment session and clear session_uuid to ensure uniqueness
        self.session += 1
        self._session_uuid = None
        
        # Generate a session_name as a concatenation of the current time,
        # and the subject name
        session_dt = datetime.datetime.now()
        session_dt_string = session_dt.strftime('%Y-%m-%d-%H-%M-%S-%f')
        session_name = '{}_{}'.format(session_dt_string, self.name)

        # Create a location to store this
        # TODO: fetch sandbox_root_dir from prefs
        sandbox_root_dir = os.path.expanduser('~/autopilot/data/sandboxes')
        sandbox_dir = os.path.join(sandbox_root_dir, session_name)
        try:
            os.mkdir(sandbox_dir)
        except OSError:
            raise OSError("cannot create sandbox dir at {}".format(sandbox_dir))
        
        # Generate the HDF5 filename for _data_thread
        hdf5_filename = os.path.join(sandbox_dir, session_name + '.hdf5')
        
        # Copy in the task_params used
        with open(os.path.join(sandbox_dir, 'task_params.json'), 'w') as fi:
            # This was copied from elsewhere
            json.dump(
                task_params, fi, indent=4, separators=(',', ': '), 
                sort_keys=True)
        
        # Choose camera_name from pilot
        if pilot == 'rpi_parent01':
            camera_name = 'e3v82e4'
        elif pilot == 'rpi_parent02':
            camera_name = 'e3v82c6'
        elif pilot == 'rpiparent03':
            camera_name = 'e3v82d2'
        elif pilot == 'rpiparent04':
            camera_name = 'e3v8370'
        else:
            camera_name = None
        
        # For now, disable saving
        camera_name = None
        
        # Set creation time (might be needed to line up with videos)
        sandbox_creation_time = datetime.datetime.now().isoformat()
        
        # Store sandbox_params, currently just pilot name and task name
        sandbox_params = {
            'pilot': pilot, 
            'task_class_name': task_class_name,
            'protocol_name': self.protocol.protocol_name,
            'camera_name': camera_name,
            'sandbox_creation_time': sandbox_creation_time,
            }
        with open(os.path.join(sandbox_dir, 'sandbox_params.json'), 'w') as fi:
            json.dump(
                sandbox_params, fi, indent=4, separators=(',', ': '), 
                sort_keys=True)
        
        # TODO: copy in the code that was used, my prefs.json, etc

        # spawn thread to accept data
        self.data_queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._data_thread,
            args=(self.data_queue, hdf5_filename, task_class_name, camera_name)
        )
        self._thread.start()
        self.running = True

        # return a completed task parameter dictionary
        task_params['subject'] = self.name
        task_params['step'] = int(self.step)
        task_params['current_trial'] = int(self.current_trial)
        task_params['session'] = int(self.session)
        return task_params

    def _data_thread(self, queue:queue.Queue, hdf5_filename:str, 
        task_class_name:str, camera_name=None):
        """Target of data thread, which writes data to hdf5 during task.

        receives data through :attr:`~.Subject.queue` as dictionaries. 
        Data can be
        partial-trial data (eg. each phase of a trial) as long as 
        the task returns a dict with
        'TRIAL_END' as a key at the end of each trial.

        each dict given to the queue should have the `trial_num`, 
        and this method can
        properly store data without passing `TRIAL_END` if so. 
        I recommend being explicit, however.

        Checks graduation state at the end of each trial.

        Args:
            queue (:class:`queue.Queue`): passed by 
                :meth:`~.Subject.prepare_run` and used by other
                objects to pass data to be stored.
            hdf_filname (str): where to write the hdf5 data for this session
            task_class_name (str): name of the task, used by autopilot.get_task
                to instantiate the task and get the TrialData
            camera_name (str or None): name of the camera to use, or None
                if None, no video is taken
                otherwise, watchtower start/stop commands are sent
        """
        ## Setup watchtower
        # (optional) Disable the "insecure requests" warning for https certs
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        if camera_name is not None:
            # what watchtower url to control
            watchtowerurl = 'https://192.168.11.148:4343'

            # login and obtain API token
            username = 'mouse'
            password = 'watchtowerpassword'
            r = requests.post(
                watchtowerurl+'/api/login', 
                data={'username': username, 'password': password}, 
                verify=False)
            j = json.loads(r.text)
            apit = j['apitoken']

            #~ # set save path
            #~ response = requests.post(
                #~ watchtowerurl+'/api/sessions/rename', 
                #~ data = {'Filepath': '/home/mouse/Videos'}, verify=False)
    
        
        ## Get the table_desc to create the HDF5 file
        task_class = autopilot.get_task(task_class_name)
        table_desc = task_class.TrialData.to_pytables_description()
        
        
        ## Open the HDF5 file where data is stored
        with tables.open_file(hdf5_filename, 'w') as h5f:
            ## Create a trial table for this session
            trial_table = h5f.create_table(
                where=h5f.root, 
                name='trial_data', 
                description=table_desc, 
                title='subject X on date Y',
                )
            
            # Get a link to the row to add
            trial_row = trial_table.row


            ## Create a continuous table for this session
            # This is where continuous data for this session lives
            continuous_session_group = h5f.create_group(h5f.root, 'continuous_data')
            
            # Also create a separate table for each column in ContinuousData
            # That table will have one column for that piece of data,
            # and a second column called "timestamp" of type StringAtom
            # Previously this was only done for items in 
            # step_group['continuous_data']._v_attrs['data']
            # But that is only set at the time of protocol assignment
            # This way, it is refreshed each time the task is changed
            cont_tables = {}
            for colname in task_class.ContinuousData.columns.keys():
                cont_tables[colname] = h5f.create_table(
                    continuous_session_group, 
                    colname,
                    description={
                        colname: task_class.ContinuousData.columns[colname],
                        'timestamp': tables.StringCol(50),
                    }, 
                )
            
            
            ## Create chunkdata 
            # Iterate over any defined CHUNKDATA_CLASSES in task_class
            chunk_table_d = {}
            for chunk_class in task_class.CHUNKDATA_CLASSES:
                # Create a chunk_table for this chunk_class
                chunk_table = h5f.create_table(
                    continuous_session_group, 
                    chunk_class.__name__,
                    chunk_class,
                    )
                
                # Store
                chunk_table_d[chunk_class.__name__] = chunk_table
            
            self.logger.debug('_data_thread: hdf5 created: {}'.format(str(h5f.root)))
            

            ## try/finally ensures video stop save is always called
            try:
                if camera_name is not None:
                    # start the video
                    self.logger.debug('starting video save')
                    response = requests.post(
                        watchtowerurl+'/api/cameras/action', 
                        data={
                            'SerialGroup[]': [camera_name], 
                            'Action': 'RECORDGROUP', 
                            'apitoken': apit,
                        }, 
                        verify=False)
                    self.logger.debug('video stop save command sent')
                    
                    if not response.ok:
                        self.logger.debug(
                            'error: response after start save command: {}'.format(
                            response.text))
    
                # stop when 'END' gets put in the queue
                for data in iter(queue.get, 'END'):
                    #self.logger.debug('_data_thread: received {}'.format(data))
                    
                    # wrap everything in try because this thread shouldn't crash
                    try:
                        # special case chunk data
                        if 'chunkclass_name' in data.keys():
                            self._save_chunk_data(h5f, data, chunk_table_d)

                        # special case continuous data
                        elif 'continuous' in data.keys():
                            self._save_continuous_data(h5f, data, cont_tables)
                        
                        # regular old trial data
                        else:
                            self.logger.debug('received trial data: {}'.format(data))
                            # If we get trial data out of order, 
                            # try and write it back in the correct row.
                            if 'trial_num' in data.keys() and 'trial_num' in trial_row:
                                trial_row = self._sync_trial_row(
                                    data['trial_num'], trial_row, trial_table)
                                del data['trial_num']
                                self.logger.debug('fixed trial data: {}'.format(data))
                            
                            self.logger.debug('saving trial data')
                            self._save_trial_data(data, trial_row, trial_table)

                    except Exception as e:
                        # we shouldn't throw any exception in this thread, just log it and move on
                        self.logger.exception(f'error: exception in data thread: {e}')
            
            finally:
                if camera_name is not None:
                    # stop the video
                    self.logger.debug('stopping video save')
                    response = requests.post(
                        watchtowerurl+'/api/cameras/action', 
                        data={
                            'SerialGroup[]': [camera_name], 
                            'Action': 'STOPRECORDGROUP', 
                            'apitoken': apit,
                        }, 
                        verify=False)
                    self.logger.debug('video stop save command sent')
                    
                    if not response.ok:
                        self.logger.debug(
                            'error: response after stop save command: {}'.format(
                            response.text))                

    def _save_chunk_data(self, 
        h5f: tables.File,
        data: dict,
        chunk_table_d: dict,
        ):
        """Pops `payload` from `data` and stores as DataFrame in chunk table.
        
        """
        # Log
        chunkclass_name = data['chunkclass_name']
        self.logger.debug('chunk data received of type {}'.format(
            chunkclass_name))
        
        # Get the appropriate chunk_table
        self.logger.debug('chunktable_d keys: {}'.format(chunk_table_d.keys()))
        chunk_table = chunk_table_d[chunkclass_name]
        
        # Pop payload from `data`, continuing if no rows to add
        payload = data.pop('payload')
        if len(payload) == 0:
            self.logger.debug('_save_chunk_data: payload was empty')
            return 

        # Pop payload_columns
        # All other items in `data` are ignored
        payload_columns = data.pop('payload_columns')
        
        # Reconstruct a DataFrame out of the payload
        payload_df = pandas.DataFrame(
            payload, columns=payload_columns)
        
        # Include exactly those columns that are in chunk_table
        # This will insert np.nan for any missing columns
        # This will cause an error if any of them are supposed to be int
        sliced_payload_df = payload_df.reindex(
            chunk_table.colnames, axis=1)
        
        # Convert to list of tuples, as expected by pytables
        to_append = list(map(tuple, sliced_payload_df.values))
        
        # Append
        try:
            chunk_table.append(to_append)
        except ValueError:
            self.logger.debug('error: failed to append chunk data!')
            self.logger.debug('payload_df:\n{}'.format(payload_df))
            self.logger.debug(
                'sliced_payload_df:\n{}'.format(sliced_payload_df))        

    def _save_continuous_data(self,
            h5f: tables.File,
            data: dict,
            cont_tables: typing.Dict[str, tables.table.Table],
            ):
        """Save trial `data` as a row in `cont_tables`"""
        # Iterate over all items in `data`
        for k, v in data.items():
            # Store the items we expected to receive
            if k in cont_tables.keys():
                # Append the received value and the timestamp
                cont_tables[k].row[k] = v
                cont_tables[k].row['timestamp'] = (
                    data.get('timestamp', ''))
                cont_tables[k].row.append()             
            
            # continue silently for items we expect to ignore
            # (unless they were explicitly included in the table)
            elif k in ['timestamp', 'subject', 'pilot', 'continuous']:
                continue
            
            # otherwise warn
            else:
                self.logger.warning(
                    "continuous data dropped because "
                    "{} not recognized".format(k))

    def _save_trial_data(self, data:dict, trial_row:Row, 
        trial_table:tables.table.Table):
        """Save `data` to `trial_row`, potentially incrementing trial
        
        Each item (k, v) in `data` is iterated. Keys that correspond to
        columns in the trial table are saved. Unrecognized keys are
        dropped with an exception.
        
        If TRIAL_END is in `data`, or if saving one of the items would 
        overwrite existing data, then the trial row is incremented.
        """
        # Iterate over all items in `data`, saving each one
        for k, v in data.items():
            if k in ('TRIAL_END', 'pilot', 'subject'):
                # Don't try to save these in the trial table
                # TRIAL_END is a flag for incrementing trial
                # 'pilot' and 'subject' are included in messages
                continue

            elif k in trial_table.colnames:
                # `k` is a known column of the trial table, so we should
                # save it
                
                # If this would overwrite an existing value, then instead,
                # increment the trial and print a warning
                if k in trial_row and trial_row[k] not in (None, b'', 0) and k != 'trial_num':
                    self.logger.warning(
                        f"Received two values for key, making new row.: {k} "
                        f"and trial row: {trial_row.nrow}, existing value: "
                        f"{trial_row[k]}, new value: {v}"
                        )
                    
                    # Increment trial
                    self._increment_trial(trial_row)
                
                # Store the value in `trial_row`
                trial_row[k] = v
            
            else:
                # `k` is not a known column of the trial table
                # This means we're dropping data, so print exception!
                # TODO: expand trial_table!
                self.logger.exception(
                    f"Trial data dropped because no column for key: {k}, "
                    f"value: {v}")

        # Increment the trial number when TRIAL_END is received
        if 'TRIAL_END' in data.keys():
            self._increment_trial(trial_row)

        # always flush so that our row iteration routines above will 
        # find what they're looking for
        trial_table.flush()

    def _sync_trial_row(self, trial_num:int, trial_row:Row, trial_table:tables.table.Table) -> Row:
        if trial_row['trial_num'] in (None, b''):
            self.logger.debug("_sync_trial_row: this row's trial_num is "
                "blank, setting to {}".format(trial_num))
            trial_row['trial_num'] = trial_num

        elif trial_num == trial_row['trial_num'] + 1:
            self.logger.debug("_sync_trial_row: this row's trial_num is "
                "one greater than expected, incrementing trial to {}".format(
                trial_num))
            self._increment_trial(trial_row)
            trial_row['trial_num'] = trial_num

        elif trial_num == trial_row['trial_num']:
            self.logger.debug(
                "_sync_trial_data: this row's trial_num is as expected")
            # fine! we're on the right one
            pass

        else:
            # we're on the wrong row somehow!

            # find row with this trial number if it exists
            # this will return a list of rows with matching trial_num.
            # if it's empty, we didn't receive a TRIAL_END and should create a new row
            # FIXME: this should also ensure that the trial_num comes from a row with a matching session_uuid

            other_row = [r for r in trial_table.where(f"trial_num == {trial_num}")]
            self.logger.debug(
                "_sync_trial_data: error, we're on the wrong row, and it's "
                "not just the next one. potential matches: {}".format(
                other_row))

            if len(other_row) == 0:
                # proceed to fill the row below, we got trial data discontinuously somehow
                self.logger.warning(f"Got discontinuous trial data")
                self.logger.debug(
                    "_sync_trial_data: no matches, using {}".format(trial_num))
                self._increment_trial(trial_row)
                trial_row['trial_num'] = trial_num

            elif len(other_row) == 1:
                # return the other row! (if an overwrite is attempted, append and go to next row anyway)
                self.logger.debug(
                    "_sync_trial_data: matched row {} and using it".format(other_row[0]))
                trial_row = other_row[0]

            else:
                # we have more than one row with this trial_num.
                # shouldn't happen, but we dont' want to throw any data away
                self.logger.warning(f'Found multiple rows with same trial_num: {trial_num}')
                self.logger.debug(
                    "_sync_trial_data: matched multiple rows, using {}".format(trial_num))
                # continue just for data conservancy's sake
                self._increment_trial(trial_row)
                trial_row['trial_num'] = trial_num

            return trial_row

    def _increment_trial(self, trial_row: Row):
        self.logger.debug('Trial Incremented')
        trial_row['session'] = self.session
        trial_row['session_uuid'] = self.session_uuid
        trial_row.append()

    def save_data(self, data):
        """
        Alternate and equivalent method of putting data in the queue as `Subject.data_queue.put(data)`

        Args:
            data (dict): trial data. each should have a 'trial_num', and a dictionary with key
                'TRIAL_END' should be passed at the end of each trial.
        """
        self.data_queue.put(data)

    def stop_run(self):
        """
        puts 'END' in the data_queue, which causes :meth:`~.Subject._data_thread` to end.
        """
        self.data_queue.put('END')
        self._thread.join(5)
        self.running = False
        if self._thread.is_alive():
            self.logger.warning('Data thread did not exit')
