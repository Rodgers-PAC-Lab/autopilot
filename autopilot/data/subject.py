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
        did_graduate (:class:`threading.Event`): Event used to signal if the subject has graduated the current step
    """
    _VERSION = 1



    def __init__(self,
                 name: str=None,
                 dir: Optional[Path] = None,
                 file: Optional[Path] = None,
                 structure: Subject_Structure = Subject_Structure()):
        """
        Args:
            name (str): subject ID
            dir (str): path where the .h5 file is located, if `None`, `prefs.get('DATADIR')` is used
            file (str): load a subject from a filename. if `None`, ignored.
            structure (:class:`.Subject_Structure`): Structure to use with this subject.
        """

        self.structure = structure

        self._lock = threading.Lock()

        # --------------------------------------------------
        # Find subject .h5 file
        # --------------------------------------------------

        if file:
            file = Path(file)
            if not name:
                name = file.stem

        else:
            if not name:
                raise FileNotFoundError('Need to either pass a name or a file, how else would we find the .h5 file?')

            if dir:
                dir = Path(dir)
            else:
                dir = Path(prefs.get('DATADIR'))

            file = dir / (name + '.h5')

        self.name = name
        self.logger = init_logger(self)
        self.file = file

        if not self.file.exists():
            raise FileNotFoundError(f"Subject file {str(self.file)} does not exist!")

        # make sure we have the expected structure
        with self._h5f() as h5f:
            self.structure.make(h5f)

        self._session_uuid = None

        # Is the subject currently running (ie. we expect data to be incoming)
        # Used to keep the subject object alive, otherwise we close the file whenever we don't need it
        self.running = False

        # We use a threading queue to dump data into a kept-alive h5f file
        self.data_queue = None
        self._thread = None
        self.did_graduate = threading.Event()

        with self._h5f() as h5f:
            # Every time we are initialized we stash the git hash
            history_row = h5f.root.history.hashes.row
            history_row['time'] = self._get_timestamp()
            try:
                history_row['hash'] = prefs.get('HASH')
                # FIXME: less implicit way of getting hash plz
            except AttributeError:
                history_row['hash'] = ''
            history_row.append()

            do_update = False
            if 'current' in h5f.root:
                do_update = True

        if do_update:
            self.logger.warning('Detected an old subject format, trying to update...')
            try:
                self._update_structure()
            except Exception as e:
                self.logger.exception(f"Unable to update! Got exception:\n{e}")

        if self.protocol:
            self.logger.debug("Attempting to update protocol")
            self._check_protocol_changed()

    @contextmanager
    def _h5f(self, lock:bool=True) -> tables.file.File:
        """
        Context manager for access to hdf5 file.

        Args:
            lock (bool): Lock the file while it is open, only use ``False`` for operations
                that are read-only: there should only ever be one write operation at a time.

        Examples:

            with self._h5f as h5f:
                # ... do hdf5 stuff

        Returns:
            function wrapped with contextmanager that will open the hdf file
        """

        # @contextmanager
        # def _h5f_context() -> tables.file.File:
        if lock:
            with self._lock:
                h5f = None
                try:
                    h5f = tables.open_file(str(self.file), mode="r+")
                    yield h5f
                except tables.exceptions.HDF5ExtError:
                    self.logger.debug('error opening with lock {}'.format(self.file))
                    raise
                finally:
                    if h5f is not None:
                        # This happens if an error happened when opening file
                        # Without this guard, we would get UnboundLocalError
                        # here, because h5f doesn't exist in that case.
                        h5f.flush()
                        h5f.close()

        else:
            h5f = None
            try:
                #~ h5f = tables.open_file(str(self.file), mode="r+")
                try:
                    h5f = tables.open_file(str(self.file), mode="r")
                except ValueError as e:
                    if 'already opened, but not in read-only mode' in e.args[0]:
                        h5f = tables.open_file(str(self.file), mode='r+')
                    else:
                        raise e

                yield h5f
            except tables.exceptions.HDF5ExtError:
                self.logger.debug('error opening wihout lock {}'.format(self.file))
                raise
            finally:
                if h5f is not None:
                    h5f.flush()
                    h5f.close()

    @property
    def info(self) -> Biography:
        """
        Subject biographical information
        """
        with self._h5f(lock=False) as h5f:
            info = h5f.get_node(self.structure.info.path)
            biodict = {}
            for k in info._v_attrs._f_list():
                biodict[k] = info._v_attrs[k]

        return Biography(**biodict)

    @property
    def bio(self) -> Biography:
        """
        Subject biographical information (alias for :meth:`.info`)
        """
        return self.info

    @property
    def protocol(self) -> Union[Protocol_Status, None]:
        """
        The status of the currently assigned protocol

        See :class:`.Protocol_Status`

        A property with an accompanying setter. When assigned to, stashes the details of the old
        protocol, and remakes the table structure to support the new task.
        """
        with self._h5f(lock=False) as h5f:
            protocol = h5f.get_node(self.structure.protocol.path)
            protocoldict = {}
            for k in protocol._v_attrs._f_list():

                protocoldict[k] = protocol._v_attrs[k]

            if 'protocol' in protocol:
                protocol_node = h5f.get_node(self.structure.protocol.path + '/protocol')
                #self.logger.debug('protocol_property: protocol_node = {}'.format(protocol_node))
                protocol_node = filenode.open_node(protocol_node)
                #self.logger.debug('protocol_property: protocol_node = {}'.format(protocol_node))
                protocoldict['protocol'] = json.loads(protocol_node.readall())
                protocol_node.close()

        if len(protocoldict) == 0:
            return None
        else:
            return Protocol_Status(**protocoldict)

    @protocol.setter
    def protocol(self, protocol:Protocol_Status):
        if self.protocol is not None and protocol.protocol != self.protocol.protocol:
            archive_name = f"{self._get_timestamp(simple=True)}_{self.protocol_name}"
            # make the group
            self._write_attrs('/history/past_protocols/' + archive_name, self.protocol.dict())
            self.logger.debug(f"Stashed old protocol details in {'/history/past_protocols/' + archive_name}")

        # check for differences
        diffs = []
        if self.protocol is None:
            diffs.append('protocol')
            diffs.append('step')
        else:
            if protocol.protocol_name != self.protocol_name:
                diffs.append('protocol')
            if protocol.step != self.step:
                diffs.append('step')

        for diff in diffs:
            if diff == 'protocol':
                self.update_history('protocol', protocol.protocol_name, value=protocol.protocol)
            elif diff == 'step':
                self.update_history('step', name=protocol.protocol[protocol.step]['step_name'],
                                    value=protocol.step)
        with self._h5f() as h5f:
            protocol_node = h5f.get_node(self.structure.protocol.path)
            for k, v in protocol.dict().items():
                if k == 'protocol':
                    if 'protocol' in protocol_node:
                        h5f.remove_node(self.structure.protocol.path + '/protocol')
                    protocol_filenode = filenode.new_node(h5f, where=self.structure.protocol.path, name='protocol')
                    protocol_filenode.write(json.dumps(v).encode('utf-8'))

                else:
                    protocol_node._v_attrs[k] = v

        # make sure that we have the required protocol structure
        try:
            self._make_protocol_structure(protocol.protocol_name, protocol.protocol)
        except ValueError as e:
            if 'Could not find subclass of' in str(e):
                task_name = str(e).split(' ')[-1].rstrip('!')
                self.logger.error(f"When attempting to make protocol data structure, could not find the task type {task_name}. If it's in a plugin, make sure that the plugin is in your plugin directory. The protocol has been assigned, but you will need to have the task code present to run it.")
            else:
                raise e

        # CR blank this because too verbose
        # self.logger.info(f"Saved new protocol status {protocol}")

    @property
    def protocol_name(self) -> str:
        """
        Name of the currently assigned protocol

        Convenience accessor for  :attr:`.Subject.protocol.protocol_name`
        """
        return self.protocol.protocol_name

    @property
    def current_trial(self) -> int:
        """
        Current number of trial for the assigned task

        Convenience accessor for ``.protocol.current_trial``

        Has Setter (can be assigned to)
        """
        return self.protocol.current_trial

    @current_trial.setter
    def current_trial(self, current_trial:int):
        protocol = self.protocol
        protocol.current_trial = current_trial
        self.protocol = protocol

    @property
    def session(self) -> int:
        """
        Current session of assigned protocol.

        Convenience accessor for ``.protocol.session``

        Has setter (can be assigned to)
        """
        return self.protocol.session

    @session.setter
    def session(self, session: int):
        protocol = self.protocol
        protocol.session = session
        self.protocol = protocol

    @property
    def step(self) -> int:
        """
        Current step of assigned protocol

        Convenience accessor for ``.protocol.step``

        Has setter (can be assigned to) to manually promote/demote subject to different steps of the protocol.
        """
        return self.protocol.step

    @step.setter
    def step(self, step: int):
        protocol = self.protocol
        protocol.step = step
        self.protocol = protocol

    @property
    def task(self) -> dict:
        """
        Protocol dictionary for the current step
        """
        return self.protocol.protocol[self.step]

    @property
    def session_uuid(self) -> str:
        """
        Automatically generated UUID given to each session, regardless of the session number.

        Ensures each session is uniquely addressable in the case of ambiguous session numbers
        (eg. subject was manually promoted or demoted and session number was unable to be recovered,
        so there are multiple sessions with the same number)
        """
        if self._session_uuid is None:
            self._session_uuid = str(uuid.uuid4())
        return self._session_uuid

    @property
    def history(self) -> History:
        """
        The Subject's history of parameter and other changes.

        See :class:`.History`
        """
        return self._read_table('/history/history', History)

    @property
    def hashes(self) -> Hashes:
        """
        History of version hashes and autopilot versions

        See :class:`.Hashes`
        """
        return self._read_table('/history/hashes', Hashes)

    @property
    def weights(self) -> Weights:
        """
        History of weights at the start and end of running a session.

        See :class:`.Weights`
        """
        return self._read_table('/history/weights', Weights)


    def _write_attrs(self, path: str, attrs:dict):
        with self._h5f() as h5f:
            try:
                node = h5f.get_node(path)

            except tables.exceptions.NoSuchNodeError:
                pathpieces = path.split('/')
                # if path was absolute, remove the blank initial one
                if pathpieces[0] == '':
                    pathpieces = pathpieces[1:]
                parent = '/' + '/'.join(pathpieces[:-1])
                node = h5f.create_group(parent, pathpieces[-1],
                                     title=pathpieces[-1], createparents=True)
            for k, v in attrs.items():
                node._v_attrs[k] = v

            h5f.flush()

    def _read_table(self, path:str, table:typing.Type[Table]) -> typing.Union[Table,pd.DataFrame]:
        with self._h5f(lock=False) as h5f:
            tab = h5f.get_node(path).read() # type: np.ndarray

        # unpack table to a dataframe
        df = pd.DataFrame.from_records(tab)
        for col in df.columns:
            if df[col].dtype == 'O':
                df[col] = df[col].str.decode("utf-8")

        try:
            return table(**df.to_dict(orient='list'))
        except Exception as e:
            self.logger.exception(f"Could not make table from loaded data, returning dataframe")
            return df

    @classmethod
    def new(cls,
            bio:Biography,
            structure: Optional[Subject_Structure] = Subject_Structure(),
            path: Optional[Path] = None,
            ) -> 'Subject':
        """
        Create a new subject file, make its structure, and populate its :class:`~.data.models.biography.Biography` .


        Args:
            bio (:class:`~.data.models.biography.Biography`): A collection of biographical information
                about the subject! Stored as attributes within `/info`
            structure (Optional[:class:`~.models.subject.Subject_Structure`]): The structure of tables and groups to
                use when creating this Subject. **Note:** This is not currently saved with the subject file,
                so if using a nonstandard structure, it needs to be passed every time on init. Sorry!
            path (Optional[:class:`pathlib.Path`]): Path of created file. If ``None``, make a file within
                the ``DATADIR`` within the user directory (typically ``~/autopilot/data``) using the subject ID as the filename.
                (eg. ``~/autopilot/data/{id}.h5``)

        Returns:
            :class:`.Subject` , Newly Created.
        """
        if path is None:
            path = Path(prefs.get('DATADIR')).resolve() / (bio.id + '.h5')
        else:
            path = Path(path)
            assert path.suffix == '.h5'

        if path.exists():
            raise FileExistsError(f"A subject file for {bio.id} already exists at {path}!")

        # use the open_file command directly here because we use mode="w"
        h5f = tables.open_file(filename=str(path), mode='w')

        # make basic structure
        structure.make(h5f)

        info_node = h5f.get_node(structure.info.path)
        for k, v in bio.dict().items():
            info_node._v_attrs[k] = v

        # compatibility - double `id` as name
        info_node._v_attrs['name'] = bio.id
        h5f.root._v_attrs['VERSION'] = cls._VERSION

        h5f.close()

        return Subject(name=bio.id, file=path)


    def update_history(self, type, name:str, value:typing.Any, step=None):
        """
        Update the history table when changes are made to the subject's protocol.

        The current protocol is flushed to the past_protocols group and an updated
        filenode is created.

        Note:
            This **only** updates the history table, and does not make the changes itself.

        Args:
            type (str): What type of change is being made? Can be one of

                * 'param' - a parameter of one task stage
                * 'step' - the step of the current protocol
                * 'protocol' - the whole protocol is being updated.

            name (str): the name of either the parameter being changed or the new protocol
            value (str): the value that the parameter or step is being changed to,
                or the protocol dictionary flattened to a string.
            step (int): When type is 'param', changes the parameter at a particular step,
                otherwise the current step is used.
        """
        self.logger.info(f'Updating subject {self.name} history - type: {type}, name: {name}, value: {value}, step: {step}')

        # Make sure the updates are written to the subject file


        # Check that we're all strings in here
        if not isinstance(type, str):
            type = str(type)
        if not isinstance(name, str):
            name = str(name)
        if not isinstance(value, str):
            value = str(value)

        # log the change
        with self._h5f() as h5f:
            history_row = h5f.root.history.history.row

            history_row['time'] = self._get_timestamp(simple=True)
            history_row['type'] = type
            history_row['name'] = name
            history_row['value'] = value
            history_row.append()

    def _check_protocol_changed(self):
        """Check whether the protocol on disk has changed. If it has, update!"""
        try:
            prot_name, disk_protocol = self._find_protocol(self.protocol_name)
        except Exception as e:
            self.logger.warning(f"Could not find protocol file to update internal representation of it. Got exception {e}")
            return

        if disk_protocol != self.protocol.protocol:
            self.logger.info('Protocol on disk changed from stored protocol. Updating')
            self.assign_protocol(disk_protocol, step_n=self.protocol.step, pilot=self.protocol.pilot, protocol_name=prot_name)

    def _find_protocol(self, protocol:typing.Union[Path, str, typing.List[dict]],
                       protocol_name: Optional[str]=None) -> typing.Tuple[str, typing.List[dict]]:
        """
        Resolve a protocol from a name, path, etc. into a list of dictionaries

        Returns:
            tuple of (protocol_name, protocol)
        """

        if isinstance(protocol, str):
            # check if it's just a json encoded dictionary
            try:
                protocol = json.loads(protocol)
            except json.decoder.JSONDecodeError:
                # try it as a path
                if not protocol.endswith('.json'):
                    protocol += '.json'
                protocol = Path(protocol)

        if isinstance(protocol, Path):
            if not protocol.exists():
                if protocol.is_absolute():
                    protocol = protocol.relative_to(prefs.get('PROTOCOLDIR'))
                else:
                    protocol = Path(prefs.get('PROTOCOLDIR')) / protocol
            if not protocol.exists():
                raise FileNotFoundError(f"Could not find protocol file {protocol}!")

            protocol_name = protocol.stem

            with open(protocol, 'r') as pfile:
                protocol = json.load(pfile)

        elif isinstance(protocol, list):
            if protocol_name is None:
                raise ValueError(f"If passed protocol as a list of dictionaries, need to also pass protocol_name")

        return protocol_name, protocol

    def _make_protocol_structure(self, protocol_name:str, protocol:typing.List[dict] ):
        """
        Use a :class:`.Protocol_Group` to make the necessary tables for the given protocol.
        """
        # make protocol structure!
        protocol_structure = Protocol_Group(
            protocol_name=protocol_name,
            protocol=protocol,
            structure=self.structure
        )
        with self._h5f() as h5f:
            protocol_structure.make(h5f)

    def assign_protocol(self, protocol:typing.Union[Path, str, typing.List[dict]],
                        step_n:int=0,
                        pilot: Optional[str] = None,
                        protocol_name:Optional[str]=None):
        """
        Assign a protocol to the subject.
        
        Specifically, a :class:`Protocol_Status` will be created using the
        provided attributes, and stored in the attribute `self.protocol`.

        If the subject has a currently assigned task, stashes it with 
        :meth:`~.Subject.stash_current`

        Creates groups and tables according to the data descriptions in the 
        task class being assigned.
        eg. as described in :class:`.Task.TrialData`.

        Updates the history table.

        Args:
            protocol (Path, str, dict): the protocol to be assigned. Can be one of

                * the name of the protocol (its filename minus .json) 
                    if it is in `prefs.get('PROTOCOLDIR')`
                * filename of the protocol (its filename with .json) 
                    if it is in the `prefs.get('PROTOCOLDIR')`
                * the full path and filename of the protocol.
                * The protocol dictionary serialized to a string
                * the protocol as a list of dictionaries

            step_n (int): Which step is being assigned?
            protocol_name (str): If passing ``protocol`` as a dict, have to 
                give a name to the protocol
        """
        # Protocol will be passed as a .json filename in prefs.get('PROTOCOLDIR')

        protocol_name, protocol = self._find_protocol(protocol, protocol_name)

        # check if this is the same protocol as we already have so we don't reset session number
        if self.protocol is not None and (protocol_name == self.protocol_name) and (step_n == self.step):
            session = self.session
            current_trial = self.current_trial

            self.logger.debug("Keeping existing session and current_trial counts")
        else:
            session = 0
            current_trial = 0

        if self.protocol is not None and pilot is None:
            self.logger.debug("Using pilot from previous assignation")
            pilot = self.protocol.pilot

        status = Protocol_Status(
            current_trial=current_trial,
            session=session,
            step=step_n,
            protocol=protocol,
            pilot = pilot,
            protocol_name=protocol_name,
        )
        # set current status (this will also stash any existing status and 
        # update the trial history tables as needed)

        self.protocol = status

    # --------------------------------------------------
    # prepare run
    # --------------------------------------------------

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

        # Get current task parameters and handles to tables
        task_params = self.protocol.protocol[self.step]
        
        # increment session and clear session_uuid to ensure uniqueness
        self.session += 1
        self._session_uuid = None
        
        # Generate a session_name as a concatenation of the current time,
        # and the subject name
        session_dt = datetime.datetime.now().isoformat()
        session_dt_string = session_dt.strftime('%Y-%m-%d-%H-%M-%S-%f')
        session_name = '{}_{}_{}'.format(session_dt, self.name)

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
        # TODO: store pilot name here too
        with open(os.path.join(sandbox_dir, 'task_params.json')) as fi:
            # This was copied from elsewhere
            json.dump(
                task_params, fi, indent=4, separators=(',', ': '), 
                sort_keys=True)
        
        # TODO: copy in the code that was used, my prefs.json, etc
        # TODO: start capturing video 

        # spawn thread to accept data
        self.data_queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._data_thread,
            args=(self.data_queue, hdf5_filename)
        )
        self._thread.start()
        self.running = True

        # return a completed task parameter dictionary
        task_params['subject'] = self.name
        task_params['step'] = int(self.step)
        task_params['current_trial'] = int(self.current_trial)
        task_params['session'] = int(self.session)
        return task_params

    # --------------------------------------------------
    # Data Thread Private Methods!
    # --------------------------------------------------

    def _data_thread(self, queue:queue.Queue, hdf5_filename:str):
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
        """
        task_class = autopilot.tasks.paft.PAFT
        table_desc = task_class.TrialData.to_pytables_description()
        
        # Open the HDF5 file where data is stored
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

            # Get the column names
            #trial_keys = trial_table.colnames


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
            

            ## start getting data
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
                    self.logger.exception(f'exception in data thread: {e}')

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
        if self.graduation:
            # set our graduation flag, the terminal will get the rest rolling
            did_graduate = self.graduation.update(trial_row)
            if did_graduate is True:
                self.did_graduate.set()
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

    # --------------------------------------------------
    # Data retrieval
    # --------------------------------------------------

    def get_trial_data(self,
                       step: typing.Union[int, list, str, None] = None
                       ) -> Union[typing.List[pd.DataFrame], pd.DataFrame]:
        """
        Get trial data from the current task.

        Args:
            step (int, list, str, None): Step that should be returned, can be one of

                * ``None``: All steps (default)
                * -1: the current step
                * int: a single step
                * list: of step numbers or step names (excluding S##_)
                * string: the name of a step (excluding S##_)

        Returns:
            :class:`pandas.DataFrame`: DataFrame of requested steps' trial data (or list of dataframes).
        """

        try:
            groups = Protocol_Group(self.protocol_name, self.protocol.protocol)
        except ValueError:
            self.logger.warning(f"Could not recreate data descriptions from protocol, likely because a plugin is missing or has not been imported. Attempting to recreate from pytables description, but this might not be fully accurate. check AUTOPLUGIN and that the plugin is in the plugin directory.")
            groups = None

        step_names = [s['step_name'].lower() for s in self.protocol.protocol]

        # convert input into a list of integers
        if isinstance(step, int):
            if step == -1:
                # the current step
                step = self.step
            steps = [step]
        elif isinstance(step, list):
            steps = []
            for s in step:
                try:
                    # check if it's an integer
                    steps.append(int(s))
                except ValueError:
                    # must be a step name!
                    steps.append(step_names.index(s.lower()))
        elif isinstance(step, str):
            # get index from step name!
            steps = [step_names.index(step.lower())]
        else:
            # get all steps
            steps = list(range(len(self.protocol.protocol)))

        ret = [self._get_step_data(i, groups) for i in steps]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _get_step_data(self, step:int, groups:Optional[Protocol_Group]=None) -> pd.DataFrame:
        """
        Get individual step data, using the protocol group if given, otherwise try and recover from pytables description
        """
        # find the table
        if groups:
            path = groups.steps[step].path + '/trial_data'
            data_table = groups.steps[step].trial_data
        else:
            group_path = f"/data/{self.protocol_name}"
            with self._h5f(lock=False) as h5f:
                step_groups = sorted(h5f.get_node(group_path)._v_children.keys())
                path = f"{group_path}/{step_groups[step]}/trial_data"
                data_node = h5f.get_node(path) # type: tables.table.Table
                data_table = Table.from_pytables_description(data_node.description)

        # get the data from the table!
        data = self._read_table(path, data_table)
        if isinstance(data, Table):
            data = data.to_df()
        return data


    def _get_timestamp(self, simple=False):
        # type: (bool) -> str
        """
        Makes a timestamp.

        Args:
            simple (bool):
                if True:
                    returns as format '%y%m%d-%H%M%S', eg '190201-170811'
                if False:
                    returns in isoformat, eg. '2019-02-01T17:08:02.058808'

        Returns:
            basestring
        """
        # Timestamps have two different applications, and thus two different formats:
        # coarse timestamps that should be human-readable
        # fine timestamps for data analysis that don't need to be
        if simple:
            return datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        else:
            return datetime.datetime.now().isoformat()

    def get_weight(self, which='last', include_baseline=False):
        """
        Gets start and stop weights.

        TODO:
            add ability to get weights by session number, dates, and ranges.

        Args:
            which (str):  if 'last', gets most recent weights. Otherwise returns all weights.
            include_baseline (bool): if True, includes baseline and minimum mass.

        Returns:
            dict
        """
        # get either the last start/stop weights, optionally including baseline
        # TODO: Get by session
        weights = {}

        with self._h5f(lock=False) as h5f:
            weight_table = h5f.root.history.weights
            if which == 'last':
                for column in weight_table.colnames:
                    try:
                        weights[column] = weight_table.read(-1, field=column)[0]
                    except IndexError:
                        weights[column] = None
            else:
                for column in weight_table.colnames:
                    try:
                        weights[column] = weight_table.read(field=column)
                    except IndexError:
                        weights[column] = None

            if include_baseline is True:
                try:
                    baseline = float(h5f.root.info._v_attrs['baseline_mass'])
                except KeyError:
                    baseline = 0.0
                minimum = baseline*0.8
                weights['baseline_mass'] = baseline
                weights['minimum_mass'] = minimum

        return weights

    def set_weight(self, date, col_name, new_value):
        """
        Updates an existing weight in the weight table.

        TODO:
            Yes, i know this is bad. Merge with update_weights

        Args:
            date (str): date in the 'simple' format, %y%m%d-%H%M%S
            col_name ('start', 'stop'): are we updating a pre-task or post-task weight?
            new_value (float): New mass.
        """

        with self._h5f() as h5f:
            weight_table = h5f.root.history.weights
            # there should only be one matching row since it includes seconds
            for row in weight_table.where('date == b"{}"'.format(date)):
                row[col_name] = new_value
                row.update()


    def update_weights(self, start=None, stop=None):
        """
        Store either a starting or stopping mass.

        `start` and `stop` can be passed simultaneously, `start` can be given in one
        call and `stop` in a later call, but `stop` should not be given before `start`.

        Args:
            start (float): Mass before running task in grams
            stop (float): Mass after running task in grams.
        """
        with self._h5f() as h5f:
            if start is not None:
                weight_row = h5f.root.history.weights.row
                weight_row['date'] = self._get_timestamp(simple=True)
                weight_row['session'] = self.session
                weight_row['start'] = float(start)
                weight_row.append()
            elif stop is not None:
                # TODO: Make this more robust - don't assume we got a start weight
                h5f.root.history.weights.cols.stop[-1] = stop
            else:
                self.logger.warning("Need either a start or a stop weight")

    def _graduate(self):
        """
        Increase the current step by one, unless it is the last step.
        """
        if len(self.protocol.protocol)<=self.step+1:
            self.logger.warning('Tried to _graduate from the last step!\n Task has {} steps and we are on {}'.format(len(self.protocol.protocol), self.step+1))
            return

        # increment step, update_history should handle the rest
        self.step += 1

    def _update_structure(self):
        """
        Update old formats to new ones
        """
        backup = self.file.with_stem(self.file.stem + f"_backup-{datetime.date.today().isoformat()}")
        append_int = 1
        while backup.exists():
            backup = self.file.with_stem(self.file.stem + f"_backup-{datetime.date.today().isoformat()}-{append_int}")
            append_int += 1
        self.logger.warning(f'Attempting to update structure, making a backup to {str(backup)}')
        shutil.copy(str(self.file), str(backup))

        protocol = None
        with self._h5f() as h5f:
            if 'current' in h5f.root:
                protocol = _update_current(h5f)

        if protocol is not None:
            self.protocol = protocol
            with self._h5f() as h5f:
                h5f.remove_node('/current')
                self.logger.debug("Removed current node")


def _update_current(h5f) -> Protocol_Status:
    """Update the old 'current' filenode to the new Protocol Status"""
    current_node = filenode.open_node(h5f.root.current)
    protocol_string = current_node.readall()
    protocol = json.loads(protocol_string)
    step = current_node.attrs['step']
    protocol_name = current_node.attrs['protocol_name']

    current_trial = 0
    session = 0

    got_protocol = False
    try:
        group_stx = Protocol_Group(protocol_name=protocol_name, protocol=protocol)
        active_step = group_stx.steps[step]
        trial_tab = h5f.get_node(active_step.path, 'trial_data')
        got_protocol = True
    except tables.NoSuchNodeError:
        print("Couldnt find trial_data node, not able to retreive data from trial table. Using zeros for current trial and session")
    except ValueError:
        print("Couldnt find task, not able to retrieve data from trial table. Using zeros for current trial and session")

    if got_protocol:
        try:
            current_trial = trial_tab['trial_num'][-1]
        except:
            print('Coudlnt get current trial, using 0')
            current_trial = 0

    try:
        session = h5f.root.info._v_attrs['session']
    except:
        print('couldnt get session from metadata')
        if got_protocol:
            print('getting session from trial table')
            try:
                session = trial_tab['session'][-1]
            except:
                print('couldnt get session from trial table, using 0')
                session = 0

    try:
        pilot = h5f.root.info._v_attrs.__dict__.get('pilot', '')
    except Exception as e:
        print(f'couldnt get pilot from subject info, leaving blank got exception {e}')
        pilot = ''

    status = Protocol_Status(
        current_trial=current_trial,
        protocol=protocol,
        step=step,
        session=session,
        protocol_name=protocol_name,
        pilot=pilot
    )
    return status
