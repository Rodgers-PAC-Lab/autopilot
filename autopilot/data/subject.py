"""

Classes for managing data and protocol access and storage.

Currently named subject, but will likely be refactored to include other data
models should the need arise.

"""
import threading
import datetime
import json
import warnings
import typing
from copy import copy
from typing import Optional
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import numpy as np
import tables
from tables.nodes import filenode

import autopilot
from autopilot import prefs
from autopilot.data.models.subject import Subject_Structure, Task_Status
from autopilot.data.models.biography import Biography
from autopilot.data.models.protocol import Protocol_Group
from autopilot.core.loggers import init_logger

import queue

# suppress pytables natural name warnings
warnings.simplefilter('ignore', category=tables.NaturalNameWarning)

# --------------------------------------------------
# Classes to describe structure of subject files
# --------------------------------------------------


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
        lock (:class:`threading.Lock`): manages access to the hdf5 file
        name (str): Subject ID
        file (str): Path to hdf5 file - usually `{prefs.get('DATADIR')}/{self.name}.h5`
        current (dict): current task parameters. loaded from
            the 'current' :mod:`~tables.filenode` of the h5 file
        step (int): current step
        protocol_name (str): name of currently assigned protocol
        current_trial (int): number of current trial
        running (bool): Flag that signals whether the subject is currently running a task or not.
        data_queue (:class:`queue.Queue`): Queue to dump data while running task
        _thread (:class:`threading.Thread`): thread used to keep file open while running task
        did_graduate (:class:`threading.Event`): Event used to signal if the subject has graduated the current step
        STRUCTURE (list): list of tuples with order:

            * full path, eg. '/history/weights'
            * relative path, eg. '/history'
            * name, eg. 'weights'
            * type, eg. :class:`.Subject.Weight_Table` or 'group'

        node locations (eg. '/data') to types, either 'group' for groups or a
            :class:`tables.IsDescriptor` for tables.
    """
    _VERSION = 1



    def __init__(self,
                 name: str=None,
                 dir: Optional[Path] = None,
                 file: Optional[Path] = None,
                 new: bool=False,
                 biography: Biography=None,
                 structure: Subject_Structure = Subject_Structure()):
        """
        Args:
            name (str): subject ID
            dir (str): path where the .h5 file is located, if `None`, `prefs.get('DATADIR')` is used
            file (str): load a subject from a filename. if `None`, ignored.
            new (bool): if True, a new file is made (a new file is made if one does not exist anyway)
            biography (dict): If making a new subject file, a dictionary with biographical data can be passed
            structure (:class:`.Subject_Schema`): Structure to use with this subject.
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

        if new or not self.file.exists():
            new = True
            self.structure = structure
            self.new_subject_file(biography, structure)
        else:
            with self._h5f as h5f:
                self.structure.make(h5f)


        h5f = self.open_hdf()

        # If subject has a protocol, load it to a dict
        self.current = None
        self.step    = None
        self.protocol_name = None
        if "/current" in h5f:
            # We load the info from 'current' but don't keep the node open
            # Stash it as a dict so better access from Python
            current_node = filenode.open_node(h5f.root.current)
            protocol_string = current_node.readall()
            self.current = json.loads(protocol_string)
            self.step = int(current_node.attrs['step'])
            self.protocol_name = current_node.attrs['protocol_name']
        elif not new:
            # if we're not being created for the first time, warn that there is no protocol assigned to the subject
            self.logger.warning('Subject has no protocol assigned!')

        # get last session number if we have it
        try:
            self.session = int(h5f.root.info._v_attrs['session'])
        except KeyError:
            self.session = None

        # We will get handles to trial and continuous data when we start running
        self.current_trial  = None

        # Is the subject currently running (ie. we expect data to be incoming)
        # Used to keep the subject object alive, otherwise we close the file whenever we don't need it
        self.running = False

        # We use a threading queue to dump data into a kept-alive h5f file
        self.data_queue = None
        self._thread = None
        self.did_graduate = threading.Event()

        # Every time we are initialized we stash the git hash
        history_row = h5f.root.history.hashes.row
        history_row['time'] = self.get_timestamp()
        try:
            history_row['hash'] = prefs.get('HASH')
            # FIXME: less implicit way of getting hash plz
        except AttributeError:
            history_row['hash'] = ''
        history_row.append()

        # we have to always open and close the h5f
        self.close_hdf(h5f)


    @property
    @contextmanager
    def _h5f(self) -> tables.file.File:
        """
        Context manager for access to hdf5 file.

        Examples:

            with self._h5f as h5f:
                # ... do hdf5 stuff

        Returns:
            function wrapped with contextmanager that will open the hdf file
        """

        # @contextmanager
        # def _h5f_context() -> tables.file.File:
        with self._lock:
            try:
                h5f = tables.open_file(str(self.file), mode="r+")
                yield h5f
            finally:
                h5f.close()
        # return _h5f_context()


    @property
    def info(self) -> Biography:
        """
        Subject biographical information

        Returns:
            dict
        """
        with self._h5f as h5f:
            info = h5f.get_node(self.structure.info.path)
            biodict = {}
            for k in info._v_attrs._f_list():
                biodict[k] = info._v_attrs[k]

        return Biography(**biodict)



    def open_hdf(self, mode='r+'):
        """
        Opens the hdf5 file.

        This should be called at the start of every method that access the h5 file
        and :meth:`~.Subject.close_hdf` should be called at the end. Otherwise
        the file will close and we risk file corruption.

        See the pytables docs
        `here <https://www.pytables.org/cookbook/threading.html>`_ and
        `here <https://www.pytables.org/FAQ.html#can-pytables-be-used-in-concurrent-access-scenarios>`_

        Args:
            mode (str): a file access mode, can be:

                * 'r': Read-only - no data can be modified.
                * 'w': Write - a new file is created (an existing file with the same name would be deleted).
                * 'a' Append - an existing file is opened for reading and writing, and if the file does not exist it is created.
                * 'r+' (default) - Similar to 'a', but file must already exist.

        Returns:
            :class:`tables.File`: Opened hdf file.
        """
        # TODO: Use a decorator around methods instead of explicitly calling
        with self._lock:
            return tables.open_file(self.file, mode=mode)

    def close_hdf(self, h5f):
        # type: (tables.file.File) -> None
        """
        Flushes & closes the open hdf file.
        Must be called whenever :meth:`~.Subject.open_hdf` is used.

        Args:
            h5f (:class:`tables.File`): the hdf file opened by :meth:`~.Subject.open_hdf`
        """
        with self._lock:
            h5f.flush()
            h5f.close()

    @classmethod
    def new(cls,
            bio:Biography,
            structure: Optional[Subject_Structure] = Subject_Structure(),
            path: Optional[Path] = None,
            ) -> 'Subject':
        """
        Create a new subject file, make its structure, and populate its :class:`~.data.models.biography.Biography` .


        Args:
            biography (:class:`~.data.models.biography.Biography`): A collection of biographical information
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

        h5f = tables.open_file(filename=str(path), mode='w')

        # make basic structure
        structure.make(h5f)

        info_node = h5f.get_node(structure.info.path)
        for k, v in bio.dict().items():
            info_node._v_attrs[k] = v

        # compatibility - double `id` as name
        info_node._v_attrs['name'] = bio.id

        # create initial values for task status
        task_node = h5f.get_node(structure.task.path)
        for k, v in Task_Status().dict().items():
            task_node._v_attrs[k] = v

        h5f.close()

        return Subject(name=bio.id, file=path)









    def new_subject_file(self, biography:Biography, structure: Subject_Structure):
        """
        Create a new subject file and make the general filestructure.

        If a file already exists, open it in append mode, otherwise create it.

        Args:
            biography (dict): Biographical details like DOB, mass, etc.
                Typically created by :class:`~.gui.New_Subject_Wizard.Biography_Tab`.
        """
        # If a file already exists, we open it for appending so we don't lose data.
        # For now we are assuming that the existing file has the basic structure,
        # but that's probably a bad assumption for full reliability
        if self.file.exists():
            h5f = self.open_hdf(mode='a')
        else:
            h5f = self.open_hdf(mode='w')

        structure.make(h5f)

        # Save biographical information as node attributes
        if biography:
            info_node = h5f.get_node(structure.info.path)
            for k, v in biography.dict():
                info_node._v_attrs[k] = v



        self.close_hdf(h5f)


    def update_biography(self, params):
        """
        Change or make a new biographical attribute, stored as
        attributes of the `info` group.

        Args:
            params (dict): biographical attributes to be updated.
        """
        h5f = self.open_hdf()
        for k, v in params.items():
            h5f.root.info._v_attrs[k] = v
        _ = self.close_hdf(h5f)

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
        if type == 'param':
            if not step:
                self.current[self.step][name] = value
            else:
                self.current[step][name] = value
            self.flush_current()
        elif type == 'step':
            self.step = int(value)
            self.flush_current()
        elif type == 'protocol':
            self.flush_current()


        # Check that we're all strings in here
        if not isinstance(type, str):
            type = str(type)
        if not isinstance(name, str):
            name = str(name)
        if not isinstance(value, str):
            value = str(value)

        # log the change
        h5f = self.open_hdf()
        history_row = h5f.root.history.history.row

        history_row['time'] = self.get_timestamp(simple=True)
        history_row['type'] = type
        history_row['name'] = name
        history_row['value'] = value
        history_row.append()

        _ = self.close_hdf(h5f)

    def assign_protocol(self, protocol:typing.Union[Path, str, typing.List[dict]],
                        step_n:int=0,
                        protocol_name:Optional[str]=None):
        """
        Assign a protocol to the subject.

        If the subject has a currently assigned task, stashes it with :meth:`~.Subject.stash_current`

        Creates groups and tables according to the data descriptions in the task class being assigned.
        eg. as described in :class:`.Task.TrialData`.

        Updates the history table.

        Args:
            protocol (Path, str, dict): the protocol to be assigned. Can be one of

                * the name of the protocol (its filename minus .json) if it is in `prefs.get('PROTOCOLDIR')`
                * filename of the protocol (its filename with .json) if it is in the `prefs.get('PROTOCOLDIR')`
                * the full path and filename of the protocol.
                * The protocol dictionary serialized to a string
                * the protocol as a list of dictionaries

            step_n (int): Which step is being assigned?
            protocol_name (str): If passing ``protocol`` as a dict, have to give a name to the protocol
        """
        # Protocol will be passed as a .json filename in prefs.get('PROTOCOLDIR')

        h5f = self.open_hdf()

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
                protocol = protocol.relative_to(prefs.get('PROTOCOLDIR'))
            if not protocol.exists():
                raise FileNotFoundError(f"Could not find protocol file {protocol}!")

            protocol_name = protocol.stem

            with open(protocol, 'r') as pfile:
                protocol = json.load(pfile)


        elif isinstance(protocol, list):
            if protocol_name is None:
                raise ValueError(f"If passed protocol as a list of dictionaries, need to also pass protocol_name")


        # check if this is the same protocol as we already have so we don't reset session number
        same_protocol = False
        if (protocol_name == self.protocol_name) and (step_n == self.step):
            same_protocol = True

        # Check if there is an existing protocol, archive it if there is.
        if "/current" in h5f:
            _ = self.close_hdf(h5f)
            self.stash_current()
            h5f = self.open_hdf()

        # Make filenode and save as serialized json
        current_node = filenode.new_node(h5f, where='/', name='current')
        current_node.write(json.dumps(protocol).encode('utf-8'))
        h5f.flush()

        # save some protocol attributes
        self.current = protocol
        current_node.attrs['protocol_name'] = protocol_name
        self.protocol_name = protocol_name
        current_node.attrs['step'] = step_n
        self.step = int(step_n)

        # always start out on session 0 on a new task
        # unless this is the same task as was already assigned
        if not same_protocol:
            h5f.root.info._v_attrs['session'] = 0
            self.session = 0

        # make protocol structure!
        protocol_structure = Protocol_Group(
            protocol_name=protocol_name,
            protocol=protocol,
            structure=self.structure
        )
        protocol_structure.make(h5f)

        _ = self.close_hdf(h5f)

        # Update history
        self.update_history(type='protocol', name=protocol_name, value=self.current)
        self.update_history(type='step',
                            name=self.current[self.step]['step_name'],
                            value=self.step)

    def flush_current(self):
        """
        Flushes the 'current' attribute in the subject object to the current filenode
        in the .h5

        Used to make sure the stored .json representation of the current task stays up to date
        with the params set in the subject object
        """

        h5f = self.open_hdf()
        h5f.remove_node('/current')
        current_node = filenode.new_node(h5f, where='/', name='current')
        current_node.write(json.dumps(self.current).encode('utf-8'))
        current_node.attrs['step'] = self.step
        current_node.attrs['protocol_name'] = self.protocol_name
        self.close_hdf(h5f)
        self.logger.debug('current protocol flushed')

    def stash_current(self):
        """
        Save the current protocol in the history group and delete the node

        Typically this is called when assigning a new protocol.

        Stored as the date that it was changed followed by its name if it has one
        """
        h5f = self.open_hdf()
        try:
            protocol_name = h5f.get_node_attr('/current', 'protocol_name')
            archive_name = '_'.join([self.get_timestamp(simple=True), protocol_name])
        except AttributeError:
            warnings.warn("protocol_name attribute couldn't be accessed, using timestamp to stash protocol")
            archive_name = self.get_timestamp(simple=True)

        archive_node = filenode.new_node(h5f, where='/history/past_protocols', name=archive_name)
        archive_node.write(json.dumps(self.current).encode('utf-8'))

        h5f.remove_node('/current')
        self.close_hdf(h5f)
        self.logger.debug('current protocol stashed')

    def prepare_run(self):
        """
        Prepares the Subject object to receive data while running the task.

        Gets information about current task, trial number,
        spawns :class:`~.tasks.graduation.Graduation` object,
        spawns :attr:`~.Subject.data_queue` and calls :meth:`~.Subject.data_thread`.

        Returns:
            Dict: the parameters for the current step, with subject id, step number,
                current trial, and session number included.
        """
        if self.current is None:
            e = RuntimeError('No task assigned to subject, cant prepare_run. use Subject.assign_protocol or protocol reassignment wizard in the terminal GUI')
            self.logger.exception(f"{e}")
            raise e

        # get step history
        try:
            step_df = self.get_step_history(use_history=True)
        except Exception as e:
            self.logger.exception(f"Couldnt get step history to trim data given to graduation objects, got exception {e}")
            step_df = None

        h5f = self.open_hdf()

        protocol_groups = Protocol_Group(
            protocol_name = self.protocol_name,
            protocol = self.current,
            structure = self.structure
        )
        group_name = protocol_groups.steps[self.step].path

        # Get current task parameters and handles to tables
        task_params = self.current[self.step]

        # tasks without TrialData will have some default table, so this should always be present
        trial_table = h5f.get_node(group_name, 'trial_data')

        ##################################3
        # first try and find some timestamp column to filter past data we give to the graduation object
        # in case the subject has been stepped back down to a previous stage, for example
        slice_start = 0
        try:
            ts_cols = [col for col in trial_table.colnames if 'timestamp' in col]
            # just use the first timestamp column
            if len(ts_cols) > 0:
                trial_ts = pd.DataFrame({'timestamp': trial_table.col(ts_cols[0])})
                trial_ts['timestamp'] = pd.to_datetime(trial_ts['timestamp'].str.decode('utf-8'))
            else:
                self.logger.warning(
                    'No timestamp column could be found in trial data, cannot trim data given to graduation objects')
                trial_ts = None

            if trial_ts is not None and step_df is not None:
                # see where, if any, the timestamp column is older than the last time the step was changed
                good_rows = np.where(trial_ts['timestamp'] >= step_df['timestamp'].iloc[-1])[0]
                if len(good_rows) > 0:
                    slice_start = np.min(good_rows)
                # otherwise if it's because we found no good rows but have trials,
                # we will say not to use them, otherwise we say not to use them by
                # slicing at the end of the table
                else:
                    slice_start = trial_table.nrows

        except Exception as e:
            self.logger.exception(
                f"Couldnt trim data given to graduation objects with step change history, got exception {e}")

        trial_tab = trial_table.read(start=slice_start)
        trial_tab_keys = tuple(trial_tab.dtype.fields.keys())

        ##############################

        # get last trial number and session
        try:
            self.current_trial = trial_tab['trial_num'][-1]+1
        except IndexError:
            if 'trial_num' not in trial_tab_keys:
                self.logger.info('No previous trials detected, setting current_trial to 0')
            self.current_trial = 0

        # should have gotten session from current node when we started
        # so sessions increment over the lifespan of the subject, even if
        # reassigned.
        if not self.session:
            try:
                self.session = trial_tab['session'][-1]
            except IndexError:
                if 'session' not in trial_tab_keys:
                    self.logger.warning('previous session couldnt be found, setting to 0')
                self.session = 0

        self.session += 1
        h5f.root.info._v_attrs['session'] = self.session
        h5f.flush()

        # prepare continuous data group and tables
        task_class = autopilot.get_task(task_params['task_type'])
        if hasattr(task_class, 'ContinuousData'):
            cont_group = h5f.get_node(group_name, 'continuous_data')
            try:
                session_group = h5f.create_group(cont_group, "session_{}".format(self.session))
            except tables.NodeError:
                pass # fine, already made it

        self.graduation = None
        if 'graduation' in task_params.keys():
            try:
                grad_type = task_params['graduation']['type']
                grad_params = task_params['graduation']['value'].copy()

                # add other params asked for by the task class
                grad_obj = autopilot.get('graduation', grad_type)

                if grad_obj.PARAMS:
                    # these are params that should be set in the protocol settings
                    for param in grad_obj.PARAMS:
                        #if param not in grad_params.keys():
                        # for now, try to find it in our attributes
                        # but don't overwrite if it already has what it needs in case
                        # of name overlap
                        # TODO: See where else we would want to get these from
                        if hasattr(self, param) and param not in grad_params.keys():
                            grad_params.update({param:getattr(self, param)})

                if grad_obj.COLS:
                    # these are columns in our trial table

                    # then give the data to the graduation object
                    for col in grad_obj.COLS:
                        try:
                            grad_params.update({col: trial_tab[col]})
                        except KeyError:
                            self.logger.warning('Graduation object requested column {}, but it was not found in the trial table'.format(col))

                #grad_params['value']['current_trial'] = str(self.current_trial) # str so it's json serializable
                self.graduation = grad_obj(**grad_params)
                self.did_graduate.clear()
            except Exception as e:
                self.logger.exception(f'Exception in graduation parameter specification, graduation is disabled.\ngot error: {e}')
        else:
            self.graduation = None

        self.close_hdf(h5f)

        # spawn thread to accept data
        self.data_queue = queue.Queue()
        self._thread = threading.Thread(target=self.data_thread, args=(self.data_queue,))
        self._thread.start()
        self.running = True

        # return a task parameter dictionary

        task = copy(self.current[self.step])
        task['subject'] = self.name
        task['step'] = int(self.step)
        task['current_trial'] = int(self.current_trial)
        task['session'] = int(self.session)
        return task

    def data_thread(self, queue):
        """
        Thread that keeps hdf file open and receives data while task is running.

        receives data through :attr:`~.Subject.queue` as dictionaries. Data can be
        partial-trial data (eg. each phase of a trial) as long as the task returns a dict with
        'TRIAL_END' as a key at the end of each trial.

        each dict given to the queue should have the `trial_num`, and this method can
        properly store data without passing `TRIAL_END` if so. I recommend being explicit, however.

        Checks graduation state at the end of each trial.

        Args:
            queue (:class:`queue.Queue`): passed by :meth:`~.Subject.prepare_run` and used by other
                objects to pass data to be stored.
        """
        h5f = self.open_hdf()

        task_params = self.current[self.step]
        step_name = task_params['step_name']

        # file structure is '/data/protocol_name/##_step_name/tables'
        group_name = f"/data/{self.protocol_name}/S{self.step:02d}_{step_name}"
        #try:
        trial_table = h5f.get_node(group_name, 'trial_data')
        trial_keys = trial_table.colnames
        trial_row = trial_table.row

        # try to get continuous data table if any
        cont_data = tuple()
        cont_tables = {}
        cont_rows = {}
        try:
            continuous_group = h5f.get_node(group_name, 'continuous_data')
            session_group = h5f.get_node(continuous_group, 'session_{}'.format(self.session))
            cont_data = continuous_group._v_attrs['data']

            cont_tables = {}
            cont_rows = {}
        except AttributeError:
            continuous_table = False

        # start getting data
        # stop when 'END' gets put in the queue
        for data in iter(queue.get, 'END'):
            # wrap everything in try because this thread shouldn't crash
            try:
                # if we get continuous data, this should be simple because we always get a whole row
                # there must be a more elegant way to check if something is a key and it is true...
                # yet here we are
                if 'continuous' in data.keys():
                    for k, v in data.items():
                        # if this isn't data that we're expecting, ignore it
                        if k not in cont_data:
                            continue

                        # if we haven't made a table yet, do it
                        if k not in cont_tables.keys():
                            # make atom for this data
                            try:
                                # if it's a numpy array...
                                col_atom = tables.Atom.from_type(v.dtype.name, v.shape)
                            except AttributeError:
                                temp_array = np.array(v)
                                col_atom = tables.Atom.from_type(temp_array.dtype.name, temp_array.shape)
                            # should have come in with a timestamp
                            # TODO: Log if no timestamp is received
                            try:
                                temp_timestamp_arr = np.array(data['timestamp'])
                                timestamp_atom = tables.Atom.from_type(temp_timestamp_arr.dtype.name,
                                                                       temp_timestamp_arr.shape)

                            except KeyError:
                                self.logger.warning('no timestamp sent with continuous data')
                                continue


                            cont_tables[k] = h5f.create_table(session_group, k, description={
                                k: tables.Col.from_atom(col_atom),
                                'timestamp': tables.Col.from_atom(timestamp_atom)
                            })

                            cont_rows[k] = cont_tables[k].row

                        cont_rows[k][k] = v
                        cont_rows[k]['timestamp'] = data['timestamp']
                        cont_rows[k].append()

                    # continue, the rest is for handling trial data
                    continue



                # Check if this is the same
                # if we've already recorded a trial number for this row,
                # and the trial number we just got is not the same,
                # we edit that row if we already have some data on it or else start a new row
                if 'trial_num' in data.keys():
                    if (trial_row['trial_num']) and (trial_row['trial_num'] is None):
                        trial_row['trial_num'] = data['trial_num']

                    if (trial_row['trial_num']) and (trial_row['trial_num'] != data['trial_num']):

                        # find row with this trial number if it exists
                        # this will return a list of rows with matching trial_num.
                        # if it's empty, we didn't receive a TRIAL_END and should create a new row
                        other_row = [r for r in trial_table.where("trial_num == {}".format(data['trial_num']))]

                        if len(other_row) == 0:
                            # proceed to fill the row below
                            trial_row.append()

                        elif len(other_row) == 1:
                            # update the row and continue so we don't double write
                            # have to be in the middle of iteration to use update()
                            for row in trial_table.where("trial_num == {}".format(data['trial_num'])):
                                for k, v in data.items():
                                    if k in trial_keys:
                                        row[k] = v
                                row.update()
                            continue

                        else:
                            # we have more than one row with this trial_num.
                            # shouldn't happen, but we dont' want to throw any data away
                            self.logger.warning('Found multiple rows with same trial_num: {}'.format(data['trial_num']))
                            # continue just for data conservancy's sake
                            trial_row.append()

                for k, v in data.items():
                    # some bug where some columns are not always detected,
                    # rather than failing out here, just log error
                    if k in trial_keys:
                        try:
                            trial_row[k] = v
                        except KeyError:
                            # TODO: Logging here
                            self.logger.warning("Data dropped: key: {}, value: {}".format(k, v))

                # TODO: Or if all the values have been filled, shouldn't need explicit TRIAL_END flags
                if 'TRIAL_END' in data.keys():
                    trial_row['session'] = self.session
                    if self.graduation:
                        # set our graduation flag, the terminal will get the rest rolling
                        did_graduate = self.graduation.update(trial_row)
                        if did_graduate is True:
                            self.did_graduate.set()
                    trial_row.append()
                    trial_table.flush()

                # always flush so that our row iteration routines above will find what they're looking for
                trial_table.flush()
            except Exception as e:
                # we shouldn't throw any exception in this thread, just log it and move on
                self.logger.exception(f'exception in data thread: {e}')

        self.close_hdf(h5f)

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
        puts 'END' in the data_queue, which causes :meth:`~.Subject.data_thread` to end.
        """
        self.data_queue.put('END')
        self._thread.join(5)
        self.running = False
        if self._thread.is_alive():
            self.logger.warning('Data thread did not exit')

    def to_csv(self, path, task='current', step='all'):
        """
        Export trial data to .csv

        Args:
            path (str): output path of .csv
            task (str, int):  not implemented, but in the future pull data from 'current' or other named task
            step (str, int, list, tuple): Step to select, see :meth:`.Subject.get_trial_data`
        """
        # TODO: Jonny just scratching out temporarily, doesn't have all features implemented
        df = self.get_trial_data(step=step)
        df['subject'] = self.name
        df.to_csv(path)
        print("""Subject {}
dataframe saved to:\n {}
========================
N Trials:   {}
N Sessions: {}""".format(self.name, path, df.shape[0], len(df.session.unique())))



    def get_trial_data(self,
                       step: typing.Union[int, list, str] = -1,
                       what: str ="data"):
        """
        Get trial data from the current task.

        Args:
            step (int, list, 'all'): Step that should be returned, can be one of

                * -1: most recent step
                * int: a single step
                * list of two integers eg. [0, 5], an inclusive range of steps.
                * string: the name of a step (excluding S##_)
                * 'all': all steps.

            what (str): What should be returned?

                * 'data' : Dataframe of requested steps' trial data
                * 'variables': dict of variables *without* loading data into memory

        Returns:
            :class:`pandas.DataFrame`: DataFrame of requested steps' trial data.
        """
        # step= -1 is just most recent step,
        # step= int is an integer specified step
        # step= [n1, n2] is from step n1 to n2 inclusive
        # step= 'all' or anything that isn't an int or a list is all steps
        h5f = self.open_hdf()
        group_name = "/data/{}".format(self.protocol_name)
        group = h5f.get_node(group_name)
        step_groups = sorted(group._v_children.keys())

        if step == -1:
            # find the last trial step with data
            for step_name in reversed(step_groups):
                if group._v_children[step_name].trial_data.attrs['NROWS']>0:
                    step_groups = [step_name]
                    break
        elif isinstance(step, int):
            if step > len(step_groups):
                ValueError('You provided a step number ({}) greater than the number of steps in the subjects assigned protocol: ()'.format(step, len(step_groups)))
            step_groups = [step_groups[step]]

        elif isinstance(step, str) and step != 'all':

            # since step names have S##_ prepended in the hdf5 file,
            # but we want to be able to call them by their human readable name,
            # have to make sure we have the right form
            _step_groups = [s for s in step_groups if s == step]
            if len(_step_groups) == 0:
                _step_groups = [s for s in step_groups if step in s]
            step_groups = _step_groups

        elif isinstance(step, list):
            if isinstance(step[0], int):
                step_groups = step_groups[int(step[0]):int(step[1])]
            elif isinstance(step[0], str):
                _step_groups = []
                for a_step in step:
                    step_name = [s for s in step_groups if s==a_step]
                    if len(step_name) == 0:
                        step_name = [s for s in step_groups if a_step in s]
                    _step_groups.extend(step_name)

                step_groups = _step_groups
        print('step groups:')
        print(step_groups)

        if what == "variables":
            return_data = {}

        for step_key in step_groups:
            step_n = int(step_key[1:3]) # beginning of keys will be 'S##'
            step_tab = group._v_children[step_key]._v_children['trial_data']
            if what == "data":
                step_df = pd.DataFrame(step_tab.read())
                step_df['step'] = step_n
                step_df['step_name'] = step_key
                try:
                    return_data = return_data.append(step_df, ignore_index=True)
                except NameError:
                    return_data = step_df

            elif what == "variables":
                return_data[step_key] = step_tab.coldescrs


        self.close_hdf(h5f)

        return return_data

    def apply_along(self, along='session', step=-1):
        h5f = self.open_hdf()
        group_name = "/data/{}".format(self.protocol_name)
        group = h5f.get_node(group_name)
        step_groups = sorted(group._v_children.keys())

        if along == "session":
            if step == -1:
                # find the last trial step with data
                for step_name in reversed(step_groups):
                    if group._v_children[step_name].trial_data.attrs['NROWS'] > 0:
                        step_groups = [step_name]
                        break
            elif isinstance(step, int):
                if step > len(step_groups):
                    ValueError(
                        'You provided a step number ({}) greater than the number of steps in the subjects assigned protocol: ()'.format(
                            step, len(step_groups)))
                step_groups = [step_groups[step]]

            for step_key in step_groups:
                step_n = int(step_key[1:3])  # beginning of keys will be 'S##'
                step_tab = group._v_children[step_key]._v_children['trial_data']
                step_df = pd.DataFrame(step_tab.read())
                step_df['step'] = step_n
                yield step_df





    def get_step_history(self, use_history=True):
        """
        Gets a dataframe of step numbers, timestamps, and step names
        as a coarse view of training status.

        Args:
            use_history (bool): whether to use the history table or to reconstruct steps and dates from the trial table itself.
                compatibility fix for old versions that didn't stash step changes when the whole protocol was updated.

        Returns:
            :class:`pandas.DataFrame`

        """
        h5f = self.open_hdf()
        if use_history:
            history = h5f.root.history.history
            step_df = pd.DataFrame(history.read())
            if step_df.shape[0] == 0:
                return None
            # encode as unicode
            # https://stackoverflow.com/a/63028569/13113166
            for col, dtype in step_df.dtypes.items():
                if dtype == np.object:  # Only process byte object columns.
                    step_df[col] = step_df[col].apply(lambda x: x.decode("utf-8"))

            # filter to step only
            step_df = step_df[step_df['type'] == 'step'].drop('type', axis=1)
            # rename and retype
            step_df = step_df.rename(columns={
                'value': 'step_n',
                'time': 'timestamp',
                'name': 'name'})

            step_df['timestamp'] = pd.to_datetime(step_df['timestamp'],
                                                  format='%y%m%d-%H%M%S')
            step_df['step_n'] = pd.to_numeric(step_df['step_n'])


        else:
            group_name = "/data/{}".format(self.protocol_name)
            group = h5f.get_node(group_name)
            step_groups = sorted(group._v_children.keys())

            # find the last trial step with data
            for step_name in reversed(step_groups):
                if group._v_children[step_name].trial_data.attrs['NROWS']>0:
                    step_groups = [step_name]
                    break

            # Iterate through steps, find first timestamp, use that.
            for step_key in step_groups:
                step_n = int(step_key[1:3])  # beginning of keys will be 'S##'
                step_name = self.current[step_n]['step_name']
                step_tab = group._v_children[step_key]._v_children['trial_data']
                # find name of column that is a timestamp
                colnames = step_tab.cols._v_colnames
                try:
                    ts_column = [col for col in colnames if "timestamp" in col][0]
                    ts = step_tab.read(start=0, stop=1, field=ts_column)

                except IndexError:
                    self.logger.warning('No Timestamp column found, only returning step numbers and named that were reached')
                    ts = 0

                step_df = pd.DataFrame(
                    {'step_n':step_n,
                     'timestamp':ts,
                     'name':step_name
                    })
                try:
                    return_df = return_df.append(step_df, ignore_index=True)
                except NameError:
                    return_df = step_df

            step_df = return_df

        self.close_hdf(h5f)
        return step_df

    def get_timestamp(self, simple=False):
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

        h5f = self.open_hdf()
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

        self.close_hdf(h5f)
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

        h5f = self.open_hdf()
        weight_table = h5f.root.history.weights
        # there should only be one matching row since it includes seconds
        for row in weight_table.where('date == b"{}"'.format(date)):
            row[col_name] = new_value
            row.update()

        self.close_hdf(h5f)


    def update_weights(self, start=None, stop=None):
        """
        Store either a starting or stopping mass.

        `start` and `stop` can be passed simultaneously, `start` can be given in one
        call and `stop` in a later call, but `stop` should not be given before `start`.

        Args:
            start (float): Mass before running task in grams
            stop (float): Mass after running task in grams.
        """
        h5f = self.open_hdf()
        if start is not None:
            weight_row = h5f.root.history.weights.row
            weight_row['date'] = self.get_timestamp(simple=True)
            weight_row['session'] = self.session
            weight_row['start'] = float(start)
            weight_row.append()
        elif stop is not None:
            # TODO: Make this more robust - don't assume we got a start weight
            h5f.root.history.weights.cols.stop[-1] = stop
        else:
            self.logger.warning("Need either a start or a stop weight")

        _ = self.close_hdf(h5f)

    def graduate(self):
        """
        Increase the current step by one, unless it is the last step.
        """
        if len(self.current)<=self.step+1:
            self.logger.warning('Tried to graduate from the last step!\n Task has {} steps and we are on {}'.format(len(self.current), self.step+1))
            return

        # increment step, update_history should handle the rest
        step = self.step+1
        name = self.current[step]['step_name']
        self.update_history('step', name, step)

