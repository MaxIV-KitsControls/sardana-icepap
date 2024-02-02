import datetime
import enum
import json
import threading
import time

import numpy as np
from sardana import State
from sardana.pool.controller import (
    MotorController,
    Type,
    Description,
    DefaultValue,
    Access,
    DataAccess,
    FSet,
    FGet,
    Memorize,
    Memorized,
    NotMemorized,
    MemorizedNoInit,
)
from sardana.sardanaevent import EventType
from tango import DeviceProxy, Database, Util

from icepap.controller import IcePAPController


class TrajectoryState(enum.Enum):
    IDLE = 0
    MOVING_TO_INTERMEDIATE = 1
    MOVING_TO_FINAL = 2
    FAULT = 3


class TrajectoryChangeError(RuntimeError):
    """Failure while changing to new trajectory."""


class MoveToTargetError(RuntimeError):
    """Not all motors reached the target position."""


class ReadPositionError(RuntimeError):
    """Position could not be determined from motors."""


class IPAPTrajCtrl(MotorController):
    """
    This class implements the base functionality of a trajectory motor controller.

    During init, the controller creates the tables for motor positions versus trajectory unit.
    In order to do this, it needs the values of all attributes as well as
    the parameters for the underlying IcePAP motors.
    To ensure that this information is available, the attributes are set as MemorizedNoInit,
    meaning that Tango will not write the memorized values after init.
    Instead the values are read from the database during the init procedure.

    The parameters for underlying motors are also needed.
    If the motor devices have already been created, then the values are read from their attributes.
    If not, the memorized attribute values are read from the Tango database.
    """

    gender = "Motor"
    model = "Icepap"
    organization = "MaxIV"
    MaxDevice = 1

    # The properties used to connect to the IcePAP motor controller

    ctrl_properties = {
        "Host": {
            Type: str,
            Description: "The host name",
        },
        "Port": {
            Type: int,
            Description: "The port number",
            DefaultValue: 5000,
        },
        "Timeout": {
            Type: int,
            Description: "Connection timeout",
            DefaultValue: 3,
        },
        "Num_points": {
            Type: int,
            Description: "Trajectory number of points",
            DefaultValue: 1000,
        },
        "Tolerance": {
            Type: float,
            Description: "Parameter error on the position of each motor.",
            DefaultValue: 1.0e-3,
        },
        "Tolerance_steps": {
            Type: int,
            Description: (
                "Max parameter position error " + "on the position of each motor."
            ),
            DefaultValue: 10,
        },
        "Tolerance_pos_diff": {
            Type: float,
            Description: (
                "Max diff allowed between nearest trajectory position and "
                + "the position to move onto the trajectory at. "
                + "Affects MoveOntoTrajectoryAt and UnsafeMoveOntoTrajectoryAt attributes and "
                + "DefinePosition command."
            ),
            DefaultValue: 1.0,
        },
    }

    ctrl_attributes = {
        "ActiveTrajectory": {
            Type: ((float,),),
            Description: "Last values loaded into IcePAP trajectory table.\n"
            "Dimensions: (M+2) x N, where M is number of motors, and "
            "N is the number of rows in the table.\n"
            "N may be as small as 2, for some direct movements.\n"
            "  attr[0]: parameter values [user units],\n"
            "  attr[1]: axis positions for motor 1 [user units],\n"
            "  attr[2]: axis positions for motor 2 [user units],\n"
            "  attr[M]: axis positions for motor M [user units],\n"
            "  attr[M+1]: parameter values [transport units if transport mode,"
            " else user units].",
            Access: DataAccess.ReadOnly,
            FGet: "get_ActiveTrajectory",
        },
        "IntermediateTrajectoryUnvalidated": {
            Type: ((float,),),
            Description: "Last values considered for intermediate movement trajectory "
            "table BEFORE validation for 'unsafe' bounds.\n"
            "Dimensions: (M+2) x N, where M is number of motors, and "
            "N is the number of rows in the table.\n"
            "N may be as small as 2, for some direct movements.\n"
            "  attr[0]: parameter values [user units],\n"
            "  attr[1]: axis positions for motor 1 [user units],\n"
            "  attr[2]: axis positions for motor 2 [user units],\n"
            "  attr[M]: axis positions for motor M [user units],\n"
            "  attr[M+1]: parameter values [transport units if transport mode,"
            " else user units].",
            Access: DataAccess.ReadOnly,
            FGet: "get_IntermediateTrajectoryUnvalidated",
        },
        "IntermediateTrajectoryValidated": {
            Type: ((float,),),
            Description: "Last values used for intermediate movement trajectory table "
            "AFTER validation - no unsafe values.\n"
            "Dimensions: (M+2) x N, where M is number of motors, and "
            "N is the number of rows in the table.\n"
            "N may be as small as 2, for some direct movements.\n"
            "  attr[0]: parameter values [user units],\n"
            "  attr[1]: axis positions for motor 1 [user units],\n"
            "  attr[2]: axis positions for motor 2 [user units],\n"
            "  attr[M]: axis positions for motor M [user units],\n"
            "  attr[M+1]: parameter values [transport units if transport mode,"
            " else user units].",
            Access: DataAccess.ReadOnly,
            FGet: "get_IntermediateTrajectoryValidated",
        },
        "FinalTrajectoryUnvalidated": {
            Type: ((float,),),
            Description: "Last values considered for final movement trajectory "
            "table BEFORE validation for 'unsafe' bounds.\n"
            "Dimensions: (M+2) x N, where M is number of motors, and "
            "N is the number of rows in the table.\n"
            "  attr[0]: parameter values [user units],\n"
            "  attr[1]: axis positions for motor 1 [user units],\n"
            "  attr[2]: axis positions for motor 2 [user units],\n"
            "  attr[M]: axis positions for motor M [user units],\n"
            "  attr[M+1]: parameter values [transport units if transport mode,"
            " else user units].",
            Access: DataAccess.ReadOnly,
            FGet: "get_FinalTrajectoryUnvalidated",
        },
        "FinalTrajectoryValidated": {
            Type: ((float,),),
            Description: "Last values used for final movement trajectory table "
            "AFTER validation - no unsafe values.\n"
            "Dimensions: (M+2) x N, where M is number of motors, and "
            "N is the number of rows in the table.\n"
            "  attr[0]: parameter values [user units],\n"
            "  attr[1]: axis positions for motor 1 [user units],\n"
            "  attr[2]: axis positions for motor 2 [user units],\n"
            "  attr[M]: axis positions for motor M [user units],\n"
            "  attr[M+1]: parameter values [transport units if transport mode,"
            " else user units].",
            Access: DataAccess.ReadOnly,
            FGet: "get_FinalTrajectoryValidated",
        },
        "TrajectoryDetails": {
            Type: str,
            Description: "Dict with details about the various trajectory table "
            "attributes.  JSON encoded string.",
            Access: DataAccess.ReadOnly,
            FGet: "get_TrajectoryDetails",
        },
    }

    axis_attributes = {
        "MaxVelocity": {
            Type: float,
            Access: DataAccess.ReadOnly,
            FGet: "get_MaxVelocity",
            Memorize: Memorized,
        },
        "DistanceToTrajectory": {
            Type: [float],
            Access: DataAccess.ReadOnly,
            FGet: "get_DistanceToTrajectory",
            Memorize: NotMemorized,
        },
        "NearestTrajectoryPosition": {
            Type: float,
            Access: DataAccess.ReadOnly,
            FGet: "get_NearestTrajectoryPosition",
            Memorize: NotMemorized,
        },
        "ParamChangeVel": {
            Type: float,
            Access: DataAccess.ReadWrite,
            FGet: "get_ParamChangeVel",
            FSet: "set_ParamChangeVel",
            Memorize: Memorized,
        },
        "TransportMode": {
            Type: bool,
            Access: DataAccess.ReadWrite,
            FGet: "get_TransportMode",
            FSet: "set_TransportMode",
            Memorize: MemorizedNoInit,
        },
        "MoveOntoTrajectoryAt": {
            Type: float,
            Access: DataAccess.ReadWrite,
            FGet: "get_MoveOntoTrajectoryAt",
            FSet: "set_MoveOntoTrajectoryAt",
            Memorize: NotMemorized,
        },
        "UnsafeMoveOntoTrajectoryAt": {
            Type: float,
            Access: DataAccess.ReadWrite,
            FGet: "get_UnsafeMoveOntoTrajectoryAt",
            FSet: "set_UnsafeMoveOntoTrajectoryAt",
            Memorize: NotMemorized,
        },
        "BackOffLimitSwitch": {
            Type: bool,
            Access: DataAccess.ReadWrite,
            FGet: "get_BackOffLimitSwitch",
            FSet: "set_BackOffLimitSwitch",
            Memorize: NotMemorized,
        },
        "TrajectoryState": {
            Type: bool,
            Access: DataAccess.ReadOnly,
            FGet: "get_TrajectoryState",
            Memorize: NotMemorized,
        },
        "ParameterLimits": {
            Type: str,
            Access: DataAccess.ReadOnly,
            FGet: "get_ParameterLimits",
            Memorize: NotMemorized,
        },
        "MinTrajRange": {
            Type: float,
            Access: DataAccess.ReadWrite,
            FGet: "get_MinTrajRange",
            FSet: "set_MinTrajRange",
            Memorize: NotMemorized,
        },
        "MaxTrajRange": {
            Type: float,
            Access: DataAccess.ReadWrite,
            FGet: "get_MaxTrajRange",
            FSet: "set_MaxTrajRange",
            Memorize: NotMemorized,
        },
    }

    # Initial step size for transport mode table in dummy transport units
    TRANSPORT_MODE_INITIAL_SLOWEST_STEP = 0.00001
    # Increase transport step size by this factor to avoid moving faster
    # than motors can.  E.g., 1.1 = 10% margin.
    TRANSPORT_MODE_SAFETY_MARGIN_SCALE_FACTOR = 1.1

    def __init__(self, inst, props, *args, **kwargs):

        MotorController.__init__(self, inst, props, *args, **kwargs)
        self.attributes = {}

        # Icepap controller connection
        self._log.debug("Init icepap connection")
        self._ipap = IcePAPController(
            self.Host, self.Port, self.Timeout, auto_axes=True
        )

        self._step_per_unit = None

        if not hasattr(self, "motor_roles"):
            raise NotImplementedError("'motor_roles' must be defined in the controller")
        if not hasattr(self, "master_motor"):
            raise NotImplementedError(
                "'master_motor' must be defined in the controller"
            )
        if not hasattr(self, "trajectory_parameters"):
            raise NotImplementedError(
                "'trajectory_parameters' must be defined in the controller"
            )
        self.motors = {}
        for mot in self.motor_roles:
            self.motors[mot] = {}
            self.motors[mot]["alias"] = getattr(self, mot + "_name")

        self._log.debug("Look up motor axes")
        self.aliases = self._get_motors_axis(self.motors)
        self._trajdic = None
        self._ipap.add_aliases(self.aliases)
        self._tolerance = self.Tolerance
        self._tolerance_steps = self.Tolerance_steps
        self._tolerance_pos_diff = self.Tolerance_pos_diff
        self._velocity = None
        self._acctime = 1
        self._max_vel = None
        self.active_parameters_dict = {}
        self._log.debug("Reading stored parameter values")
        for par in self.trajectory_parameters:
            try:
                self.active_parameters_dict[par] = float(
                    self._get_memorized_attribute(par)
                )
                self._log.info("using memorized value for {}".format(par))
            except Exception:
                try:
                    self.active_parameters_dict[par] = self.axis_attributes[par][
                        DefaultValue
                    ]
                    self._log.info("using default value for {}".format(par))
                except Exception:
                    self.active_parameters_dict[par] = 0.0
                    self._log.warning(
                        "failed to get an initial value for {}".format(par)
                    )
        try:
            transp_str = self._get_memorized_attribute("TransportMode")
            if transp_str.lower() == "true":
                self.transport_mode = True
            elif transp_str.lower() == "false":
                self.transport_mode = False
            else:
                self._log.warning(
                    "Invalid memorized value for TransportMode, defaulting to False"
                )
                self.transport_mode = False
        except Exception:
            self._log.warning(
                "Failed to read memorized value for TransportMode, defaulting to False"
            )
            self.transport_mode = False

        self.traj_coord_min, self.traj_coord_max = self.calc_limits(
            self.active_parameters_dict
        )
        self.next_parameters_dict = dict(self.active_parameters_dict)
        self.param_change_vel = 0.5
        self.move_on_write = True
        self.lock = threading.RLock()
        self._trajdic = self._calc_trajectories()
        self.thread = None
        self.move_onto_trajectory_at = 0.0
        self._changing_traj_state = TrajectoryState.IDLE
        self._changing_traj_error_message = ""
        self._init_trajectory_reporting_attributes()

        self._log.info("IPAPTrajCtrl init done")

    def _get_memorized_attribute(self, attr):
        """
        Read the memorized value of an attribute for a controller axis.
        We are currently setting up the controller, and the Tango device
        for the axis has not yet been created.
        Thus we need to search the database for the right device
        of class "Motor" by checking the ctrl_id property.
        Once found, read the memorized value.
        Since this controller only supports one axis,
        we skip checking the axis number.
        """
        db = Util.instance().get_database()
        sardana_id = str(self._getPoolController().get_id()).lower()
        devs = db.get_device_name("*", "Motor").value_string
        devname = ""
        for dev in devs:
            ctrl_id = db.get_device_property(dev, "ctrl_id")["ctrl_id"][0]
            if ctrl_id.lower() == sardana_id:
                devname = dev
                break
        attrvalue = db.get_device_attribute_property(devname, attr)[attr]["__value"][0]
        return attrvalue

    def _get_motors_axis(self, motors):
        """
        Get motor axis numbers from sardana.
        """
        db = Database()
        aliases = {}
        for mot, motdata in motors.items():
            devname = motdata["alias"]
            fullname = db.get_device_from_alias(devname)
            axis = db.get_device_property(fullname, "axis")
            aliases[mot] = int(axis["axis"][0])
        return aliases

    def _get_motors_param(self):
        """
        Get parameters for all motors from sardana.
        When the Pool starts it may happen that we are trying to read
        from the motor devices before they have been created.
        Then reading the values via a DeviceProxy fails.
        In this case, fall back to reading the memorized values from the Tango database.
        """
        for mot, motdata in self.motors.items():
            try:
                motor = DeviceProxy(motdata["alias"])
                motdata["spu"] = motor.Step_per_unit
                motdata["offset"] = motor.Offset
                motdata["sign"] = motor.Sign
                self._log.info("read motor parameters for {}".format(motdata["alias"]))
            except Exception:
                # load from memorized values
                db = Util.instance().get_database()
                devname = db.get_device_from_alias(motdata["alias"])
                try:
                    motdata["spu"] = float(
                        db.get_device_attribute_property(devname, "Step_per_unit")[
                            "Step_per_unit"
                        ]["__value"][0]
                    )
                except Exception:
                    self._log.error(
                        (
                            "Unable to read memorized value " + "for {}/Step_per_unit"
                        ).format(motdata["alias"])
                    )
                try:
                    motdata["offset"] = float(
                        db.get_device_attribute_property(devname, "Offset")["Offset"][
                            "__value"
                        ][0]
                    )
                except Exception:
                    self._log.error(
                        "Unable to read memorized value for {}/Offset".format(
                            motdata["alias"]
                        )
                    )
                try:
                    motdata["sign"] = float(
                        db.get_device_attribute_property(devname, "Sign")["Sign"][
                            "__value"
                        ][0]
                    )
                except Exception:
                    self._log.error(
                        "Unable to read memorized value for {}/Sign".format(
                            motdata["alias"]
                        )
                    )

    def _init_trajectory_reporting_attributes(self):
        now_str = datetime.datetime.now().astimezone().isoformat()
        self._traj_details = {
            "ActiveTrajectory": {
                "_description": "Initial trajectory",
                "_table": self._get_trajectory_table_from_icepap(),
                "_time": now_str,
            },
            "IntermediateTrajectoryUnvalidated": {
                "_description": "Unset",
                "_table": [[]],
                "_time": now_str,
            },
            "IntermediateTrajectoryValidated": {
                "_description": "Unset",
                "_table": [[]],
                "_time": now_str,
            },
            "FinalTrajectoryUnvalidated": {
                "_description": "Unset",
                "_table": [[]],
                "_time": now_str,
            },
            "FinalTrajectoryValidated": {
                "_description": "Unset",
                "_table": [[]],
                "_time": now_str,
            },
        }

    def _update_traj_details(self, trajdic, table, validated):
        """Update various trajectory fields, depending on movement type and validity."""
        fields_to_update = []
        if self._changing_traj_state in (
            TrajectoryState.MOVING_TO_INTERMEDIATE,
            TrajectoryState.MOVING_TO_FINAL,
        ):
            suffix = "Validated" if validated else "Unvalidated"
            movement_type = self._changing_traj_state.name.split("_")[-1]
            field = f"{movement_type.capitalize()}Trajectory{suffix}"
            fields_to_update.append(field)
        if validated:
            fields_to_update.append("ActiveTrajectory")

        for field in fields_to_update:
            self._traj_details[field]["_description"] = trajdic["_description"]
            self._traj_details[field]["_table"] = table
            now_str = datetime.datetime.now().astimezone().isoformat()
            self._traj_details[field]["_time"] = now_str

    def _units2steps(self, motor_name, pos):
        """
        Convert units to motor steps.
        """
        spu = self.motors[motor_name]["spu"]
        offset = self.motors[motor_name]["offset"]
        sign = self.motors[motor_name]["sign"]
        return (pos - offset) * spu * sign

    def _steps2units(self, motor_name, pos):
        """
        Convert motor steps to units.
        """
        spu = self.motors[motor_name]["spu"]
        offset = self.motors[motor_name]["offset"]
        sign = self.motors[motor_name]["sign"]
        return pos / (spu * sign) + offset

    def calc_positions(self, traj_coord, parameters):
        """
        Calculate a set of motor positions in units for one position.
        """
        raise NotImplementedError("calc_positions must be defined in the controller")

    def calc_limits(self, new_parameters_dict):
        """
        Update the parameter limits for the new parameter values.
        Must be overriden!
        """
        raise NotImplementedError("calc_limits must be defined in the controller")

    def calc_parameter_limits(self):
        """
        Get a dictionary with the limits for all parameter values.
        The default implementation returns None for both min and max, override as needed.
        """
        limits = {}
        for param in self.trajectory_parameters:
            limits[param] = {}
            limits[param]["min"] = None
            limits[param]["max"] = None
        try:
            minpos = self._trajdic["_traj_pos"][0]
            maxpos = self._trajdic["_traj_pos"][-1]
        except (KeyError, IndexError):
            minpos = None
            maxpos = None
        limits["Position"] = {"min": minpos, "max": maxpos}
        return limits

    def _calc_trajectory_coords(self, param_min, param_max, nbrpoints):
        return np.linspace(param_min, param_max, num=nbrpoints)

    def _calc_trajectories(self, parameters=None):
        """
        Calculate a new trajectory given the relevant parameters.
        """
        if parameters is None:
            self.traj_coord_min, self.traj_coord_max = self.calc_limits(
                self.active_parameters_dict
            )
            parameters = [[self.traj_coord_min, self.traj_coord_max]] + [
                self.active_parameters_dict[par] for par in self.trajectory_parameters
            ]
        self._log.info("make trajectory, " + str(parameters))
        self._get_motors_param()
        all_parameters = []
        trajectory_pos = None
        description = "Normal trajectory, "
        traj_safe = True
        for n, param in enumerate(parameters):
            if isinstance(param, (list, tuple)):
                p_array = np.linspace(param[0], param[1], num=self.Num_points)
                if n == 0:
                    p_array = self._calc_trajectory_coords(
                        param[0], param[1], self.Num_points
                    )
                    trajectory_pos = p_array
                elif trajectory_pos is None:
                    description = "Temp parameter change trajectory, "
                    trajectory_pos = np.linspace(0, 100, num=self.Num_points)
            else:
                p_array = param * np.ones(self.Num_points)
            p_desc = "{}-{}, ".format(p_array[0], p_array[-1])
            description = description + p_desc
            all_parameters.append(p_array)
        all_positions = []
        for n in range(0, len(self.motors)):
            all_positions.append([])
        for params in zip(*all_parameters):
            trajcoord = params[0]
            pars = params[1:]
            positions = self.calc_positions(trajcoord, pars)
            if not self.check_positions_safe(positions):
                traj_safe = False
            for pos, pos_array in zip(positions, all_positions):
                pos_array.append(pos)
        # create dictionary with trajectories
        trajdic = {}
        trajdic["_description"] = description
        trajdic["_traj_pos"] = trajectory_pos
        trajdic["_parameters"] = all_parameters
        trajdic["_is_safe"] = traj_safe
        for n, pos_array in enumerate(all_positions):
            mot = self.motor_roles[n]
            trajdic[mot] = np.rint(self._units2steps(mot, np.array(pos_array)))
        self._log.info(self.compile_trajdic_info(trajdic))
        transport_table = self._make_transport_table(trajdic)
        trajdic["_transport"] = transport_table
        return trajdic

    def compile_trajdic_info(self, trajdic):
        """
        Create a message describing a trajectory.
        Mainly for the status message.
        """
        res = "Current trajectory:\n"
        if self.transport_mode:
            res = res + "Transport mode ON, velocity is not constant.\n"
        else:
            res = res + "Transport mode OFF, velocity is constant in units/s.\n"
        res = res + trajdic["_description"] + "\n"
        for motor in trajdic.keys():
            if motor == "_traj_pos" or not motor.startswith("_"):
                npoints = len(trajdic[motor])
                line = (
                    "Motor: {:>10s}  Points: {:4d}  "
                    "Start: {:10.2f}  End: {:10.2f}\n".format(
                        motor, npoints, trajdic[motor][0], trajdic[motor][-1]
                    )
                )
                res = res + line
        return res

    def _get_nearest_trajectory_position(self, trajdic=None):
        """
        Get the trajectory position closest to the current master motor position.
        """
        if trajdic is None:
            trajdic = self._trajdic
        traj_pos = trajdic["_traj_pos"]
        traj_master = trajdic[self.master_motor]
        master_pos = self._ipap.get_pos(self.master_motor)[0]
        idx_min = (np.abs(traj_master - master_pos)).argmin()
        nearest_pos = traj_pos[idx_min]
        return nearest_pos

    def _get_trajectory_positions(self, traj_coord, trajdic=None):
        """
        Get motor positions in steps for a given position on a trajectory.
        """
        traj_pos = []
        if trajdic is None:
            trajdic = self._trajdic
        traj_energy = trajdic["_traj_pos"]
        for mot in self.motor_roles:
            pos_steps = np.interp(traj_coord, traj_energy, trajdic[mot])
            traj_pos.append(pos_steps)
        return traj_pos

    def _get_trajectory_positions_units(self, traj_coord, trajdic=None):
        """
        Get motor positions in units for a given position on a trajectory.
        """
        pos_steps = self._get_trajectory_positions(traj_coord, trajdic=trajdic)
        traj_pos = []
        for (n, mot) in enumerate(self.motor_roles):
            traj_pos.append(self._steps2units(mot, pos_steps[n]))
        return traj_pos

    def get_distance_to_trajectory(self, trajdic=None):
        """
        For when motors are not synced: Get the how far the motors
        need to move (in units) to reach the closest point on the current
        desired trajectory.
        """
        if trajdic is None:
            trajdic = self._trajdic
        traj_energy = self._get_nearest_trajectory_position(trajdic=trajdic)
        traj_positions = self._get_trajectory_positions(traj_energy, trajdic=trajdic)
        motor_positions = self._ipap.get_pos(list(self.motor_roles))
        dist_steps = [a - b for a, b in zip(traj_positions, motor_positions)]
        dist_units = []
        for n, mot in enumerate(self.motor_roles):
            spu = self.motors[mot]["spu"]
            sign = self.motors[mot]["sign"]
            pos_units = dist_steps[n] / (spu * sign)
            dist_units.append(pos_units)
        return dist_units

    def _make_direct_trajectory_to(self, new_positions_steps):
        """
        Make a simple linear trajectory to move from current motor positions to
        a set of new positions. Trajectory position 0 to 100.
        """
        new_positions_units = []
        for mot, pos in zip(self.motor_roles, new_positions_steps):
            new_positions_units.append(self._steps2units(mot, pos))
        new_pos_safe = True
        if not self.check_positions_safe(new_positions_units):
            new_pos_safe = False

        with self.lock:
            curr_positions = self._ipap.get_pos(list(self.motor_roles))
        positions = np.array([0.0, 100.0])
        all_positions = []
        for curr_pos, new_pos in zip(curr_positions, new_positions_steps):
            all_positions.append([curr_pos, new_pos])

        # create dictionary with trajectories
        trajdic = {}
        trajdic["_description"] = "Temporary direct trajectory"
        trajdic["_traj_pos"] = positions
        trajdic["_parameters"] = None
        trajdic["_is_safe"] = new_pos_safe
        for n, pos_array in enumerate(all_positions):
            mot = self.motor_roles[n]
            trajdic[mot] = np.rint(np.array(pos_array))
        transport_table = self._make_transport_table(trajdic)
        trajdic["_transport"] = transport_table
        self._log.info(self.compile_trajdic_info(trajdic))
        return trajdic

    def check_positions_safe(self, positions):
        """
        Check if a combination of positions is safe.
        Default is always safe, override if needed!
        """
        return True

    def move_to_new_trajectory(self, new_parameters_dict, keep_traj_pos=True):
        """
        Start a move towards a new trajectory. The move is handled by a
        separate thread, so this function returns right away.
        """
        if self.thread is not None:
            if self.thread.is_alive():
                raise Exception("Thread already running!")
        try:
            param_min, param_max = self.calc_limits(new_parameters_dict)
            active_par = [
                self.active_parameters_dict[par] for par in self.trajectory_parameters
            ]
            new_par = [new_parameters_dict[par] for par in self.trajectory_parameters]
            new_trajdic = self._calc_trajectories(
                parameters=([param_min, param_max],) + tuple(new_par)
            )
            if keep_traj_pos:
                # Keeping trajectory position
                with self.lock:
                    current_pos = self._transport_to_unit(
                        self._ipap[self.master_motor].parpos
                    )
                new_traj_coord = current_pos
                intermediate_trajdic = self._calc_trajectories(
                    parameters=(current_pos,) + tuple(zip(active_par, new_par))
                )
            else:
                # Keeping master motor pos
                new_traj_coord = self._get_nearest_trajectory_position(new_trajdic)
                new_positions = self._get_trajectory_positions(
                    new_traj_coord, trajdic=new_trajdic
                )
                intermediate_trajdic = self._make_direct_trajectory_to(new_positions)

            newpositions_units = []
            new_trajpositions = self._get_trajectory_positions(
                new_traj_coord, trajdic=new_trajdic
            )
            for mot, pos in zip(self.motor_roles, new_trajpositions):
                newpositions_units.append(self._steps2units(mot, pos))
        except Exception as e:
            raise Exception(
                "Fail to calculate new trajectory parameters: {}".format(str(e))
            )
        if not self.check_positions_safe(newpositions_units):
            raise ValueError("Target position is not allowed")
        self.thread = threading.Thread(
            target=self._move_to_new_trajectory_worker,
            args=(
                new_trajdic,
                intermediate_trajdic,
                new_traj_coord,
                new_parameters_dict,
            ),
        )
        self.thread.start()

    def _move_to_new_trajectory_worker(
        self,
        new_trajdic,
        intermediate_trajdic,
        new_traj_coord,
        new_parameters_dict,
    ):
        """
        Worker thread that performs the move to a new trajectory.
        This thread is needed in order to not exceed the Tango timeouts
        when executing commands and reading/writing attributes.
        """
        self._log.debug("_move_to_new_trajectory_worker thread starting")
        try:
            self._set_changing_traj_state(TrajectoryState.MOVING_TO_INTERMEDIATE)
            self._use_intermediate_traj_to_switch_parameters(intermediate_trajdic)

            self._set_changing_traj_state(TrajectoryState.MOVING_TO_FINAL)
            self._use_final_traj_to_move_to_new_position(new_traj_coord, new_trajdic)
            self._store_active_params_and_traj(new_parameters_dict, new_trajdic)

            self._set_changing_traj_state(TrajectoryState.IDLE)
        except Exception as exc:
            self._log.exception("Failed to move to new trajectory")
            self._set_changing_traj_state(TrajectoryState.FAULT, str(exc))
        self._send_state_status_update()
        self._send_position_update_if_done()
        self._log.debug("_move_to_new_trajectory_worker thread finished")

    def _set_changing_traj_state(self, state, error_message=""):
        self._changing_traj_state = state
        if state is TrajectoryState.FAULT:
            self._changing_traj_error_message = error_message
        else:
            self._changing_traj_error_message = ""

    def _use_intermediate_traj_to_switch_parameters(self, intermediate_trajdic):
        """Use intermediate trajectory to safely move close to new position.

        The start of the intermediate trajectory is very close to the current
        positions of the motors. This means minimal movement when we "sync"
        the motors onto the parametric trajectory table. During this stage
        of movement the motors are not guaranteed to maintain safe positions
        relative to each other (e.g., ratio of mirror to grating angle for a
        plane grating monochromator like FlexPES).

        The end of the intermediate trajectory places the motors very close to
        where they need to be on once we switch to the final trajectory.
        In other words, what the user actually wants after a change in parameter
        like diffraction order, line density, etc.
        The benefit of the intermediate trajectory is that the movement along
        it ensures the motors maintain safe positioning.
        """
        self._load_intermediate_trajectory(intermediate_trajdic)
        self._sync_motors_to_intermediate_trajectory(intermediate_trajdic)
        self._set_velocity_and_acceleration_for_intermediate_trajectory()
        self._move_to_end_of_intermediate_trajectory(intermediate_trajdic)

    def _use_final_traj_to_move_to_new_position(self, new_traj_coord, new_trajdic):
        """Move to new position on final trajectory.

        After this, the newly requested trajectory based on user's new parameters,
        will be loaded in IcePAP, and ready to use.  E.g., the user can then sweep
        the beamline energy.
        """
        self._load_final_trajectory(new_trajdic)
        self._set_velocity_and_acceleration_for_final_trajectory()
        self._sync_motors_to_final_trajectory(new_traj_coord, new_trajdic)

    def _store_active_params_and_traj(self, new_parameters_dict, new_trajdic):
        try:
            self._trajdic = new_trajdic
            self.active_parameters_dict = dict(new_parameters_dict)
            self.next_parameters_dict = dict(new_parameters_dict)
            self.traj_coord_min = min(new_trajdic["_traj_pos"])
            self.traj_coord_max = max(new_trajdic["_traj_pos"])
        except Exception as e:
            raise TrajectoryChangeError(
                f"Failed while saving state after movement done (final trajectory): {e}"
            ) from e

    def _send_state_status_update(self):
        axis = 1
        state, status, limit_switches = self.StateOne(axis)
        element = self._getPoolController().get_element(axis=axis)
        element.set_status(status)
        element.set_state(state)

    def _send_position_update_if_done(self):
        if self._changing_traj_state is TrajectoryState.IDLE:
            try:
                axis = 1
                position = self.ReadOne(axis)
                if position is not None:
                    element = self._getPoolController().get_element(axis=axis)
                    element.fire_event(EventType("position", priority=1), position)
            except ReadPositionError:
                pass

    def _load_intermediate_trajectory(self, intermediate_trajdic):
        try:
            self._load_trajectories(trajdic=intermediate_trajdic)
        except Exception as e:
            raise TrajectoryChangeError(
                f"Failed to load intermediate trajectory: {e}"
            ) from e

    def _sync_motors_to_intermediate_trajectory(self, intermediate_trajdic):
        try:
            initial_pos = self._unit_to_transport(
                intermediate_trajdic["_traj_pos"][0], trajdic=intermediate_trajdic
            )
            with self.lock:
                self._log.debug(
                    "Syncing motors onto intermediate trajectory (1st movep)"
                )
                self._ipap.movep(initial_pos, list(self.motor_roles), group=True)
            self._wait_while_moving()
            self._log.debug("Synced motors onto intermediate trajectory (1st movep)")
            self._send_state_status_update()
        except Exception as e:
            raise TrajectoryChangeError(
                f"Failed syncing motors to intermediate trajectory (1st movep): {e}"
            ) from e

    def _set_velocity_and_acceleration_for_intermediate_trajectory(self):
        try:
            self._set_vel(self._max_vel * self.param_change_vel)
            self._set_acctime(self._acctime)
        except Exception as e:
            raise TrajectoryChangeError(
                f"Failed to set velocity / acceleration time "
                f"(intermediate trajectory): {e}"
            ) from e

    def _move_to_end_of_intermediate_trajectory(self, intermediate_trajdic):
        try:
            end_pos = self._unit_to_transport(
                intermediate_trajdic["_traj_pos"][-1], trajdic=intermediate_trajdic
            )
            end_pos = end_pos * 0.999
            self._check_safe_to_move(end_pos, intermediate_trajdic)
            with self.lock:
                self._log.debug(
                    "Parametric move to end of intermediate trajectory (1st pmove)"
                )
                self._ipap.pmove(end_pos, list(self.motor_roles), group=True)
            self._send_state_status_update()
            self._wait_while_moving()
            self._log.debug(
                "Finished parametric move to end of intermediate trajectory (1st pmove)"
            )
        except Exception as e:
            raise TrajectoryChangeError(
                f"Failed during/after 1st pmove (end intermediate trajectory): {e}"
            ) from e

    def _load_final_trajectory(self, new_trajdic):
        try:
            self._load_trajectories(trajdic=new_trajdic)
        except Exception as e:
            raise TrajectoryChangeError(f"Failed to load final trajectory: {e}") from e

    def _set_velocity_and_acceleration_for_final_trajectory(self):
        try:
            self._set_vel(self._velocity)
        except RuntimeError:
            try:
                self._log.warning(
                    f"Failed to set velocity to {self._velocity}, "
                    f"setting default value of {self._max_vel * self.param_change_vel}"
                )
                self._velocity = self._max_vel * self.param_change_vel
                self._set_vel(self._velocity)
            except Exception as e:
                raise TrajectoryChangeError(
                    f"Failed to set velocity (final trajectory): {e}"
                ) from e
        try:
            self._set_acctime(self._acctime)
        except Exception as e:
            raise TrajectoryChangeError(
                f"Failed to set acceleration time (final trajectory): {e}"
            ) from e

    def _sync_motors_to_final_trajectory(self, new_traj_coord, new_trajdic):
        try:
            final_pos = self._unit_to_transport(new_traj_coord, trajdic=new_trajdic)
            with self.lock:
                self._log.debug("Syncing motors onto final trajectory (2nd movep)")
                self._ipap.movep(final_pos, list(self.motor_roles), group=True)
            self._wait_while_moving()
            self._log.debug("Synced motors onto final trajectory (2nd movep)")
            self._send_state_status_update()
        except Exception as e:
            raise TrajectoryChangeError(
                f"Failed during/after 2nd movep (final trajectory): {e}"
            ) from e

    def _wait_while_moving(self):
        """
        Wait for all motors to stop moving
        Raises an exception if at least one stopped before reaching the target.
        """
        nbr_moving = 1
        while nbr_moving > 0:
            nbr_moving = 0
            time.sleep(0.1)
            with self.lock:
                motors_states = self._ipap.get_states(list(self.motor_roles))
            for state in motors_states:
                moving_flags = [state.is_moving(), state.is_settling()]
                # Check if the moving flag is active(True)
                # & Check if the settling flag is active(True)
                if any(moving_flags):  # is still moving
                    nbr_moving += 1
        with self.lock:
            motors_states = self._ipap.get_states(list(self.motor_roles))
        errors = []
        for index, state in enumerate(motors_states):
            stopcode = state.get_stop_code()
            if stopcode != 0:
                errors.append(f"motor={self.motor_roles[index]} stopcode={stopcode}")
        if errors:
            raise MoveToTargetError(
                f"Stopped before reaching target: ({','.join(errors)})"
            )

    def _load_trajectories(
        self,
        trajdic=None,
        mode="LINEAR",
    ):
        """
        Upload new trajectory tables to the icepap.
        Interpolation mode "mode" can be LINEAR or SPLINE.
        """
        if trajdic is None:
            trajdic = self._calc_trajectories()

        self._max_vel = None
        max_vels = []

        par_name = "_traj_pos"
        par_user_units = list(trajdic[par_name])
        traj_detail_table = [par_user_units]
        for motor_name in self.motor_roles:
            motor_table = list(trajdic[motor_name])
            traj_detail_table.append(
                [self._steps2units(motor_name, step) for step in motor_table]
            )
        par_transport_units = [
            self._unit_to_transport(pos, trajdic=trajdic) for pos in par_user_units
        ]
        traj_detail_table.append(par_transport_units)
        self._update_traj_details(trajdic, traj_detail_table, validated=False)

        if not trajdic["_is_safe"]:
            raise ValueError("The new trajectory contains forbidden positions")

        for motor_name in self.motor_roles:
            motor_table = list(trajdic[motor_name])
            with self.lock:
                try:
                    self._ipap[motor_name].clear_parametric_table()
                    self._ipap[motor_name].set_parametric_table(
                        par_transport_units, motor_table, mode=mode
                    )
                    max_vels.append(
                        float(self._ipap[motor_name].send_cmd("?PARVEL MAX")[0])
                    )
                except Exception as e:
                    self._log.error(
                        "Failed to send tables to motor {}, error: {}".format(
                            motor_name, e
                        )
                    )
                    raise
        self._log.debug("New trajectories loaded, max parvels: {0}".format(max_vels))
        self._max_vel = min(max_vels)
        self._trajdic = trajdic
        self._update_traj_details(trajdic, traj_detail_table, validated=True)

    def _get_trajectory_table_from_icepap(self):
        table = []
        for motor_name in self.motor_roles:
            with self.lock:
                try:
                    par, motor_steps, _ = self._ipap[motor_name].get_parametric_table()
                    if len(table) == 0:
                        table.append(par)  # expect same params in all motor tables
                    motor_units = [
                        self._steps2units(motor_name, step) for step in motor_steps
                    ]
                    table.append(motor_units)
                except Exception as e:
                    self._log.error(
                        "Failed to read table for motor {}, error: {}".format(
                            motor_name, e
                        )
                    )
        return table

    def _get_max_trajectory_velocity(self):
        """
        Read the maximum parametric velocity
        from all icepaps and return the smallest one.
        """
        max_vels = []
        for motor_name in self.motor_roles:
            with self.lock:
                max_vels.append(
                    float(self._ipap[motor_name].send_cmd("?PARVEL MAX")[0])
                )
        return min(max_vels)

    def _unit_to_transport(self, pos, trajdic=None):
        """
        Translate from normal units to transport dummy units.
        """
        if not self.transport_mode:
            return pos
        else:
            if trajdic is None:
                trajdic = self._trajdic
            return np.interp(pos, trajdic["_traj_pos"], trajdic["_transport"])

    def _transport_to_unit(self, pos, trajdic=None):
        """
        Translate from transport dummy units to normal units.
        """
        if not self.transport_mode:
            return pos
        else:
            if trajdic is None:
                trajdic = self._trajdic
            if pos < min(trajdic["_transport"]):
                return min(trajdic["_transport"])
            elif pos > max(trajdic["_transport"]):
                return max(trajdic["_transport"])
            pos_interp = np.interp(pos, trajdic["_transport"], trajdic["_traj_pos"])
            return pos_interp

    def _make_transport_table(self, trajdic):
        """
        Create the lookup table used when moving in transport mode.
        """
        if trajdic is None:
            trajdic = self._trajdic
        vels = {}
        positions = np.zeros(len(trajdic["_traj_pos"]))
        for motor_name in self.motor_roles:
            with self.lock:
                vels[motor_name] = self._ipap[motor_name].velocity
        for n, pos in enumerate(trajdic["_traj_pos"]):
            if n == 0:
                positions[0] = 0
            else:
                slowest = IPAPTrajCtrl.TRANSPORT_MODE_INITIAL_SLOWEST_STEP
                for mot in self.motor_roles:
                    step = abs(trajdic[mot][n] - trajdic[mot][n - 1])
                    steptime = step / vels[mot]
                    if steptime > slowest:
                        slowest = steptime
                # set traj step so 1 unit per second moves at max speed with
                # extra margin
                transportstep = (
                    IPAPTrajCtrl.TRANSPORT_MODE_SAFETY_MARGIN_SCALE_FACTOR * slowest
                )
                positions[n] = positions[n - 1] + transportstep
        return positions

    def _set_vel(self, vel):
        """
        Set the parametric velocity on all icepaps.
        """
        if vel is None or vel == 0.0:
            raise RuntimeError("Velocity cannot be zero.")
        elif vel < 0:
            vel = -vel / 100.0 * self._get_max_trajectory_velocity()
        for motor in self.motor_roles:
            with self.lock:
                self._ipap[motor].parvel = vel

    def _set_acctime(self, acctime):
        """
        Set the parametric acceleration time on all icepaps.
        """
        for motor in self.motor_roles:
            with self.lock:
                self._ipap[motor].paracct = acctime

    def _check_safe_to_move(self, pos, trajdic):
        """
        Check that everything is set up correctly so that it"s safe to move
        to the new position.
        """
        parvels = []
        paraccts = []
        accts = []
        ipappositions = []
        ipappositions_units = []
        for motor in self.motor_roles:
            with self.lock:
                parvels.append(self._ipap[motor].parvel)
                paraccts.append(self._ipap[motor].paracct)
                accts.append(self._ipap[motor].acctime)
                pos_temp = self._unit_to_transport(pos, trajdic=trajdic)
                self._log.debug(
                    "Check position, units: {}, in transport units: {}, motor: {}".format(
                        pos, pos_temp, motor
                    )
                )
                try:
                    motpos = float(
                        self._ipap[motor].send_cmd("?parval {}".format(pos_temp))[0]
                    )
                    ipappositions.append(motpos)
                    ipappositions_units.append(self._steps2units(motor, motpos))
                except RuntimeError:
                    ipappositions.append(None)
                    ipappositions_units.append(None)

        if (max(parvels) - min(parvels)) > (max(parvels) / 100.0):
            raise Exception("Parametric velocities not matched")

        if None in ipappositions or not self.check_positions_safe(ipappositions_units):
            raise ValueError("Target position is not allowed")
        for acct, paracct in zip(accts, paraccts):
            if paracct < acct:
                raise ValueError("Parametric acceleration time is too short")
        trajpositions = self._get_trajectory_positions(pos, trajdic=trajdic)
        for ipappos, trajpos in zip(ipappositions, trajpositions):
            if abs(ipappos - trajpos) > self._tolerance_steps:
                raise ValueError(
                    "Parametric position calculated by IcePAP is wrong\n\
                    Trajectory: {}, IcePAP: {}".format(
                        trajpos, ipappos
                    )
                )

    def _move_onto_trajectory_at(self, position, tolerance=1.0):
        """
        Start a move that brings all motors onto the trajectory at the given position.
        """
        currpos = self._get_nearest_trajectory_position()
        if abs(currpos - position) > tolerance:
            raise ValueError(
                "Must provide a position that is close to the current, "
                "read NearestTrajectoryPosition"
            )
        if self._changing_traj_state in (
            TrajectoryState.MOVING_TO_INTERMEDIATE,
            TrajectoryState.MOVING_TO_FINAL,
        ):
            raise RuntimeError(
                f"Cannot move directly onto trajectory while busy with previous "
                f"trajectory change. Currently in {self._changing_traj_state}. "
                f"Wait for it to finish, or use Stop() or Abort() command."
            )
        self._get_motors_param()
        try:
            self._load_trajectories()
        except ValueError as e:
            self._log.error(str(e))
            raise e
        temp_pos = self._unit_to_transport(position)
        with self.lock:
            self._ipap.movep(temp_pos, list(self.motor_roles), group=True)
        self.move_onto_trajectory_at = position
        self._set_changing_traj_state(TrajectoryState.IDLE)

    def _is_limit_switch_active(self, limit_switches):
        """
        Check if upper and/or lower limit switches are active.
        """
        return bool(
            limit_switches & self.UpperLimitSwitch
            or limit_switches & self.LowerLimitSwitch
        )

    def _reset_stopcode(self):
        """
        Work around:
        Small move on all motors not on limit switch
        to reset stopcode value to 0='End of movement'.
        """
        MOVE_STEPS = 10
        with self.lock:
            motor_states = self._ipap.get_states(list(self.motor_roles))
        for state, mot in zip(motor_states, list(self.motor_roles)):
            if not any([state.is_limit_positive(), state.is_limit_negative()]):
                with self.lock:
                    self._ipap.rmove([(mot, MOVE_STEPS)])
                try:
                    self._wait_while_moving()
                except MoveToTargetError as e:
                    self._log.debug(f"Non-critical movement incomplete: {e}")

    def _back_off_limit_switch_worker(self, axis):
        """
        Worker thread that attempts to move off limit switches.

        Checks which motors are on their limit switch
        and back those motors off individually in incremental steps.
        Note that this will not move all motors in sync.
        """
        self._reset_stopcode()
        MOVE_STEPS = 100
        outside_limit = True
        while outside_limit:
            outside_limit = False
            time.sleep(0.1)
            with self.lock:
                motor_states = self._ipap.get_states(list(self.motor_roles))
            # For all motors check if a limit switch is active.
            for state, mot in zip(motor_states, list(self.motor_roles)):
                rel_pos = 0
                if state.is_limit_positive():
                    rel_pos = -MOVE_STEPS
                    outside_limit = True
                elif state.is_limit_negative():
                    rel_pos = MOVE_STEPS
                    outside_limit = True
                # If LS active, back off that motor incrementally using relative move.
                if abs(rel_pos) > 0:
                    with self.lock:
                        self._ipap.rmove([(mot, rel_pos)])
                    try:
                        self._wait_while_moving()
                    except MoveToTargetError as e:
                        self._log.debug(f"Move back off limit switch failed. {e}")
                        return

    def AddDevice(self, axis):
        """Set default values for the axis and try to connect to it
        @param axis to be added
        """
        self._step_per_unit = 1

    def DeleteDevice(self, axis):
        pass

    def DefinePosition(self, axis, position):
        # Moved to new attribute MoveOntoTrajectoryAt,
        # but keep backward compatibility for this method.
        self.set_MoveOntoTrajectoryAt(axis, position)

    def StateOne(self, axis):
        """
        Connect to the hardware and check the state.
        If no connection available, return ALARM.
        @param axis to read the state
        @return the state value: {ALARM|ON|MOVING}
        """
        with self.lock:
            motors_states = self._ipap.get_states(list(self.motor_roles))
        moving = []
        warning = []
        alarm = []
        limit_switches = self.NoLimitSwitch

        for motstate, motor in zip(motors_states, self.motor_roles):
            moving_flags = [motstate.is_moving(), motstate.is_settling()]
            warning_flags = [motstate.is_outofwin(), motstate.is_warning()]
            if any(moving_flags):
                moving.append(motor)
            if any(warning_flags):
                warning.append(motor)
            alarm_flags = [
                motstate.is_limit_positive(),
                motstate.is_limit_negative(),
                not motstate.is_poweron(),
            ]
            if any(alarm_flags):
                alarm.append(motor)

            if motstate.is_inhome():
                limit_switches |= self.HomeLimitSwitch
            if motstate.is_limit_positive():
                limit_switches |= self.UpperLimitSwitch
            if motstate.is_limit_negative():
                limit_switches |= self.LowerLimitSwitch

        if len(alarm) > 0:
            state = State.Alarm
            limsw_msg = ""
            if self._is_limit_switch_active(limit_switches):
                limsw_msg = (
                    "\nBack off limit switch using the BackOffLimitSwitch attribute."
                )
            status = "The motors {0} are in alarm state.{1}\n".format(
                " ".join(alarm), limsw_msg
            )
        elif len(moving) > 0:
            state = State.Moving
            status = "The motors {0} are moving.\n".format(" ".join(moving))
        # elif len(warning) > 0:
        #    state = State.Warning
        #    status = "The motors {0} are in warning state".format(" ".join(
        # warning))
        else:
            state = State.On
            status = "All motors are ready"

        if state != State.Moving and self._changing_traj_state in (
            TrajectoryState.IDLE,
            TrajectoryState.FAULT,
        ):
            limsw_msg = ""
            if self._is_limit_switch_active(limit_switches):
                limsw_msg = (
                    "- Back off limit switch using the BackOffLimitSwitch attribute.\n"
                )
            resync_msg = (
                "To resynch, please do the following:\n{0}"
                "- Read the trajectory position with the NearestTrajectoryPosition attribute.\n"
                "- Write that trajectory position to the MoveOntoTrajectoryAt attribute.\n".format(
                    limsw_msg
                )
            )

            sync = []
            for motor in self.motor_roles:
                try:
                    with self.lock:
                        self._ipap[motor].parpos
                except RuntimeError:
                    msg = "{} is not on trajectory\n".format(motor)
                    sync.append(msg)

            if len(sync) > 0:
                state = State.Alarm
                status = "There are motors not synchronized.\n {0}\n{1}".format(
                    " ".join(sync), resync_msg
                )
            else:
                with self.lock:
                    master_pos = self._transport_to_unit(
                        self._ipap[self.master_motor].parpos
                    )
                out_pos = []
                msg = "{0} at {1} but should be within [{2}..{3}]\n"
                for motor in self.motor_roles:
                    with self.lock:
                        motor_pos = self._transport_to_unit(self._ipap[motor].parpos)
                    diff = abs(motor_pos - master_pos)
                    is_close = diff <= self._tolerance
                    if not is_close:
                        out_pos.append(
                            msg.format(
                                motor,
                                motor_pos,
                                master_pos - self._tolerance,
                                master_pos + self._tolerance,
                            )
                        )

                if len(out_pos) > 0:
                    state = State.Alarm
                    status = (
                        "There are motors with different positions.\n {0}\n{1}".format(
                            " ".join(out_pos), resync_msg
                        )
                    )
                # self._log.debug("State: %r Status: %r" % (state, status))
                status = status + "\n" + self.compile_trajdic_info(self._trajdic)
        if self._changing_traj_state in (
            TrajectoryState.MOVING_TO_INTERMEDIATE,
            TrajectoryState.MOVING_TO_FINAL,
        ):
            # We don't want the state to fluctuate while Sardana is scanning.
            # if StateOne is called while the worker thread is busy updating the
            # trajectory some hardware checks above might report an error temporarily.
            # This was noticeable at Veritas when scanning a trajectory parameter
            # like Cff.
            state = State.Moving
            status += "\nMoving to a new trajectory - force State.Moving."
        elif self._changing_traj_state == TrajectoryState.FAULT:
            state = State.Alarm
            status += (
                f"\nError while moving to new trajectory:"
                f"\n\t{self._changing_traj_error_message}"
            )
        return state, status, limit_switches

    def ReadOne(self, axis):
        """Read the position of the axis.
        @param axis to read the position
        @return the current axis position
        """

        state, status, limit_switches = self.StateOne(axis)
        if state == State.Alarm:
            raise ReadPositionError(status)
        with self.lock:
            pos_icepap = self._ipap[self.master_motor].parpos
            pos = self._transport_to_unit(pos_icepap)
        trajpos_list = self._trajdic["_traj_pos"]
        if self._trajdic["_parameters"] is None:
            return None
        pos_list = self._trajdic["_parameters"][0]
        value = np.interp(pos, trajpos_list, pos_list)
        return value

    def StartOne(self, axis, pos):
        """Start movement of the axis.
        :param axis: int
        :param pos: float
        :return: None
        """
        if self.thread is not None:
            if self.thread.is_alive():
                raise RuntimeError("Thread already running!")
        state, status, limit_switches = self.StateOne(axis)
        if state != State.On:
            raise RuntimeError(status)
        if pos < self.traj_coord_min or pos > self.traj_coord_max:
            raise ValueError(
                "Bad position, allowed range is {} - {}".format(
                    self.traj_coord_min, self.traj_coord_max
                )
            )
        self._set_vel(self._velocity)
        self._set_acctime(self._acctime)
        self._check_safe_to_move(pos, self._trajdic)
        pos_transport = self._unit_to_transport(pos)
        with self.lock:
            self._ipap.pmove(pos_transport, list(self.motor_roles), group=True)

    def StopOne(self, axis):
        """
        Stop axis
        :param axis: int
        :return: None
        """
        with self.lock:
            self._ipap.stop(list(self.motor_roles))
        self._set_changing_traj_state(TrajectoryState.IDLE)

    def AbortOne(self, axis):
        """
        Abort movement
        :param axis: int
        :return: None
        """
        with self.lock:
            self._ipap.abort(list(self.motor_roles))
        self._set_changing_traj_state(TrajectoryState.IDLE)

    def SetAxisPar(self, axis, name, value):
        """Set the standard pool motor parameters.
        @param axis to set the parameter
        @param name of the parameter
        @param value to be set
        """
        attr_name = name.lower()

        if attr_name == "step_per_unit":
            self._step_per_unit = float(value)
        elif attr_name == "velocity":
            velocity = value * self._step_per_unit
            try:
                self._max_vel = self.get_MaxVelocity(axis)
            except RuntimeError as e:
                self._log.error(
                    "Failed to read max velocity from IcePAP, error:" + str(e)
                )
            if self._velocity is None:
                self._log.debug(
                    "First write of velocity, probably to restore the memorized value."
                )
                # Pool just started, writing memorized attributes
                if velocity > self._max_vel:
                    self._log.warning(
                        "Memorized velocity is too high! Reverting to a safe default."
                    )
                    velocity = self.param_change_vel * self._max_vel
            if velocity > self._max_vel and self._max_vel != 0.0:
                raise ValueError(
                    "The value must be lower/equal than/to " "{0}".format(self._max_vel)
                )
            self._velocity = velocity
        elif attr_name == "acceleration":
            self._acctime = value
            try:
                self._set_acctime(self._acctime)
            except RuntimeError as e:
                self._log.error(
                    "Unable to write acceleration time to IcePAP, error: {}".format(e)
                )
        elif attr_name == "base_rate":
            pass
        elif attr_name == "deceleration":
            pass
        else:
            MotorController.SetAxisPar(self, axis, name, value)

    def GetAxisPar(self, axis, name):
        """Get the standard pool motor parameters.
        @param axis to get the parameter
        @param name of the parameter to get the value
        @return the value of the parameter
        """

        attr_name = name.lower()
        if attr_name == "step_per_unit":
            result = self._step_per_unit
        elif attr_name == "velocity":
            result = self._velocity
        elif attr_name == "acceleration" or attr_name == "deceleration":
            # The IcePAP always returns 0.0, firmware bug?
            # with self.lock:
            #    acctime = self._ipap[self.master_motor].paracct
            # result = acctime
            result = self._acctime
        elif attr_name == "base_rate":
            result = 0
        else:
            result = MotorController.GetAxisPar(self, axis, name)
        return result

    # ------------------ Attribute setters and getters
    def get_MaxVelocity(self, axis):
        if axis == 1:
            return self._get_max_trajectory_velocity()

    def get_DistanceToTrajectory(self, axis):
        if axis == 1:
            return self.get_distance_to_trajectory()

    def get_NearestTrajectoryPosition(self, axis):
        if axis == 1:
            return self._get_nearest_trajectory_position()

    def get_ParamChangeVel(self, axis):
        if axis == 1:
            return self.param_change_vel

    def set_ParamChangeVel(self, axis, value):
        if axis == 1:
            self.param_change_vel = value

    def get_TransportMode(self, axis):
        if axis == 1:
            return self.transport_mode

    def set_TransportMode(self, axis, value):
        if axis == 1:
            state, status, limit_switches = self.StateOne(axis)
            if state == State.On:
                self.transport_mode = value
                # move to another trajectory!
                self.move_to_new_trajectory(
                    self.next_parameters_dict, keep_traj_pos=False
                )
            elif state == State.Alarm:
                # update parameter for MoveOntoTrajectoryAt
                self.transport_mode = value
            else:
                raise ValueError(
                    "It is not allowed to write TransportMode in state {}".format(state)
                )

    def get_MoveOntoTrajectoryAt(self, axis):
        if axis == 1:
            return self.move_onto_trajectory_at

    def set_MoveOntoTrajectoryAt(self, axis, value):
        if axis == 1:
            self._move_onto_trajectory_at(value, self._tolerance_pos_diff)

    def get_UnsafeMoveOntoTrajectoryAt(self, axis):
        if axis == 1:
            return self.move_onto_trajectory_at

    def set_UnsafeMoveOntoTrajectoryAt(self, axis, value):
        # "Unsafe" move with 10x the movement allowed
        if axis == 1:
            self._move_onto_trajectory_at(value, self._tolerance_pos_diff * 10)

    def get_BackOffLimitSwitch(self, axis):
        # Return limit switch status
        if axis == 1:
            state, status, limit_switches = self.StateOne(axis)
            return self._is_limit_switch_active(limit_switches)

    def set_BackOffLimitSwitch(self, axis, value):
        # Used as command, write value ignored
        # The move is handled by a separate thread, so this function returns right away.
        if axis == 1:
            state, status, limit_switches = self.StateOne(axis)
            if self._is_limit_switch_active(limit_switches):
                if self.thread is not None:
                    if self.thread.is_alive():
                        raise Exception("Thread already running!")
                self.thread = threading.Thread(
                    target=self._back_off_limit_switch_worker, args=(axis,)
                )
                self.thread.start()
            else:
                raise ValueError("No limit switch activated")

    def get_TrajectoryState(self, axis):
        state, _, _ = self.StateOne(axis)
        good_states = {State.On, State.Moving}
        return state in good_states

    def get_ParameterLimits(self, axis):
        limits = self.calc_parameter_limits()
        return json.dumps(limits)

    def get_MinTrajRange(self, axis):
        return max(
            [
                float(self._ipap[motor_name].send_cmd("?PRANGE MIN")[0])
                for motor_name in self.motor_roles
            ]
        )

    def set_MinTrajRange(self, axis, value):
        for motor_name in self.motor_roles:
            self._ipap[motor_name].send_cmd(f"PRANGE MIN {value}")

    def get_MaxTrajRange(self, axis):
        return min(
            [
                float(self._ipap[motor_name].send_cmd("?PRANGE MAX")[0])
                for motor_name in self.motor_roles
            ]
        )

    def set_MaxTrajRange(self, axis, value):
        for motor_name in self.motor_roles:
            self._ipap[motor_name].send_cmd(f"PRANGE MAX {value}")

    def get_ActiveTrajectory(self):
        return self._traj_details["ActiveTrajectory"]["_table"]

    def get_IntermediateTrajectoryUnvalidated(self):
        return self._traj_details["IntermediateTrajectoryUnvalidated"]["_table"]

    def get_IntermediateTrajectoryValidated(self):
        return self._traj_details["IntermediateTrajectoryValidated"]["_table"]

    def get_FinalTrajectoryUnvalidated(self):
        return self._traj_details["FinalTrajectoryUnvalidated"]["_table"]

    def get_FinalTrajectoryValidated(self):
        return self._traj_details["FinalTrajectoryValidated"]["_table"]

    def get_TrajectoryDetails(self):
        """Return details, with indentation to be more human-readable."""
        return json.dumps(self._traj_details, indent=4)

    # ------------------ Setters and getter for general trajectory parameters.
    def write_traj_parameter(
        self, attribute, value, axis=1, keep_traj_pos=True, skip_intermediate=False
    ):
        """
        Method for writing a trajectory parameter.
        If the trajectory mode is ok, start a motion.
        If not, store the new value to it will be used
        when the trajectory mode is initialized.
        """
        if axis == 1:
            state, status, limit_switches = self.StateOne(axis)
            if state == State.On:
                self.next_parameters_dict[attribute] = value
                if skip_intermediate:
                    # Used when changing offsets
                    self._move_without_intermediate(attribute, value)
                else:
                    # move to another trajectory!
                    self.move_to_new_trajectory(
                        self.next_parameters_dict, keep_traj_pos=keep_traj_pos
                    )
            elif state == State.Alarm:
                # update parameter for MoveOntoTrajectoryAt
                self.active_parameters_dict[attribute] = value
            else:
                raise ValueError(
                    "It is not allowed to write parameters while in state: {}".format(
                        state
                    )
                )

    def _move_without_intermediate(self, attribute, value):
        previous_value = self.active_parameters_dict[attribute]
        self.active_parameters_dict[attribute] = value
        try:
            self._set_changing_traj_state(TrajectoryState.MOVING_TO_FINAL)
            self._load_trajectories()
            new_traj_coord = self._get_nearest_trajectory_position(self._trajdic)
            self._set_velocity_and_acceleration_for_final_trajectory()
            self._sync_motors_to_final_trajectory(new_traj_coord, self._trajdic)
            self._set_changing_traj_state(TrajectoryState.IDLE)
        except Exception as e:
            self.active_parameters_dict[attribute] = previous_value
            raise Exception(f"Fail to move without intermediate trajectory: {e}")

    def read_traj_parameter(self, attribute, axis=1):
        """
        Method for reading the current value of a trajectory parameter.
        """
        if axis == 1:
            state, status, limit_switches = self.StateOne(axis)
            if state == State.Alarm:
                if attribute in self.active_parameters_dict:
                    return self.active_parameters_dict[attribute]
                else:
                    raise RuntimeError(status)
            try:
                with self.lock:
                    trajpos = self._transport_to_unit(
                        self._ipap[self.master_motor].parpos
                    )
            except RuntimeError:
                # IcePAP sync error is possible when moving to new trajectory, so use
                # previously active parameter, if available
                if (
                    self._changing_traj_state
                    in (
                        TrajectoryState.MOVING_TO_INTERMEDIATE,
                        TrajectoryState.MOVING_TO_FINAL,
                    )
                    and attribute in self.active_parameters_dict
                ):
                    return self.active_parameters_dict[attribute]
                else:
                    raise
            trajpos_list = self._trajdic["_traj_pos"]
            param_idx = self.trajectory_parameters.index(attribute) + 1
            if self._trajdic["_parameters"] is None:
                # Temporary trajectory!
                return None
            param_list = self._trajdic["_parameters"][param_idx]
            value = np.interp(trajpos, trajpos_list, param_list)
            return value

    # ------------------ Helpers for creating axis attributes and properties
    @staticmethod
    def addAttributes(attrs):
        axis_attributes = dict(IPAPTrajCtrl.axis_attributes)
        axis_attributes.update(attrs)
        return axis_attributes

    @staticmethod
    def addProperties(props):
        ctrl_properties = dict(IPAPTrajCtrl.ctrl_properties)
        ctrl_properties.update(props)
        return ctrl_properties

    @staticmethod
    def addMotorRoles(motors):
        ctrl_properties = dict(IPAPTrajCtrl.ctrl_properties)
        newprops = {}
        for motor in motors:
            key = "{}_name".format(motor)
            desc = "{} motor alias".format(motor)
            newprops[key] = {Type: str, Description: desc}
        ctrl_properties.update(newprops)
        return ctrl_properties

    @staticmethod
    def makeMotorRoleProps(motors):
        newprops = {}
        for motor in motors:
            key = "{}_name".format(motor)
            desc = "{} motor alias".format(motor)
            newprops[key] = {Type: str, Description: desc}
        return newprops
