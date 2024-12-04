#!/usr/bin/env python3
"""Sardana motor controller for dynamic IcePAP parametric trajectories

A proof-of-concept implementation of a Sardana motor controller specifically
designed for _dynamic_ IcePAP parametric trajectories.

## Motivation

Whilst the `sardana-icepaptrajctrl` controller is robust and performant for
strictly defined trajectories (e.g. monochromator gratings, spectormeter arms,
…), it has some limitations which make it unsuitable for more dynamically
defined trajectories (e.g. continuous scannning of samples, multi-modal
spectroscopy, …);

* Max one axis — new controller instance is required for each new trajectory
* Number of motors defined in code (controller roles)
* Motors defined on controller level (controller roles) — Pool restart required
  on change
* Trajectory defined in code (albeit based on axis attributes [trajectory
  parameters])

The controller defined below is a new proof-of-concept implementation of an
IcePAP parametric trajectory Sardana motor controller which aims to address
these issues in order to better enable dynamic trajectories.


## Design

Each axis of the controller is a Sardana motor whose position corresponds to
the trajectory parameter. Moving this 'parametric motor' in 1-dimension, moves
the multiple 'physical motors' along the multi-dimensional trajectory. In this
way, motion along the trajectory is presented to Sardana in the same way as any
other motor.

The 'physical motors' involved the trajectory are defined for each axis in the
`motors` axis attribute (`DEV_STRING + SPECTRUM`). These are specified by their
Sardana names (Tango aliases), e.g. for a trajectory involving 3 'physical
motors';

```python
>>> parametric_motor0.motors = ("motor0", "motor1", "motor2")
```

The trajectory is defined for each axis in the `motor_positions` attribute
(`DEV_DOUBLE + IMAGE`). This is an array of 'physical motor' positions; e.g.
for a trajectory involving 3 'physical motors;

```python
>>> parametric_motor0.motor_positions = [
    [motor0_position0, motor1_position0, motor2_position0],
    [motor0_position1, motor1_position1, motor2_position1],
    [motor0_position2, motor1_position2, motor2_position2],
    [motor0_position3, motor1_position3, motor2_position3],
    [motor0_position4, motor1_position4, motor2_position4],
    [motor0_position5, motor1_position5, motor2_position5],
    [motor0_position6, motor1_position6, motor2_position6],
    …
]
```

The length of the first dimension must match the number of motors involved in
the trajectory. The length of the second dimension is arbitrary and defines the
number of points in the trajectory.

The 'parametric motor' position is defined to be the Euclidean distance between
the points of the trajectory; e.g. for a simple 2-dimensional trajectory
involving motors `x` and `y`, the position `p` of the 'parametric motor' would
be;

```
y
│                ⊗(x3,y3);p3=52
│                ┃
│  (x1,y1);p1=10 ┃
│  ⊗━━━━━━━━━━━━━⊗(x2,y2);p2=41
│  ┃
│  ┃
│  ⊗(x0,y0);p0=0
│
│
└────────────────────── x
```

The maximum position of the 'paramtric motor' (`p3` in the example above) for
each axis is available in the `max_position` attribute. The 'parametric motor'
position will therefore always be in the interval `[0, max_position]`.

The maximum velocity of the 'parametric motor' is limited by the velocities and
accelerations of the 'physical motors'. It is available in the `max_velocity`
attribute.

_N.b. The parametric acceleration time at high parametric velocities may be
very large!_ As such, the parametric velocity defaults to 95% of the maximum.

## Summary

### Axis attributes

Attribute          | Description                                        | Dimensionality
-------------------|----------------------------------------------------|---------------
`motors`           | Aliases of motors involved in the trajectory       | 1
`motor_positions`  | Positions of the motors involved in the trajectory | 2
`max_position`     | Maximum parametric position                        | 0
`max_velocity`     | Maximum parametric velocity                        | 0

"""

# Std. lib.
import functools
import typing

# External
import numpy
import icepap
import tango
import sardana
import sardana.pool.controller


def _log_exceptions(wrapped):
	"""Decorator to catch and log all exceptions"""
	@functools.wraps(wrapped)
	def wrapper(self, *args, **kwargs):
		try:
			return wrapped(self, *args, **kwargs)
		except Exception as err:
			self._log.error(str(err))
			raise
	return wrapper


class IcepapDynamicTrajectoryController(sardana.pool.controller.MotorController):
	"""Sardana motor controller for dynamic IcePAP parametric trajctories

	A proof-of-concept implementation of a Sardana motor controller for dynamic
	IcePAP parametric trajectories.

	## Notes

	* Each axis of the controller is a Sardana motor whose position corresponds
	  to the trajectory parameter. Moving this 'parametric motor' in 1-dimension,
	  moves the multiple 'physical motors' along the multi-dimensional
	  trajectory.

	* The controller code mainly handles and exposes the required Sardana
	  (motor) API

	* Each axis is stored as an instance of the `_IcepapDynamicTrajectoryAxis`
	  class. Most of the IcePAP specific functionality is handled in this class.

	"""

	# Sardana Motor API ----------------------------------------------------- #

	axis_attributes = {
		"motors": {
			sardana.pool.controller.Type: (str,),
			sardana.pool.controller.Access: sardana.pool.controller.DataAccess.ReadWrite,
			sardana.pool.controller.MaxDimSize: (128,),
			sardana.pool.controller.Memorize: sardana.pool.controller.Memorized,
			sardana.pool.controller.Description: (
				"Names (Tango aliases) of Sardana motors involved in the trajectory"
			)
		},
		"motor_positions": {
			sardana.pool.controller.Type: ((float,),),
			sardana.pool.controller.Access: sardana.pool.controller.DataAccess.ReadWrite,
			sardana.pool.controller.MaxDimSize: (
				129,	# Max number of motors. Limited by max drivers in an IcePAP system
				2048	# Max number of trajectory points. Limited by Tango image attribute size(?)
			),
			sardana.pool.controller.Memorize: sardana.pool.controller.Memorized,
			sardana.pool.controller.Description: (
				"Positions of Sardana motors along the trajectory."
				" Array dimensions must match number of motors,"
				" e.g. [[motor0_pos0, motor1_pos0], [[motor0_pos1, motor1_pos1], ...]"
			)
		},
		"max_position": {
			sardana.pool.controller.Type: float,
			sardana.pool.controller.Access: sardana.pool.controller.DataAccess.ReadOnly,
			sardana.pool.controller.MaxDimSize: [],
			sardana.pool.controller.Memorize: sardana.pool.controller.NotMemorized,
			sardana.pool.controller.Description: (
				"Maximum parametric position (trajectory length)."
				" Defined as sum of Euclidean distances between motor positions"
			)
		},
		"max_velocity": {
			sardana.pool.controller.Type: float,
			sardana.pool.controller.Access: sardana.pool.controller.DataAccess.ReadOnly,
			sardana.pool.controller.MaxDimSize: [],
			sardana.pool.controller.Memorize: sardana.pool.controller.NotMemorized,
			sardana.pool.controller.Description: (
				"Maximum parametric velocity"
			)
		},
	}

	def __init__(self, inst, props, *args, **kwargs):

		# Superclass constructor
		super().__init__(
			inst,
			props,
			*args,
			**kwargs
		)											 # Calls MotorController.__init__

		# Init axes
		self._axes = {}

	def AddDevice(self, axis):
		self._axes[axis] = _IcepapDynamicTrajectoryAxis(
			self,
			axis
		)

	def DeleteDevice(self, axis):
		try:
			del self._axes[axis]
		except KeyError:
			self._log.error(f"No axis {axis}")

	@_log_exceptions
	def StateOne(self, axis) -> tuple[sardana.State, str]:
		return self._axes[axis].state

	@_log_exceptions
	def ReadOne(self, axis) -> float:
		return self._axes[axis].position

	@_log_exceptions
	def StartOne(self, axis, position):
		self._axes[axis].position = position

	@_log_exceptions
	def StopOne(self, axis):
		self._axes[axis].stop()

	@_log_exceptions
	def AbortOne(self, axis):
		self._axes[axis].abort()

	@_log_exceptions
	def GetAxisPar(self, axis, name):
		'''													# Python >= 3.10 😭
		match name.lower():
			case "acceleration" | "deceleration":
				return self._axes[axis].acceleration
			case "velocity" | "base_rate":
				return self._axes[axis].velocity
			case "step_per_unit":
				return 1									# position == parameter
		'''
		name = name.lower()
		if name in ("acceleration" , "deceleration"):
			return self._axes[axis].acceleration
		elif name in ("velocity" , "base_rate"):
			return self._axes[axis].velocity
		elif name == "step_per_unit":
			return 1									# position == parameter

	@_log_exceptions
	def SetAxisPar(self, axis, name, value):
		'''													# Python >= 3.10 😭
		match name.lower():
			case "acceleration" | "deceleration":
				self._axes[axis].acceleration = value
			case "velocity" | "base_rate":
				self._axes[axis].velocity = value
			case "step_per_unit":
				pass										# position == parameter
		'''
		name = name.lower()
		if name in ("acceleration" , "deceleration"):
			self._axes[axis].acceleration = value
		elif name in ("velocity" , "base_rate"):
			self._axes[axis].velocity = value
		elif name == "step_per_unit":
			pass										# position == parameter

	@_log_exceptions
	def GetAxisExtraPar(self, axis, name):
		name = name.lower()
		if name == "motors":
			value = self._axes[axis].motor_aliases
			empty_DEVVAR_STRINGARRAY = tuple()
			return (
				empty_DEVVAR_STRINGARRAY if value is None
				else value
			)
		elif name == "motor_positions":
			value = self._axes[axis].motor_positions
			empty_DEVVAR_DOUBLEARRAY = numpy.ndarray((0,0))
			return (
				empty_DEVVAR_DOUBLEARRAY if value is None
				else value
			)
		elif name == "max_position":
			return self._axes[axis].max_position
		elif name == "max_velocity":
			return self._axes[axis].max_velocity

	@_log_exceptions
	def SetAxisExtraPar(self, axis, name, value):
		name = name.lower()
		if name == "motors":
			self._axes[axis].motor_aliases = value
		elif name == "motor_positions":
			self._axes[axis].motor_positions = value

	# Private interface ----------------------------------------------------- #

	_axes = None


class _IcepapDynamicTrajectoryAxis:
	"""'Parametric motor' class

	A class encapsulating a 'parametric motor' — a motor whose position
	corresponds to the parameter of a parametric trajectory. Moving the
	'parametric motor' in 1-dimension, moves multiple 'physical motors' along a
	multi-dimensional trajectory.

	The 'physical motors' involved are defined in the `motor_aliases`
	attribute. See the `motor_alias` docstring for further details.

	The positions of the 'physical motors' along the trajectory are defined in
	the `motor_positions` attribute. See the `motor_positions` docstring for
	further details.

	The relationship between the 'physical motor' positions and the 'parametric
	motor' position (the parameter), is computed from the `motor_positions`
	array and is defined to be the Euclidean distances between the points of
	the trajectory. See the `position` docstring for further details.

	"""

	# Public interface ------------------------------------------------------ #

	def __init__(
		self,
		controller : IcepapDynamicTrajectoryController,
		axis_index : int
	):
		self._controller = controller
		self._axis_index = axis_index

	@property
	def motor_aliases(self) -> typing.Sequence[str]:
		"""'Physical motors' involved in the trajectory

		Specified by their Tango aliases

		"""
		return self._motor_aliases

	@motor_aliases.setter
	def motor_aliases(
		self,
		#value : typing.Sequence[str] | None						# Python >= 3.10 😭
		value : typing.Union[typing.Sequence[str], None]
	):

		# Validate
		if not value:
			motor_aliases = None
			motor_names = None
			motor_addresses = None
			icepap = None
		else:
			motor_aliases = tuple(value)
			self._validate_motor_aliases(motor_aliases)
			motor_names = tuple(self._device_names(motor_aliases))
			motor_addresses = tuple(self._device_axes(motor_names))
			icepap = self._common_icepap(motor_names)

		# Clean-up
		if self._icepap:
			self._clear_trajectory()
			self._icepap.disconnect()

		# Set
		self._motor_aliases = motor_aliases
		self._motor_names = motor_names
		self._motor_addresses = motor_addresses
		self._icepap = icepap

		# Load trajectory
		if (
			(self.motor_aliases is not None)
			and (self.motor_positions is not None)
		):
			self._load_trajectory()

	@property
	def motor_positions(self) -> numpy.ndarray:
		"""Positions of the 'physical motors' along the trajectory
		
		2-dimenstional array of 'physical motor' positions; e.g. for a trajectory
		involving 3 'physical motors';

		```
			motor_positions = [
				[motor0_position0, motor1_position0, motor2_position0],
				[motor0_position1, motor1_position1, motor2_position1],
				[motor0_position2, motor1_position2, motor2_position2],
				[motor0_position3, motor1_position3, motor2_position3],
				[motor0_position4, motor1_position3, motor2_position4],
				[motor0_position5, motor1_position3, motor2_position5],
				[motor0_position6, motor1_position3, motor2_position6],
				…
			]
		```

		The length of the first dimension must match the number of motors involved
		in the trajectory (i.e. `len(self.motor_aliases)`.

		The length of the second dimension is arbitrary and defines the number of
		points in the trajectory.

		"""
		return self._motor_positions

	@motor_positions.setter
	def motor_positions(
		self,
		#value : numpy.ndarray | None								# Python >= 3.10 😭
		value : typing.Union[numpy.ndarray, None]
	):

		# Validate
		if (value is None) or (not value.size):
			motor_positions = None
		else:
			motor_positions = value
			self._validate_motor_positions(motor_positions)

		# Set
		self._motor_positions = motor_positions

		# Load trajectory
		if (
			(self.motor_aliases is not None)
			and (self.motor_positions is not None)
		):
			self._load_trajectory()

		# Set trajectory dependent bounds
		self._set_device_attribute_ranges(
			(
				self._axis_alias + "/position",
				self._axis_alias + "/velocity"
			),
			(
				(0.0, self.max_position),
				(0.0, self.max_velocity)
			)
		)

		# Set parametric velocity (trajectory dependent)
		#
		#	Default to 95% of max to avoid very long acceleration times
		#
		self.velocity = 0.95 * self.max_velocity

	@property
	def state(self) -> tuple[sardana.State, str]:
		"""Sardana state"""

		# INIT — Motors or trajectory undefined
		try:
			self._assert_motor_aliases()
			self._assert_motor_positions()
			self.position									# On trajectory
		except Exception as err:
			return sardana.State.Init, str(err)

		# FAULT — One of the motors is in fault
		try: 
			states = self._icepap.get_states(
				#self._motor_addresses
				list(self._motor_addresses),				# Srsly…‽ 😒
			)
		except Exception:
			return sardana.StateFault, str(err)
		registers = " ".join(
			hex(state.status_register)
			for state in
			states
		)
		if (
			any(not state.is_present() for state in states)
		):
			return sardana.State.Fault, registers

		# ALARM — One of the motors is in alarm
		elif (
			any(not state.is_poweron() for state in states)
			or any(state.is_limit_negative() for state in states)
			or any(state.is_limit_positive() for state in states)
		):
			return sardana.State.Alarm, registers

		# MOVING — Moving along the trajectory
		elif (
			any(state.is_moving() for state in states)
			or any(state.is_settling() for state in states)
		):
			return sardana.State.Moving, registers

		# ON — Motors and trajectory defined, but not moving
		else:
			return sardana.State.On, registers

	@property
	def position(self) -> float:
		"""Parametric position (parameter value)

		The parametric motor position (the parameter) is computed from the
		`motor_positions` array. The parameter is defined to be the Euclidean
		distance between the points of the trajectory; e.g. for a simple
		2-dimensional trajectory involving motors X and Y, and parameterized by
		parameter p;

		```
		y
		│                ⊗(x3,y3);p3=52
		│                ┃
		│  (x1,y1);p1=10 ┃
		│  ⊗━━━━━━━━━━━━━⊗(x2,y2);p2=41
		│  ┃
		│  ┃
		│  ⊗(x0,y0);p0=0
		│
		│
		└────────────────────── x
		```

		As such, the parameter always starts at 0.

		N.b. the relationship between the parameter value and the 'physical
		motor' positions is defined as a look-up table and stored in the
		IcePAP, e.g.;

		Index | p  | x  | y
		------|----|----|----
		   0  | p0 | x0 | y0
		   1  | p1 | x1 | y1
		   2  | p2 | x2 | y2
		   …

		Once the tables have been loaded, the parameter value can be queried
		from the IcePAP (using the `?PARPOS` query). The IcePAP also
		interpolates to allow positions between the values defined in the
		table.

		As such, the parameter values only need to be computed once when
		loading the tables.

		"""
		self._assert_motor_aliases()
		self._assert_motor_positions()
		try:
			#
			# As parameter values are floats, difficult to check equality
			# without introducing tolerances.
			#
			# Only way of having different parameter values though is by moving
			# individual motors along the trajectory (e.g. `PMOVE <parameter>
			# <address>` instead of `PMOVE <parameter> <address> <address> …`.
			# If all motion is done from controller using `STRICT` and `GROUP`,
			# this should never happen…
			#
			# As such, just return parameter value of first motor
			#
			return [
				self._icepap[address].parpos
				for address in
				self._motor_addresses
			][0]
		except RuntimeError:
			self._controller._log.warning(
				"One or more motors not on trajectory"
				f" (motor aliases: {self.motor_aliases})"
			)
			return float("nan")

	@position.setter
	def position(self, value : float):
		self._assert_motor_aliases()
		self._assert_motor_positions()
		try:
			self._icepap.pmove(
				value,
				#self._motor_addresses,
				list(self._motor_addresses),					# Srsly…‽ 😒
				group=True,
				strict=True
			)
		except RuntimeError:
			# Try to move onto trajectory
			try:
				self._icepap.movep(
					value,
					#self._motor_addresses,
					list(self._motor_addresses),				# Srsly…‽ 😒
					group=False,
					strict=False
				)
			except RuntimeError:
				raise RuntimeError(
					"Cannot move one or more motors onto trajectory"
					" ("
					f"motor aliases: {self.motor_aliases}"
					f", trajectory position: {value}"
					")"
				) from None

	@property
	def max_position(self):
		"""Maximum parametric position

		Defined as the sum of the Euclidean distances between the motor positions.

		The parametric position will always be within the interval [0, max_position].

		"""
		return self._trajectory_positions[-1]

	@property
	def velocity(self) -> float:
		"""Parametric velocity

		Velocity along the trajectory in parametric units (trajectory length)
		per second.

		"""
		self._assert_motor_aliases()
		self._assert_motor_positions()
		#
		# Parametric velocity of a motor is always defined, even when not on
		# trajectory. As parametric velocities are floats though, difficult to
		# check equality without introducing tolerances.
		#
		# Only way of having different parametric velocities though is by
		# setting them manually from outside controller. If all motion is done
		# from this controller, should always be equal…
		#
		# As such, just return parametric velocity of first motor
		#
		return [
			self._icepap[address].parvel
			for address in
			self._motor_addresses
		][0]

	@velocity.setter
	def velocity(self, value : float):
		self._assert_motor_aliases()
		self._assert_motor_positions()
		try:
			#
			# Would be nice to do the follwing with `PARVEL` system command, but
			# not currently exposed in `icepap` library.
			#
			for address in self._motor_addresses:
				self._icepap[address].parvel = value
		except RuntimeError:
			raise RuntimeError(
				"Parametric velocity out of range"
				" ("
				f"velocity: {value}"
				f", maximum parametric velocity: {self.max_velocity}"
				")"
			) from None
		#
		# IcePAP automatically scales the acceleration time proportionally when
		# setting velocities. In the case of of parametric trajectories, this
		# is undesirable due to the non-linear relationship between parametric
		# velocity and minimium parametric acceleration time (see
		# `_IcepapDynamicTrajectoryAxis._min_acceleration`).
		#
		self.acceleration = max(self._min_accelerations)

	@property
	def max_velocity(self):
		"""Maximum parametric velocity

		Maximum velocity along the trajectory in parametric units (trajectory
		length) per second.

		Requiring physical motor velocities not to be exceded during motion
		along the trajectory imposes a first-order upper bound on the
		parametric velocity

		Requiring physical motor accelerations not to be exceded during motion
		along the trajectory imposes a further second-order upper bound on the
		parametric velocity;

		As such, the maximum parametric velocity depends on the motor
		velocities and the loaded trajectory.

		See `_IcepapDynamicTrajectoryAxis._max_velocities` docstring for
		further details.

		"""
		return min(self._max_velocities)

	@property
	def acceleration(self) -> float:
		"""Parametric acceleration time

		Time required to reach the set parametric velocity.

		"""
		self._assert_motor_aliases()
		self._assert_motor_positions()
		#
		# Parametric acceleration time of a motor is always defined, even when not on
		# trajectory. As parametric acceleration times are floats though, difficult to
		# check equality without introducing tolerances.
		#
		# Only way of having different parametric acceleration times though is by
		# setting them manually from outside controller. If all motion is done
		# from this controller, should always be equal…
		#
		# As such, just return parametric acceleration of first motor
		#
		return [
			self._icepap[address].paracct
			for address in
			self._motor_addresses
		][0]

	@acceleration.setter
	def acceleration(self, value : float):
		self._assert_motor_aliases()
		self._assert_motor_positions()
		try:
			#
			# Would be nice to do the follwing with `PARACCT` system command, but
			# not currently exposed in `icepap` library.
			#
			# N.b. Parametric acceleration not limited by motor accelerations
			#
			for address in self._motor_addresses:
				self._icepap[address].paracct = value
		except RuntimeError:
			raise RuntimeError(
				"Parametric acceleration out of range"
				" ("
				f"acceleration: {value}"
				")"
			) from None

	def stop(self):
		self._assert_motor_aliases()
		#self._assert_motor_positions()				# Stop regardless of trajectory
		self._icepap.stop(
			#self._motor_addresses
			list(self._motor_addresses),			# Srsly…‽ 😒
		)

	def abort(self):
		self._assert_motor_aliases()
		#self._assert_motor_positions()				# Abort regardless of trajectory
		self._icepap.abort(
			#self._motor_addresses
			list(self._motor_addresses),			# Srsly…‽ 😒
		)


	# Private interface ----------------------------------------------------- #

	# Property data members
	_motor_aliases : tuple[str] = None
	_motor_positions : numpy.ndarray = None

	# Private data members
	_controller : IcepapDynamicTrajectoryController = None
	_axis_index : int = None
	_motor_addresses : tuple[int] = None
	_motor_names : tuple[str] = None
	_icepap : icepap.IcePAPController = None

	@property
	def _axis_alias(self):
		return self._controller.GetAxisName(self._axis_index)

	@staticmethod
	def _device_names(
		device_aliases : typing.Sequence[str]
	) -> typing.Iterator[str]:
		"""Yield Tango device names from multiple Tango aliases"""
		database = tango.Database()
		for device_alias in device_aliases:
			try:
				yield database.get_device_from_alias(device_alias)
			except Exception:
				raise ValueError(
					"Device not defined in database"
					f" (device alias: {device_alias})"
				) from None

	@staticmethod
	def _device_properties(
		device_names : typing.Sequence[str],
		properties : typing.Sequence[str]
	) -> typing.Iterator[list[str]]:
		"""Yield Tango device property values from multiple Tango names"""
		database = tango.Database()
		for device_name in device_names:
			try:
				yield [
					value[0]
					for value in
					database.get_device_property(
						device_name,
						properties
					).values()
				]
			except Exception:
				raise ValueError(
					"One or more device properties not defined in database"
					" ("
					f"device: {device_name}"
					f", properties: {properties}"
					")"
				) from None

	@staticmethod
	def _device_attribute_properties(
		device_names : typing.Sequence[str],
		properties : typing.Sequence[str]
	) -> typing.Iterator[list[str]]:
		"""Yield Tango device attribute property values from multiple Tango names"""
		database = tango.Database()
		for device_name in device_names:
			try:
				yield [
					value["__value"][0]
					for value in
					database.get_device_attribute_property(
						device_name,
						properties
					).values()
				]
			except Exception:
				raise ValueError(
					"One or more device attribute properties not defined in database"
					" ("
					f"device: {device_name}"
					f", properties: {properties}"
					")"
				) from None

	@staticmethod
	def _set_device_attribute_ranges(
		attribute_names : typing.Sequence[str],
		attribute_ranges : typing.Sequence[tuple[float,float]]
	) -> None:
		"""Set atttribute range for multiple Tango device names"""
		for attribute_name, attribute_range in zip(
			attribute_names,
			attribute_ranges
		):
			attribute = tango.AttributeProxy(attribute_name)
			config = attribute.get_config()
			config.min_value, config.max_value = map(str,attribute_range)
			attribute.set_config(config)

	@staticmethod
	def _device_axes(
		device_names : typing.Sequence[str]
	) -> typing.Iterator[int]:
		"""Yield 'axis' property values from multiple Tango device names"""
		for axis in _IcepapDynamicTrajectoryAxis._device_properties(
			device_names,
			("axis",)
		):
			yield int(axis[0])

	@staticmethod
	def _device_controller_ids(
		device_names : typing.Sequence[str]
	) -> typing.Iterator[str]:
		"""Yield 'ctrl_id' property values from multiple Tango device names"""
		for axis in _IcepapDynamicTrajectoryAxis._device_properties(
			device_names,
			("ctrl_id",)
		):
			yield axis[0]

	@staticmethod
	def _common_icepap(
		motor_names : str
	) -> icepap.IcePAPController:
		"""Return controller's 'host' property value from multiple Sardana element names"""
		cls = _IcepapDynamicTrajectoryAxis
		hosts = [
			host[0]
			for host in
			cls._device_properties(
				cls._device_names(			# Assumes USE_NUMERIC_ELEMENT_IDS = False
					cls._device_controller_ids(
						motor_names
					)
				),
				("host",)
			)
		]
		if any(host != hosts[0] for host in hosts):
			raise ValueError(
				"Motors are on different IcePAP systems"
				f" (hosts: {hosts})"
			)
		return icepap.IcePAPController(
			host=hosts[0],
			auto_axes=True
		)

	def _assert_motor_aliases(self):
		if not self.motor_aliases:
			raise ValueError(
				"Motors not defined"
				f" (motors: {self.motor_aliases})"
			)

	def _assert_motor_positions(self):
		if self.motor_positions is None:
			raise ValueError(
				"Trajectory motor positions not defined"
				f" (motor positions: {self.motor_positions})"
			)

	def _validate_motor_aliases(
		self,
		motor_aliases : tuple[str]
	):
		if self.motor_positions is not None:
			self._validate_trajectory_shape(
				motor_aliases,
				self.motor_positions
			)
			self._validate_trajectory_bounds(
				motor_aliases,
				self.motor_positions
			)

	def _validate_motor_positions(
		self,
		motor_positions : numpy.ndarray
	):
		self._validate_trajectory_dimensions(motor_positions)
		if self.motor_aliases:
			self._validate_trajectory_shape(
				self.motor_aliases,
				motor_positions
			)
			self._validate_trajectory_bounds(
				self.motor_aliases,
				motor_positions
			)

	@staticmethod
	def _validate_trajectory_dimensions(motor_positions : numpy.ndarray):
		if motor_positions.ndim != 2:
			raise ValueError(
				"Motor positions not bi-dimensional "
				f" (motor positions dimensions: {motor_positions.ndim})"
			)

	@staticmethod
	def _validate_trajectory_shape(
		motor_aliases : tuple[str],
		motor_positions : numpy.ndarray
	):
		if motor_positions.shape[1] != len(motor_aliases):
			raise ValueError(
				"Motor positions incompatible with number of motors"
				" ("
				f"len(motors): {len(motor_aliases)}"
				f", motor_positions.shape[1]: {motor_positions.shape[1]}"
				")"
			)

	@staticmethod
	def _validate_trajectory_bounds(
		motor_aliases : tuple[str],
		motor_positions : numpy.ndarray
	):
		for motor_alias, _motor_positions in zip(
			motor_aliases,
			motor_positions.T
		):
			try:
				motor = tango.DeviceProxy(motor_alias)
				position_config = motor.get_attribute_config("position")
			except Exception:
				#
				# Cannot guarantee Pool element startup order (motor axes may
				# be instantiated _after_ parametric axes), but can only fetch
				# attribute configuration for exported devices. As such, must
				# allow this assertion to fail gracefully here
				#
				return
			min_motor_position = (
					float("-Infinity") if position_config.min_value == "Not specified"
					else float(position_config.min_value)
			)
			max_motor_position = (
					float("Infinity") if position_config.max_value == "Not specified"
					else float(position_config.max_value)
			)
			if (
				(_motor_positions.min() < min_motor_position)
				or (_motor_positions.max() > max_motor_position)
			):
				raise ValueError(
					"Motor positions outside allowed range"
					" ("
					f"motor: {motor_alias}"
					f", motor range: ({min_motor_position},{max_motor_position})"
					f", motor positions range: ({motor_positions.min()},{motor_positions.max()})"
					")"
				)

	@property
	def _trajectory_vectors(self) -> numpy.ndarray:
		"""Vectors between the motor positions

		e.g. for a 2-dimensional trajectory defined by the motor positions;

		```
			motor positions = [
				[x0,y0],
				[x1,y1],
				[x2,y2],
				[x3,y3],
				…
			]
		```

		The vectors between the motor positions would be;

		```
			trajectory_vectors = [
				[x1-x0, y1-y0],
				[x2-x1, y2-y1],
				[x3-x2, y3-y2],
				…
			]
		```

		"""
		self._assert_motor_positions()
		return numpy.diff(self.motor_positions, axis=0)

	@property
	def _trajectory_vector_magnitudes(self) -> numpy.ndarray:
		"""Magnitude of the trajectory vectors

		e.g. for a 2-dimensional trajectory defined by the motor positions;

		```
			motor positions = [
				[x0,y0],
				[x1,y1],
				[x2,y2],
				[x3,y3],
				…
			]
		```

		```
			trajectory_vector_magnitudes = [
				[√((x1-x0)² + (y1-y0)²)],
				[√((x2-x1)² + (y2-y1)²)],
				[√((x3-x2)² + (y3-y2)²)],
				…
			]
		```

		"""
		return numpy.sqrt((self._trajectory_vectors ** 2).sum(axis=1))

	@property
	def _trajectory_positions(self) -> typing.Sequence:
		"""Parametric position (parameter value)

		Defined as the cumulative sum of the Euclidean distances between the
		points of the trajectory.

		"""
		return (
			[0.0]												# Initial position
			+ self._trajectory_vector_magnitudes.cumsum().tolist()
		)

	def _load_trajectory(self):
		"""Load the trajectory look-up tables in the IcePAP drivers

		The relationship between the parameter value and a 'physical motor'
		position is defined as a look-up table and stored in the corresponding
		IcePAP driver for the `physical motor`, e.g. for a 2-dimensional
		trajectory involving motors X and Y, the table in driver X would be;

		Index | p  | x
		------|----|---
		   0  | p0 | x0
		   1  | p1 | x1
		   2  | p2 | x2
		   …

		and the table in driver Y would be;

		Index | p  | y
		------|----|---
		   0  | p0 | y0
		   1  | p1 | y1
		   2  | p2 | y2
		   …
		```

		N.b. Whilst the 'physical motor' positions in the `motor_positions`
		attribute are in physical units (e.g. mm, degrees, …), they must be
		stored in the IcePAP tables in _steps_!

		"""
		self._assert_motor_aliases()
		for motor_address, motor_steps in zip(
			self._motor_addresses,
			self._motor_steps
		):
			self._icepap[motor_address].set_parametric_table(
				self._trajectory_positions,
				motor_steps,
				mode="LINEAR",
				param_type="FLOAT",
				pos_type="DWORD",
			)

	def _clear_trajectory(self):
		"""Clear the trajectory look-up tables in the IcePAP drivers"""
		self._assert_motor_aliases()
		for address in self._motor_addresses:
			self._icepap[address].clear_parametric_table()

	@property
	def _motor_steps(self) -> typing.Iterator[numpy.ndarray]:
		"""'Physical motor' positions in steps"""
		#
		# Fetch attributes required for converting motor positions to steps
		# from the DB as cannot guarantee Pool element startup order (motor
		# axes may be instantiated _after_ parametric axes)
		#
		self._assert_motor_aliases()
		self._assert_motor_positions()
		for (motor_sign, motor_offset, motor_step_per_unit), motor_positions in zip(
			self._device_attribute_properties(
				self._motor_names,
				("sign", "offset", "step_per_unit")
			),
			self.motor_positions.T
		):
			motor_steps = (
					(motor_positions - float(motor_offset))
					* float(motor_step_per_unit)
					/ int(motor_sign)
				).round().astype(int)
			yield motor_steps

	@property
	def _motor_accelerations(self) -> numpy.ndarray:
		"""'Physical motor' acceleration times (in seconds)"""
		return numpy.array(
			self._icepap.get_acctime(
				#self._motor_addresses
				list(self._motor_addresses)				# Srsly…‽ 😒
			)
		)

	@property
	def _motor_velocities(self) -> numpy.ndarray:
		"""'Physical motor' velocities (steps/second)"""
		return numpy.array(
			self._icepap.get_velocity(
				#self._motor_addresses
				list(self._motor_addresses)				# Srsly…‽ 😒
			)
		)

	@property
	def _max_velocities(self) -> numpy.ndarray:
		"""Maximum parametric velocities

		Requiring physical motor velocities not to be exceded during motion
		along the trajectory imposes a first-order upper bound, ṗ⁽¹⁾ₘₐₓ, on the
		parametric velocity;

		    ẋ = dxₗ/dt
		       = [dx/dp](p)·[dp/dt](t)
		       = [dx/dp](p)·ṗ(t)

		    => ṗ⁽¹⁾ₘₐₓ = ẋₘₐₓ / [dx/dp]ₘₐₓ

		N.b. This is a strict requirement for IcePAP parametric trajectories —
		IcePAP will raise an error when attempting to set parametric velocities
		above ṗ⁽¹⁾ₘₐₓ.

		Requiring physical motor accelerations not to be exceded during motion
		along the trajectory imposes a second-order upper bound, ṗ⁽²⁾ₘₐₓ, on
		the parametric velocity;

		    ẍ = dẋ/dt
		      = [dx/dp](p)·p̈(t) + [d²x/dp²](p)·ṗ²(t)

		    => ẍₘₐₓ = [dxₗ/dp]ₘₐₓ·p̈ₘₐₓ + [d²x/dp²]ₘₐₓ·ṗ⁽²⁾ₘₐₓ²

		    assuming constant parametric acceleration;

		    p̈ₘₐₓ = ṗ⁽²⁾ₘₐₓ/tₚₘᵢₙ

		    => ẍₘₐₓ = [dx/dp]ₘₐₓ·(ṗ⁽²⁾ₘₐₓ/tₚₘᵢₙ) + [d²x/dp²]ₘₐₓ·ṗ⁽²⁾ₘₐₓ²

		    => tₚₘᵢₙ = ([dx/dp]ₘₐₓ·ṗ⁽²⁾ₘₐₓ) / (ẍₘₐₓ - [d²x/dp²]ₘₐₓ·ṗ⁽²⁾ₘₐₓ²)

		    which is asymptotic;

		           lim
		    ṗ⁽²⁾ₘₐₓ → √(ẍₘₐₓ/[d²x/dp²]ₘₐₓ)  ;  tₚₘᵢₙ = ∞

		    => ṗ⁽²⁾ₘₐₓ = √(ẍₘₐₓ/[d²x/dp²]ₘₐₓ)

		N.b. This _not_ a strict requirement for IcePAP parametric trajectories
		— IcePAP will  _not_ raise an error when attempting to set parametric
		velocities above ṗ⁽²⁾ₘₐₓ. Respecting this bound however significantly
		improves trajectory accuracy.

		"""
		self._assert_motor_aliases()
		self._assert_motor_positions()
		dxdp = numpy.gradient(
			numpy.array(list(self._motor_steps)).T,
			self._trajectory_positions,
			axis=0
		)
		d2xdp2 = numpy.gradient(
			dxdp,
			self._trajectory_positions,
			axis=0
		)
		dxdt_max = self._motor_velocities
		d2xdt2_max = dxdt_max / self._motor_accelerations
		dpdt_1_max = dxdt_max / abs(dxdp).max(axis=0)
		dpdt_2_max = numpy.sqrt( d2xdt2_max / abs(d2xdp2).max(axis=0) )
		return dpdt_1_max if any(dpdt_1_max < dpdt_2_max) else dpdt_2_max

	@property
	def _min_accelerations(self) -> numpy.ndarray:
		"""Minimum parametric accelerations

		Requiring physical motor accelerations not to be exceded during motion
		along the trajectory imposes a lower bound, tₚₘᵢₙ, on the parametric
		acceleration time;

		    ẍ = dẋ/dt
		      = [dx/dp](p)·p̈(t) + [d²x/dp²](p)·ṗ²(t)

		    => ẍₘₐₓ = [dxₗ/dp]ₘₐₓ·p̈ₘₐₓ + [d²x/dp²]ₘₐₓ·ṗ⁽²⁾ₘₐₓ²

		    assuming constant parametric acceleration;

		    p̈ₘₐₓ = ṗ⁽²⁾ₘₐₓ/tₚₘᵢₙ

		    => ẍₘₐₓ = [dx/dp]ₘₐₓ·(ṗ⁽²⁾ₘₐₓ/tₚₘᵢₙ) + [d²x/dp²]ₘₐₓ·ṗ⁽²⁾ₘₐₓ²

		    => tₚₘᵢₙ = ([dx/dp]ₘₐₓ·ṗ⁽²⁾ₘₐₓ) / (ẍₘₐₓ - [d²x/dp²]ₘₐₓ·ṗ⁽²⁾ₘₐₓ²)

		"""
		self._assert_motor_aliases()
		self._assert_motor_positions()
		dxdp = numpy.gradient(
			numpy.array(list(self._motor_steps)).T,
			self._trajectory_positions,
			axis=0
		)
		d2xdp2 = numpy.gradient(
			dxdp,
			self._trajectory_positions,
			axis=0
		)
		d2xdt2_max = self._motor_velocities / self._motor_accelerations
		dpdt_max = self.velocity
		return (
			(abs(dxdp).max(axis=0) * dpdt_max)
			/ (
				d2xdt2_max - (abs(d2xdp2).max() * (dpdt_max**2))
			)
		)
