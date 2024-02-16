import sardana.macroserver.macro
import time

class ipap_resync(sardana.macroserver.macro.Macro):
	"""Macro to resynchronize IcePAP linked axes"""

	# Sardana Macro API #######################################################

	# Constants
	DEFAULT_REFERENCE_ADDRESS = -1

	# Macro
	param_def = (
		(
			"motor",
			sardana.macroserver.macro.Type.Motor,
			None,
			"Linked axis motor to resynchronize."
		),
		(
			"addr_ref",
			sardana.macroserver.macro.Type.Integer,
			DEFAULT_REFERENCE_ADDRESS,
			(
				"Reference motor address to synchronize to."
				+ " Must be part of the linked axis motor."
				+ " Defaults to first address in linked axis motor."
			)
		)
	)
	interactive = True

	def prepare(self, motor, addr_ref):

		try:

			# Get Sardana objects
			self.motor = motor
			self.controller = self.getController(self.motor.controller)
			self.pool = self.motor.getPoolObj()

			# Assert motor is IcePAP linked axis
			if self.controller.getClassName() != "IcepapLinkedAxisController":
				raise ValueError(
					f"Motor '{self.motor}' (axis {self.motor.axis})"
					+ " is not an IcePAP linked axis motor"
				)

			# Init lnkname and addresses
			self._init_lnkname_and_addrs()

			# Init reference address
			self._init_addr_ref(addr_ref)

			# Init control encoder errors
			self._init_ctrlerrors()

			# Init closed loop statuses
			self.pcl_mask = list(self._get_pcls(self.addrs))


		except Exception as err:
			self.error(str(err))

	def run(self, motor, addr_ref):

		self.info(f"Resynchronizing {self.motor}")
		self.debug(f"Resynchronizing {self.addrs} to {self.addr_ref}")

		try:

			# Disable PCL
			self.info("Disabling position closed loop")
			self.debug(
				"Initial PCL states: "
				+ f"{{addr:pcl for addr,pcl in zip(self.addrs, self.pcl_mask)}}"
			)
			self._set_pcls(
				(
					addr
					for addr, enabled in
					zip(self.addrs, self.pcl_mask)
					if enabled
				),
				enabled=False
			)

			# Power on motors
			self.info("Powering on motors")
			self.debug(f"Powering on {self.lnkname} == {self.addrs}")
			self._set_powers(self.addrs, on=True)

			# Align motors (sychronize all measure registers)
			self.info("Aligning motors")
			deltas = list(self._delta_pos_measures())
			while any(deltas):
				self._align(deltas)
				time.sleep(1)			# FIXME: End of motion detection
				deltas = list(self._delta_pos_measures())
			if any(self._delta_pos_measures()):
				self.debug(
					f"Measure registers: {self._get_pos(self.addrs, 'measure')}"
				)
				raise ValueError("Could not align motors")

			# Synchronize axis registers to measure registers (esync)
			self.info("Synchronizing axis registers to measure registers")
			self._esync(self.addrs)

			# Synchronize control registers to measure registers (ctrlrst)
			self.info("Synchronizing control registers to measure registers")
			self._ctrlrst(self.addrs)

		except Exception as err:
			self.error(str(err))
			self.info("Re-enabling position closed loop")
			self._set_pcls(
				(
					addr
					for addr, enabled in
					zip(self.addrs, self.pcl_mask)
					if enabled
				),
				enabled=True
			)
			self.error(f"Resynchronization of {self.motor} aborted")

		else:
			self.info(f"Resynchronization of {self.motor} complete")


	# Private #################################################################

	# Sardana elements
	motor = None
	controller = None
	pool = None

	# IcePAP
	lnkname = None
	addrs = None
	addr_ref = None
	pcl_mask = None
	ctrlerrors = None

	def _init_lnkname_and_addrs(self):
		"""Query lnkname and addresses from IcePAP

		* These are not available from the Sardana controller (only one
		  address, viz. motor's)

		"""
		response = self.pool.SendToController(
			(
				self.controller.name,
				f"?linked"
			)
		)							# N.b. SendToController eats line-breaks — all one string
		lnknames = {}
		for word in response.split():
			if not word.isdecimal():
				lnkname = word
				lnknames[lnkname] = []
			else:
				lnknames[lnkname].append(int(word))
		for lnkname, addrs in lnknames.items():
			if self.motor.axis in addrs:
				self.lnkname = lnkname
				self.addrs = addrs
		
		if (self.lnkname is None) or (self.addrs is None):
			raise ValueError(
				f"Motor '{self.motor}' (axis {self.motor.axis})"
				+ " is not an IcePAP linked axis motor"
			)

	def _init_addr_ref(self, addr_ref):
		"""Validate reference address

		* Default to first address in linked axis motor if not supplied or not
		  in linked axis motor

		"""
		if addr_ref in self.addrs:
			self.addr_ref = addr_ref
		else:
			self.addr_ref = self.addrs[0]
			if addr_ref != self.DEFAULT_REFERENCE_ADDRESS:
				self.warning(
					f"Reference motor address {addr_ref} not part of linked"
					+ f" axis motor {self.motor}. Defaulting to first address"
					+ f" in linked axis motor (address {self.addr_ref})"
				)

	def _init_ctrlerrors(self):
		"""Query control encoder error windows from IcePAP

		* IcePAP configuration parameter — not available from Sardana

		"""
		self.ctrlerrors = [
			int(
				self.pool.SendToController(
					(
						self.controller.name,
						f"{addr}:?cfg ctrlerror"
					)
				).split()[-1]
			)
			for addr
			in self.addrs
		]

	def _get_pcls(self, addrs):
		"""Query position closed loop states

		* These are not available from Sardana as the contituent motors of a
		  linked axis are not typically deployed

		"""
		for addr in addrs:
			yield self.pool.SendToController(
				(
					self.controller.name,
					f"{addr}:?pcloop"
				)
			).lower() == "on"
	
	def _set_pcls(self, addrs, enabled):
		"""Enable/disable position closed loop

		* These are not available from Sardana as the contituent motors of a
		  linked axis are not typically deployed

		"""
		states = {
			False: "off",
			True: "on"
		}
		for addr in addrs:
			self.pool.SendToController(
				(
					self.controller.name,
					f"{addr}:disprot linked;"
					+ f"{addr}:pcloop {states[enabled]}"
				)
			)
		if not all(pcl == enabled for pcl in self._get_pcls(addrs)):
			raise ValueError(
				f"Could not set PCLs of address {addrs} to {enabled}"
			)

	def _get_powers(self, addrs):
		"""Query power states

		* These are not available from Sardana as the contituent motors of a
		  linked axis are not typically deployed

		"""
		for power in self.pool.SendToController(
			(
				self.controller.name,
				f"?power " + ' '.join(map(str, addrs))
			)
		).strip().split():
			yield power.lower() == "on"

	def _set_powers(self, addrs, on):
		"""Power on/off motors

		* These are not available from Sardana as the contituent motors of a
		  linked axis are not typically deployed

		"""
		states = {
			False: "off",
			True: "on"
		}
		addrs_str = " ".join(map(str, addrs))
		self.pool.SendToController(
			(
				self.controller.name,
				f"disprot linked hardctrl {addrs_str};"
				+ f"power {states[on]} {addrs_str}"
			)
		)
		if not all(power == on for power in self._get_powers(addrs)):
			raise ValueError(
				f"Could not set power of address {addrs} to {on}"
			)

	def _get_pos(self, addrs, register):
		"""Query power states

		* These are not available from Sardana as the contituent motors of a
		  linked axis are not typically deployed

		"""
		for pos in map(
			int,
			self.pool.SendToController(
				(
					self.controller.name,
					f"?pos {register} " + ' '.join(map(str, addrs))
				)
			).strip().split()
		):
			yield pos

	def _delta_pos_measures(self):
		pos_measures = list(self._get_pos(self.addrs, "measure"))
		pos_measure_ref = pos_measures[self.addrs.index(self.addr_ref)]
		for pos_measure in pos_measures:
			yield pos_measure_ref - pos_measure

	def _align(self, deltas):
		self.print(
			"Deltas: "
			+ str(
				{
					addr: -delta / self.motor.step_per_unit
					for addr, delta in
					zip(self.addrs, deltas)
				}
			)
		)
		addrs = [
			addr
			for addr, delta in
			zip(self.addrs, deltas)
			if delta
		]
		steps = [
			delta if abs(delta) <= ctrlerror
			else ((-1)**int(delta < 0)) * ctrlerror
			for delta, ctrlerror in
			zip(deltas, self.ctrlerrors)
			if delta
		]
		units = [
			_steps / self.motor.step_per_unit
			for _steps in
			steps
		]
		if self.input(
			f"Move motors {dict(zip(addrs,units))} [y/N]?: "
		).lower() != "y":
			raise RuntimeError("Aborting alignment")
		self.debug(f"Alignment motions: {dict(zip(addrs, steps))}")
		self.pool.SendToController(
			(
				self.controller.name,
				f"disprot linked hardctrl {' '.join(map(str, addrs))};"
				+ "rmove "
				+ ' '.join(
					' '.join(map(str, addr_steps))
						for addr_steps in
						zip(addrs, steps)
				)
			)
		)

	def _esync(self, addrs):
		addrs_str = " ".join(map(str, addrs))
		self.pool.SendToController(
			(
				self.controller.name,
				f"disprot linked {addrs_str};"
				+ f"esync {addrs_str}"
			)
		)
		if any(
			axis != measure
			for axis, measure in
			zip(
				self._get_pos(addrs, "axis"),
				self._get_pos(addrs, "measure")
			)
		):
			raise ValueError(
				f"Axis registers not synchronized to measure registers"
			)

	def _ctrlrst(self, addrs):
		addrs_str = " ".join(map(str, addrs))
		self.pool.SendToController(
			(
				self.controller.name,
				f"disprot linked {addrs_str};"
				+ f"ctrlrst {addrs_str}"
			)
		)
		if any(
			axis != measure
			for axis, measure in
			zip(
				self._get_pos(addrs, "ctrlenc"),
				self._get_pos(addrs, "measure")
			)
		):
			raise ValueError(
				f"Control registers not synchronized to measure registers"
			)
