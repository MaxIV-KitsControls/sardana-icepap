"""
    Macro library containing icepap related macros for the macro
    server Tango device server as part of the Sardana project.
"""

from PyTango import DeviceProxy
import icepap
import time
from sardana.macroserver.macro import macro, Type, Macro, Optional
from sardana.macroserver.msexception import UnknownEnv

# globals
ENV_FROM = "_IcepapEmailAuthor"
ENV_TO = "_IcepapEmailRecipients"
SUBJECT = "Icepap: %s was reset by a Sardana macro"
ENV_ROCKIT = "_IcepapRockit"


# util functions
def isIcepapMotor(macro, motor):
    """Checks if pool motor belongs to the IcepapController"""

    controllers = macro.getControllers()
    ctrl_name = motor.controller
    controller_obj = controllers[ctrl_name]
    return isIcepapController(macro, controller_obj)


def isIcepapController(macro, controller):
    """Checks if pool controller is of type IcepapController"""

    if isinstance(controller, str):
        controller_name = controller
        controllers = macro.getControllers()
        controller_obj = controllers[controller_name]
    else:
        controller_obj = controller
    controller_class_name = controller_obj.getClassName()
    if controller_class_name != "IcepapController":
        return False
    return True


def fromAxisToCrateNr(axis_nr):
    """Translates axis number to crate number"""

    # TODO: add validation for wrong axis numbers
    crate_nr = axis_nr / 10
    return crate_nr


def getIcepapMotor(motor_name):
    axis = motor_name.getAxis()
    ctrl_obj = motor_name.getControllerObj()
    icepap_host = ctrl_obj.get_property("host")["host"][0]
    ipap = icepap.IcePAPController(icepap_host)
    return ipap[axis]


def restoreFromRockitEnv(macroObj):
    rockit_env = macroObj.getEnv(ENV_ROCKIT)
    motor = macroObj.rockit_motor
    try:
        start_pos = rockit_env[motor.name]["startPos"]
        original_vel = rockit_env[motor.name]["velocity"]

        macroObj.info("Stopping motor %s" % motor.name)
        macroObj.debug(motor.status())
        macroObj.debug(motor.state)
        macroObj.ipap_motor.stop()
        timeout = 3
        initime = time.time()
        while macroObj.ipap_motor.state_moving:
            if time.time() - initime > timeout:
                raise Exception("Failed to stop motor %s" % motor.name)
            time.sleep(0.5)
        macroObj.debug(motor.status())
        macroObj.debug(motor.state)

        macroObj.info("Returning %s to initial position %f" % (motor.name, start_pos))
        motor.move(start_pos)

        macroObj.info("Restoring %s velocity to %f" % (motor.name, original_vel))
        motor.write_attribute("velocity", original_vel)
        # remove motor info from ENV
        rockit_env.pop(motor.name)
        macroObj.setEnv(ENV_ROCKIT, rockit_env)
    except KeyError:
        macroObj.error(
            "Motor %s not found in env %s, cannot restore pos/vel"
            % (motor.name, ENV_ROCKIT)
        )
        return


def sendMail(efrom, eto, subject, message):
    """sends email using smtp"""

    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText
    import smtplib

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = efrom
    msg["To"] = ",".join(eto)
    body = MIMEText(message)
    msg.attach(body)
    smtp = smtplib.SMTP("localhost")
    smtp.sendmail(msg["From"], msg["To"], msg.as_string())
    smtp.quit()


def waitSeconds(macro, seconds):
    """an "abort safe" wait"""

    for i in range(seconds):
        time.sleep(1)
        macro.checkPoint()


def getResetNotificationAuthorAndRecipients(macro):
    """gets a recipients list and author from the environment variable.
    In case the variable is not defined it rises a verbose exception"""
    try:
        recipients = macro.getEnv(ENV_TO)
        if not (isinstance(recipients, list) and len(recipients)):
            msg = '"%s" variable is not a list or is empty.' % ENV_TO
            raise Exception(msg)
        author = macro.getEnv(ENV_FROM)
        if not (isinstance(author, str) and len(author)):
            msg = '"%s" variable is not a string or is empty.' % ENV_FROM
            raise Exception(msg)
    except Exception as e:
        macro.debug(e)
        msg = (
            "Icepap resets should be executed with caution. "
            + "It is recommended to notify the Icepap experts about the "
            + "reset. Automatic notifications WILL NOT be send. "
            + str(e)
        )
        raise Exception(msg)
    return author, recipients


@macro(
    [
        ["motor", Type.Motor, None, "motor to jog"],
        ["velocity", Type.Integer, None, "velocity"],
    ]
)
def ipap_jog(self, motor, velocity):
    poolObj = motor.getPoolObj()
    ctrlName = motor.getControllerName()
    axis = motor.getAxis()
    poolObj.SendToController([ctrlName, "%d: JOG %d" % (axis, velocity)])


@macro()
def ipap_rockit_list(self):
    try:
        rockit_env = self.getEnv(ENV_ROCKIT)
    except UnknownEnv:
        self.error("Rockit ENV var not found")
        return
    self.info("The following motors are currently ROCKING:")
    for motor in rockit_env.keys():
        self.info("- %s: %s" % (motor, rockit_env[motor]["rockit"]))


@macro([["motor", Type.Motor, Optional, None, "motor to stop rockit"]])
def ipap_rockit_stop(self, motor):
    if motor is None:
        self.error("Please provide a motor to stop.")
        self.execMacro("ipap_rockit_list")
        return
    self.rockit_motor = motor
    self.ipap_motor = getIcepapMotor(motor)
    restoreFromRockitEnv(self)


class ipap_rockit(Macro):
    """Moves continuously a motor back and forth. It can be launched in
    the background (for several motors if necesary) and then motion
    can be stopped with the macro ipap_rockit_stop <motor>.
    Original position and velocity are stored in an _IcepapRockit variable
    and recovered when rockit is stopped"""

    param_def = [
        ["motor", Type.Motor, None, "motor to move"],
        [
            "rockit_range",
            Type.Float,
            None,
            "Move between [current - range / 2, current + range / 2]",
        ],
        ["background", Type.Boolean, False, "Run in background (default = False)"],
        ["velocity", Type.Float, Optional, "velocity (default = current)"],
        [
            "force",
            Type.Boolean,
            False,
            "Ignore previous saves in environment (default = False)",
        ],
    ]

    def run(self, motor, rockit_range, background, velocity, force):
        self.rockit_motor = motor
        self.ipap_motor = getIcepapMotor(motor)

        startPos = motor.read_attribute("position").value
        # Check position limits and velocity
        low_pos = startPos - rockit_range / 2.0
        high_pos = startPos + rockit_range / 2.0
        pos_obj = motor.getPositionObj()
        min_pos, max_pos = pos_obj.getRange()
        if low_pos < min_pos:
            self.error("Position below low user limit (%f), aborting" % min_pos)
            return
        if high_pos > max_pos:
            self.error("Position above high user limit (%f), aborting" % max_pos)
            return

        if velocity is not None:
            vel_obj = motor.getVelocityObj()
            min_vel, max_vel = vel_obj.getRange()
            if velocity < min_vel:
                self.error("Velocity below minimum allowed (%f), aborting" % min_vel)
                return
            if velocity > max_vel:
                self.error("Velocity above maximum allowed (%f), aborting" % max_vel)
                return
            motor.write_attribute("velocity", velocity)
            rockit_vel = velocity
        else:
            rockit_vel = motor.read_attribute("velocity").value

        ipap_rockit_info = {
            "startPos": startPos,
            "velocity": motor.read_attribute("velocity").value,
            "rockit": "From %f to %f at velocity=%f" % (low_pos, high_pos, rockit_vel),
        }

        # Save position and velocity to env var to recover when stopping
        # If there are already saved values for the motor, notify and abort
        try:
            rockit_env = self.getEnv(ENV_ROCKIT)
        except UnknownEnv:
            rockit_env = {}
        if motor.name in rockit_env.keys() and force is False:
            self.error(
                "Rockit info for %s already stored in " % motor.name
                + "env var %s, aborting (use force=True to overwrite)" % (ENV_ROCKIT)
            )
            return
        rockit_env[motor.name] = ipap_rockit_info
        self.setEnv(ENV_ROCKIT, rockit_env)

        # Convert to step units (needed for icepap ltrack command)
        dial_pos = motor.read_attribute("dialposition").value
        step_per_unit = motor.read_attribute("step_per_unit").value
        rockit_half_range_steps = (rockit_range * step_per_unit) / 2.0
        start_pos_steps = dial_pos * step_per_unit
        low_pos_steps = start_pos_steps - rockit_half_range_steps
        high_pos_steps = start_pos_steps + rockit_half_range_steps

        self.debug(
            "Rockit in steps: low_pos={} start_pos={} high_pos={}".format(
                low_pos_steps, start_pos_steps, high_pos_steps
            )
        )

        # Run the rockit movement
        self.ipap_motor.set_list_table([low_pos_steps, start_pos_steps, high_pos_steps])
        self.ipap_motor.ltrack(signal="", mode="CYCLIC")

        if not background:
            self.info("Rocking {}, Ctrl+C to stop".format(motor.name))
            while self.ipap_motor.state_moving:
                self.outputBlock(
                    "{} {}".format(motor.name, motor.read_attribute("position").value)
                )
                time.sleep(0.5)

    def on_abort(self):
        restoreFromRockitEnv(self)


@macro([["motor", Type.Motor, None, "motor to reset"]])
def ipap_reset_motor(self, motor):
    """Resets a crate where the Icepap motor belongs to. This will send an
    autmatic notification to recipients declared
    in '_IcepapEmailRecipients' variable"""

    motor_name = motor.getName()
    if not isIcepapMotor(self, motor):
        self.error("Motor: %s is not an Icepap motor" % motor_name)
        return
    pool_obj = motor.getPoolObj()
    ctrl_name = motor.getControllerName()
    ctrl_obj = motor.getControllerObj()
    icepap_host = ctrl_obj.get_property("host")["host"][0]
    axis_nr = motor.getAxis()
    crate_nr = fromAxisToCrateNr(axis_nr)
    status = motor.read_attribute("StatusDetails").value
    cmd = "RESET %d" % crate_nr
    self.debug("Sending command: %s" % cmd)
    pool_obj.SendToController([ctrl_name, cmd])
    msg = (
        "Crate nr: %d of the Icepap host: %s " % (crate_nr, icepap_host)
        + "is being reset. It will take a while..."
    )
    self.info(msg)

    waitSeconds(self, 5)
    self.debug("RESET finished")
    # _initCrate(self, ctrl_obj, crate_nr)

    try:
        efrom, eto = getResetNotificationAuthorAndRecipients(self)
    except Exception as e:
        self.warning(e)
        return

    ms = self.getMacroServer()
    ms_name = ms.get_name()
    efrom = "%s <%s>" % (ms_name, efrom)
    subject = SUBJECT % icepap_host
    message = "Summary:\n"
    message += "Macro: ipap_reset_motor(%s)\n" % motor_name
    message += "Pool name: %s\n" % pool_obj.name()
    message += "Controller name: %s\n" % ctrl_name
    message += "Motor name: %s\n" % motor_name
    message += "Icepap host: %s\n" % icepap_host
    message += "Axis: %s\n" % axis_nr
    message += "Status: %s\n" % status
    sendMail(efrom, eto, subject, message)
    self.info("Email notification was send to: %s" % eto)
    # waiting 3 seconds so the Icepap recovers after the reset
    # it is a dummy wait, probably it could poll the Icepap
    # and break if the reset is already finished


#    waitSeconds(self, 3)


@macro(
    [
        ["icepap_ctrl", Type.Controller, None, "icepap controller name"],
        ["crate_nr", Type.Integer, -1, "crate_nr"],
    ]
)
def ipap_reset(self, icepap_ctrl, crate_nr):
    """Resets Icepap. This will send an autmatic notification to recipients
    declared in '_IcepapEmailRecipients' variable"""

    if not isIcepapController(self, icepap_ctrl):
        self.error("Controller: %s is not an Icepap controller" % icepap_ctrl.getName())
        return
    ctrl_obj = icepap_ctrl.getObj()
    pool_obj = ctrl_obj.getPoolObj()
    icepap_host = ctrl_obj.get_property("host")["host"][0]
    ipap = icepap.IcePAPController(icepap_host)
    while not ipap.connected:
        time.sleep(0.5)

    # TODO: Implement equivalent method on icepap API 3
    # crate_list = ice_dev.getRacksAlive()
    rack_mask = int(ipap.send_cmd("?SYSSTAT")[0], 16)
    crate_list = []
    for rack in range(16):
        if rack_mask & (1 << rack) != 0:
            crate_list.append(rack)

    if crate_nr >= 0:
        msg = (
            "Crate nr: %d of the Icepap host: " % crate_nr
            + "%s is being reset." % icepap_host
        )
        if crate_nr in crate_list:
            cmd = "RESET %d" % crate_nr
        else:
            self.error("The crate number is not valid")
            return
    else:
        msg = "Icepap host: %s is being reset." % icepap_host
        cmd = "RESET"

    driver_list = ipap.find_axes()
    if crate_nr >= 0:
        nr = crate_nr
        driver_list = [i for i in driver_list if i > (nr * 10) and i <= (nr * 10 + 8)]

    status_message = ""
    for driver in driver_list:
        status_message += "Axis: %d\nStatus: %s\n" % (driver, ipap[driver].vstatus)

    pool_obj.SendToController([icepap_ctrl.getName(), cmd])
    msg += " It will take aprox. 3 seconds..."
    self.info(msg)

    try:
        efrom, eto = getResetNotificationAuthorAndRecipients(self)
    except Exception as e:
        self.warning(e)
        return

    ms = self.getMacroServer()
    ms_name = ms.get_name()
    efrom = "%s <%s>" % (ms_name, efrom)
    subject = SUBJECT % icepap_host
    ctrl_name = icepap_ctrl.getName()
    message = "Macro: %s(%s)\n" % (self.getName(), ctrl_name)
    message += "Pool name: %s\n" % pool_obj.name()
    message += "Controller name: %s\n" % ctrl_name
    message += "Icepap host: %s\n" % icepap_host
    if crate_nr >= 0:
        message += "Crate: %d\n" % crate_nr
    message += status_message
    sendMail(efrom, eto, subject, message)
    self.info("Email notification was send to: %s" % eto)
    # waiting 3 seconds so the Icepap recovers after the0 reset
    # it is a dummy wait, probably it could poll the Icepap
    # and break if the reset is already finished
    waitSeconds(self, 3)


def _initCrate(macro, ctrl_obj, crate_nr):
    # It initializes all axis found in the same crate
    # than the target motor given.
    # We could have decided to initialize all motors in the controller.

    # Define axes range to re-initialize after reset
    # These are the motors in the same crate than the given motor
    first = crate_nr * 10
    last = first + 8
    macro.info("Initializing Crate number %s:" % crate_nr)
    macro.info("axes range [%s,%s]" % (first, last))

    # Get the alias for ALL motors for the controller
    motor_list = ctrl_obj.elementlist
    macro.debug("Element in controller: %s" % repr(motor_list))

    # Crate a proxy to each element and
    # get the axis for each of them
    for alias in motor_list:
        m = DeviceProxy(alias)
        a = int(m.get_property("axis")["axis"][0])
        # Execute init command for certain motors:
        if first <= a <= last:
            macro.debug("alias: %s" % alias)
            macro.debug("device name: %s" % m.name())
            macro.debug("axis number: %s" % a)
            macro.info("Initializing %s..." % alias)
            try:
                m.command_inout("Init")
            # HOTFIX!!! only if offsets are lost 24/12/2016
            # print 'IMPORTANT: OVERWRITTING centx/centy offsets!'
            # if alias == 'centx':
            #    centx_offset = -4.065240223463690
            #    m['offset'] = centx_offset
            #    print 'centx offset overwritten: %f' % centx_offset
            # if alias == 'centy':
            #    centy_offset = -2.759407821229050
            #    m['offset'] = centy_offset
            #    print 'centy offset overwritten: %f' % centy_offset

            except Exception:
                macro.error("axis %s cannot be initialized" % alias)
