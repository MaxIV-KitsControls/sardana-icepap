"""
    Macro library containing icepap related macros for the macro
    server Tango device server as part of the Sardana project.
"""
from PyTango import DeviceProxy
import pyIcePAP
import time
from sardana.macroserver.macro import *

# globals
ENV_FROM = '_IcepapEmailAuthor'
ENV_TO = '_IcepapEmailRecipients'
SUBJECT = 'Icepap: %s was reset by a Sardana macro'


# util functions
def isIcepapMotor(macro, motor):
    '''Checks if pool motor belongs to the IcepapController'''

    controllers = macro.getControllers()
    ctrl_name = motor.controller
    controller_obj = controllers[ctrl_name]
    return isIcepapController(macro, controller_obj)


def isIcepapController(macro, controller):
    '''Checks if pool controller is of type IcepapController'''

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
    '''Translates axis number to crate number'''

    # TODO: add validation for wrong axis numbers
    crate_nr = axis_nr / 10
    return crate_nr


def sendMail(efrom, eto, subject, message):
    '''sends email using smtp'''

    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText
    import smtplib
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = efrom
    msg["To"] = ','.join(eto)
    body = MIMEText(message)
    msg.attach(body)
    smtp = smtplib.SMTP('localhost')
    smtp.sendmail(msg["From"], msg["To"], msg.as_string())
    smtp.quit()


def waitSeconds(macro, seconds):
    '''an "abort safe" wait'''

    for i in range(seconds):
        time.sleep(1)
        macro.checkPoint()


def getResetNotificationAuthorAndRecipients(macro):
    '''gets a recipients list and author from the environment variable.
       In case the variable is not defined it rises a verbose exception'''
    try:
        recipients = macro.getEnv(ENV_TO)
        if not (isinstance(recipients, list) and len(recipients)):
            msg = '"%s" variable is not a list or is empty.' % ENV_TO
            raise Exception(msg)
        author = macro.getEnv(ENV_FROM)
        if not (isinstance(author, str) and len(author)):
            msg = '"%s" variable is not a string or is empty.' % ENV_FROM
            raise Exception(msg)
    except Exception, e:
        macro.debug(e)
        msg = 'Icepap resets should be executed with caution. ' + \
              'It is recommended to notify the Icepap experts about the ' + \
              'reset. Automatic notifications WILL NOT be send. ' + str(e)
        raise Exception(msg)
    return author, recipients


# macros
class ipap_get_closed_loop(Macro):
    """Returns current closed loop configuration value for a given motor"""

    param_def = [
        ["motor", Type.Motor, None,
         "motor to request (must be and IcePAP motor)"],
    ]

    icepap_ctrl = "IcePAPCtrl.py"

    def prepare(self, motor):
        """Check that parameters for the macro are correct"""
        motorOk = False

        # check that motor controller is of type icepap
        controller = motor.getControllerName()
        pool = motor.getPoolObj()
        ctrls_list = pool.read_attribute("ControllerList").value
        for ctrl in ctrls_list:
            found = ctrl.find(controller)
            if found >= 0:
                if ctrl.find(self.icepap_ctrl) >= 0:
                    motorOk = True
                break
        if not motorOk:
            raise Exception("Motor %s is not an IcePAP motor" % str(motor))

    def run(self, motor):
        """Run macro"""

        if motor.read_attribute("ClosedLoop").value:
            status = "ON"
        else:
            status = "OFF"

        self.output("Closed loop is %s in motor %s" % (status, str(motor)))
        return status


class ipap_set_closed_loop(Macro):
    """Enables/Disables closed loop in a given motor"""

    param_def = [
        ["motor", Type.Motor, None,
         "motor to configure (must be and IcePAP motor)"],
        ["ON/OFF", Type.String, None,
         "ON to enable / OFF to disable closed loop"]
    ]

    icepap_ctrl = "IcePAPCtrl.py"
    actions = ("ON", "OFF")

    def prepare(self, motor, action):
        """Check that parameters for the macro are correct"""
        motorOk = False

        # check that motor controller is of type icepap
        controller = motor.getControllerName()
        pool = motor.getPoolObj()
        ctrls_list = pool.read_attribute("ControllerList").value
        for ctrl in ctrls_list:
            found = ctrl.find(controller)
            if found >= 0:
                if ctrl.find(self.icepap_ctrl) >= 0:
                    motorOk = True
                break
        if not motorOk:
            raise Exception("Motor %s is not an IcePAP motor" % str(motor))

        # check that "action" is valid
        if action.upper() in self.actions:
            pass
        else:
            raise Exception("action must be one of: %s" % str(self.actions))

    def run(self, motor, action):
        """Run macro"""
        action = action.upper()

        if action == "ON":
            closed_loop = True
        else:
            closed_loop = False
        # read back closed loop status to check it's the one we have just set
        motor.write_attribute("ClosedLoop", closed_loop)
        closed_loop_rb = motor.read_attribute("ClosedLoop").value
        if closed_loop == closed_loop_rb:
            self.output("Closed loop %s correctly set in motor %s" % (
            action, str(motor)))
            return True
        else:
            self.output(
                "WARNING!: read back from the controller (%s) didn't match the requested parameter (%s)" % (
                str(closed_loop_rb), str(closed_loop)))
            return False


class ipap_get_steps_per_turn(Macro):
    """Returns current steps per turn value for a given motor"""

    param_def = [
        ["motor", Type.Motor, None,
         "motor to request (must be and IcePAP motor)"],
    ]

    icepap_ctrl = "IcePAPCtrl.py"
    config_command = "%d:CONFIG"
    get_command = "%d:?CFG ANSTEP"

    def prepare(self, motor):
        """Check that parameters for the macro are correct"""
        motorOk = False

        # check that motor controller is of type icepap
        controller = motor.getControllerName()
        pool = motor.getPoolObj()
        ctrls_list = pool.read_attribute("ControllerList").value
        for ctrl in ctrls_list:
            found = ctrl.find(controller)
            if found >= 0:
                if ctrl.find(self.icepap_ctrl) >= 0:
                    motorOk = True
                break
        if not motorOk:
            raise Exception(
                "Motor %s doesn't support closed loop" % str(motor))

    def run(self, motor):
        """Run macro"""
        # get axis number, controller name and pool
        axis = motor.getAxis()
        controller = motor.getControllerName()
        pool = motor.getPoolObj()

        # write command to icepap
        cmd = self.get_command % axis
        result = pool.SendToController([controller, cmd])

        # read result and return value
        steps = result.split()[2]
        self.output("% s steps per turn in motor %s" % (steps, str(motor)))
        return int(steps)


class ipap_set_steps_per_turn(Macro):
    """Set steps per turn value for a given motor"""

    param_def = [
        ["motor", Type.Motor, None,
         "motor to configure (must be and IcePAP motor)"],
        ["steps", Type.Integer, None, "steps per turn value"]
    ]

    icepap_ctrl = "IcePAPCtrl.py"
    config_command = "%d:CONFIG"
    set_command = "%d:CFG ANSTEP %d"
    get_command = "%d:?CFG ANSTEP"
    sign_command = "%d:CONFIG %s"

    def prepare(self, motor, steps):
        """Check that parameters for the macro are correct"""
        motorOk = False

        # check that motor controller is of type iceepap
        controller = motor.getControllerName()
        pool = motor.getPoolObj()
        ctrls_list = pool.read_attribute("ControllerList").value
        for ctrl in ctrls_list:
            found = ctrl.find(controller)
            if found >= 0:
                if ctrl.find(self.icepap_ctrl) >= 0:
                    motorOk = True
                break
        if not motorOk:
            raise Exception("Motor %s is not an IcePAP motor" % str(motor))

    def run(self, motor, steps):
        """Run macro"""
        # get axis number, controller name and pool
        axis = motor.getAxis()
        controller = motor.getControllerName()
        pool = motor.getPoolObj()

        # set controller in config mode
        cmd = self.config_command % axis
        result = pool.SendToController([controller, cmd])

        # set the requested steps per turn
        cmd = self.set_command % (axis, steps)
        result = pool.SendToController([controller, cmd])

        # sign the change in icepap controller
        cmd = self.sign_command % (axis,
                                   "Steps per turn changed on user request from macro %s on %s" % (
                                   str(self.__class__.__name__),
                                   str(time.ctime())))
        result = pool.SendToController([controller, cmd])

        # read back steps per turn to confirm command worked OK
        cmd = self.get_command % axis
        result = pool.SendToController([controller, cmd])
        steps_rb = int(result.split()[2])
        if steps == steps_rb:
            self.output("Steps per turn %d correctly set in motor %s" % (
            steps, str(motor)))
            return True
        else:
            self.output(
                "WARNING!: read back from the controller (%s) didn't match the requested parameter (%d)" % (
                str(result), steps))
            return False


@macro([["motor", Type.Motor, None, "motor to jog"],
        ["velocity", Type.Integer, None, "velocity"]])
def ipap_jog(self, motor, velocity):
    poolObj = motor.getPoolObj()
    ctrlName = motor.getControllerName()
    axis = motor.getAxis()
    poolObj.SendToController([ctrlName, "%d: JOG %d" % (axis, velocity)])


@macro([["motor", Type.Motor, None, "motor to reset"]])
def ipap_reset_motor(self, motor):
    '''Resets a crate where the Icepap motor belongs to. This will send an
       autmatic notification to recipients declared
       in '_IcepapEmailRecipients' variable'''

    motor_name = motor.getName()
    if not isIcepapMotor(self, motor):
        self.error('Motor: %s is not an Icepap motor' % motor_name)
        return
    pool_obj = motor.getPoolObj()
    ctrl_name = motor.getControllerName()
    ctrl_obj = motor.getControllerObj()
    icepap_host = ctrl_obj.get_property('host')['host'][0]
    axis_nr = motor.getAxis()
    crate_nr = fromAxisToCrateNr(axis_nr)
    status = motor.read_attribute('StatusDetails').value
    cmd = "RESET %d" % crate_nr
    self.debug('Sending command: %s' % cmd)
    pool_obj.SendToController([ctrl_name, cmd])
    msg = 'Crate nr: %d of the Icepap host: %s ' % (crate_nr, icepap_host) + \
          'is being reset. It will take a while...'
    self.info(msg)

    waitSeconds(self, 5)
    self.debug("RESET finished")
    # _initCrate(self, ctrl_obj, crate_nr)

    try:
        efrom, eto = getResetNotificationAuthorAndRecipients(self)
    except Exception, e:
        self.warning(e)
        return

    ms = self.getMacroServer()
    ms_name = ms.get_name()
    efrom = '%s <%s>' % (ms_name, efrom)
    subject = SUBJECT % icepap_host
    message = 'Summary:\n'
    message += 'Macro: ipap_reset_motor(%s)\n' % motor_name
    message += 'Pool name: %s\n' % pool_obj.name()
    message += 'Controller name: %s\n' % ctrl_name
    message += 'Motor name: %s\n' % motor_name
    message += 'Icepap host: %s\n' % icepap_host
    message += 'Axis: %s\n' % axis_nr
    message += 'Status: %s\n' % status
    sendMail(efrom, eto, subject, message)
    self.info('Email notification was send to: %s' % eto)
    # waiting 3 seconds so the Icepap recovers after the reset
    # it is a dummy wait, probably it could poll the Icepap
    # and break if the reset is already finished


#    waitSeconds(self, 3)


@macro([["icepap_ctrl", Type.Controller, None, "icepap controller name"],
        ["crate_nr", Type.Integer, -1, "crate_nr"]])
def ipap_reset(self, icepap_ctrl, crate_nr):
    """Resets Icepap. This will send an autmatic notification to recipients
       declared in '_IcepapEmailRecipients' variable"""

    if not isIcepapController(self, icepap_ctrl):
        self.error('Controller: %s is not an Icepap controller' % \
                   icepap_ctrl.getName())
        return
    ctrl_obj = icepap_ctrl.getObj()
    pool_obj = ctrl_obj.getPoolObj()
    icepap_host = ctrl_obj.get_property('host')['host'][0]
    ice_dev = pyIcePAP.EthIcePAP(icepap_host, 5000)
    while not ice_dev.connected:
        time.sleep(0.5)

    crate_list = ice_dev.getRacksAlive()
    if crate_nr >= 0:
        msg = 'Crate nr: %d of the Icepap host: ' % crate_nr + \
              '%s is being reset.' % icepap_host
        if crate_nr in crate_list:
            cmd = "RESET %d" % crate_nr
        else:
            self.error('The crate number is not valid')
            return
    else:
        msg = 'Icepap host: %s is being reset.' % icepap_host
        cmd = "RESET"

    driver_list = ice_dev.getDriversAlive()
    if crate_nr >= 0:
        nr = crate_nr
        driver_list = [i for i in driver_list if
                       i > (nr * 10) and i <= (nr * 10 + 8)]

    status_message = ''
    for driver in driver_list:
        status_message += 'Axis: %d\nStatus: %s\n' % \
                          (driver, ice_dev.getVStatus(driver))

    pool_obj.SendToController([icepap_ctrl.getName(), cmd])
    msg += ' It will take aprox. 3 seconds...'
    self.info(msg)

    try:
        efrom, eto = getResetNotificationAuthorAndRecipients(self)
    except Exception, e:
        self.warning(e)
        return

    ms = self.getMacroServer()
    ms_name = ms.get_name()
    efrom = '%s <%s>' % (ms_name, efrom)
    subject = SUBJECT % icepap_host
    ctrl_name = icepap_ctrl.getName()
    message = 'Macro: %s(%s)\n' % (self.getName(), ctrl_name)
    message += 'Pool name: %s\n' % pool_obj.name()
    message += 'Controller name: %s\n' % ctrl_name
    message += 'Icepap host: %s\n' % icepap_host
    if crate_nr >= 0:
        message += 'Crate: %d\n' % crate_nr
    message += status_message
    sendMail(efrom, eto, subject, message)
    self.info('Email notification was send to: %s' % eto)
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
    macro.info('Initializing Crate number %s:' % crate_nr)
    macro.info('axes range [%s,%s]' % (first, last))

    # Get the alias for ALL motors for the controller
    motor_list = ctrl_obj.elementlist
    macro.debug("Element in controller: %s" % repr(motor_list))

    # Crate a proxy to each element and
    # get the axis for each of them
    for alias in motor_list:
        m = DeviceProxy(alias)
        a = int(m.get_property('axis')['axis'][0])
        # Execute init command for certain motors:
        if first <= a <= last:
            macro.debug('alias: %s' % alias)
            macro.debug('device name: %s' % m.name())
            macro.debug('axis number: %s' % a)
            macro.info("Initializing %s..." % alias)
            try:
                m.command_inout('Init')
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
                macro.error('axis %s cannot be initialized' % alias)
