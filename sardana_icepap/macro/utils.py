"""
    Macro library containing icepap related macros for the macro
    server Tango device server as part of the Sardana project.
"""
from PyTango import DeviceProxy
import icepap
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
    except Exception as e:
        macro.debug(e)
        msg = 'Icepap resets should be executed with caution. ' + \
              'It is recommended to notify the Icepap experts about the ' + \
              'reset. Automatic notifications WILL NOT be send. ' + str(e)
        raise Exception(msg)
    return author, recipients


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
    except Exception as e:
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
    ipap = icepap.IcePAPController(icepap_host)
    while not ipap.connected:
        time.sleep(0.5)

    # TODO: Implement equivalent method on icepap API 3
    # crate_list = ice_dev.getRacksAlive()
    rack_mask = int(ipap.send_cmd('?SYSSTAT')[0], 16)
    crate_list = []
    for rack in range(16):
        if rack_mask & (1 << rack) != 0:
            crate_list.append(rack)

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

    driver_list = ipap.find_axes()
    if crate_nr >= 0:
        nr = crate_nr
        driver_list = [i for i in driver_list if
                       i > (nr * 10) and i <= (nr * 10 + 8)]

    status_message = ''
    for driver in driver_list:
        status_message += 'Axis: %d\nStatus: %s\n' % \
                          (driver, ipap[driver].vstatus)

    pool_obj.SendToController([icepap_ctrl.getName(), cmd])
    msg += ' It will take aprox. 3 seconds...'
    self.info(msg)

    try:
        efrom, eto = getResetNotificationAuthorAndRecipients(self)
    except Exception as e:
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
