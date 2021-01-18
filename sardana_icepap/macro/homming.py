import time
import PyTango
from sardana.macroserver.macro import Table, Macro, Type

TIMEOUT_LIM = 1


def create_motor_info_dict(motor, direction):
    """
    Creates a dictionary with motor informations (which is required by homing
    functions).
    It has follwing keys('motor','direction','homed','status','position',
    'home_ice_pos','encin').
    Motor and direction values are set with the function arguments

    :param motor: (motor obj) motor to be homed
    :param direction: (int) homing direction - in pool sense <-1,1>

    :return: (dictionary) dictionary with motor info"""

    return {'motor': motor,
            'direction': direction,
            'homed': False,
            'status': None,
            'position': None,
            'home_ice_pos': None,
            'encin': None}


def populate_homing_commands(motors, directions, group=False, strict=False):
    """
    Populates a set of icepap homing commands: homing command, homing status
    command, homing position command, homing encoder command, abort command.

    :param motors: (list<motor obj>) list of motors to be homed
    :param directions: (list<int>) list of homing directions -
                       in pool sense <-1,1>
    :param group: group homing
    :param strict: strict homing
    :return: (list<str>) list of homing commands"""

    homing_cmd = 'home'
    homing_status_cmd = '?homestat'
    homing_pos_cmd = '?homepos {0}'
    homing_encin_cmd = '?homeenc encin {0}'
    abort_cmd = 'stop'

    if group is True:
        homing_cmd += ' group'

    if strict is True:
        homing_cmd += ' strict'

    for m, d in zip(motors, directions):
        icepap_axis = m.getAxis()
        icepap_direction = m.getSign() * d
        homing_cmd += ' %d %d' % (icepap_axis, icepap_direction)
        homing_status_cmd += ' %s' % icepap_axis
        abort_cmd += ' %s' % icepap_axis

    return homing_cmd, homing_status_cmd, homing_pos_cmd, homing_encin_cmd, \
        abort_cmd


def output_homing_status(macro, motorsInfoList):
    """
    Flushes homing status to the door output attribute.
    Status is represented in a table with homed state, home icepap position,
    home status, and motor positions as
    columns and motors as rows.

    :param macro: (macro obj) macro which will perform homing
    :param motorsInfoList: (list<dict>) list of motors info dictionary"""

    rowHead = []
    colHead = [['homed'], ['home icepap pos'], ['status'], ['position']]
    data = [[], [], [], []]
    for motInfo in motorsInfoList:
        rowHead.append(motInfo['motor'].alias())
        data[0].append(str(motInfo['homed']))
        data[1].append(str(motInfo['home_ice_pos']))
        data[2].append(str(motInfo['status']))
        data[3].append(str(motInfo['position']))

    table = Table(data, elem_fmt=['%*s'], term_width=None,
                  col_head_str=colHead, col_head_fmt='%*s', col_head_width=15,
                  row_head_str=rowHead, row_head_fmt='%-*s', row_head_width=15,
                  col_sep='|', row_sep='_', col_head_sep='-', border='=')
    output = ''
    output += '{0}\n'.format(colHead)
    output += '{0}\n'.format(rowHead)
    output += '{0}'.format(data)
    for l in table.genOutput():
        output += '\n{0}'.format(l)
    macro.outputBlock(output)
    macro.flushOutput()


def home(macro, motorsInfoList, group=False, strict=False):
    """Performs icepap homing routine.

    :param macro: (macro obj) macro which will perform homing
    :param motorsInfoList: (list<dict>) list of motors info dictionaries

    :return: (boolean) True if all motors finds home, False in all other cases
    """

    someMotor = motorsInfoList[0]['motor']
    pool = someMotor.getPoolObj()
    ctrlName = someMotor.getControllerName()
    macro.debug('Pool: %s, Controller: %s' % (repr(pool), ctrlName))

    motors = [i.get('motor') for i in motorsInfoList]
    hmDirections = [i.get('direction') for i in motorsInfoList]

    if strict is True:
        group = True

    HM_CMD, HM_STATUS_CMD, HM_POS_CMD, HM_ENCIN_CMD, ABORT_CMD = \
        populate_homing_commands(motors, hmDirections, group=group,
                                 strict=strict)
    macro.debug('HM_CMD: %s', HM_CMD)
    macro.debug('HM_STATUS_CMD: %s', HM_STATUS_CMD)
    macro.debug('HM_POS_CMD: %s', HM_POS_CMD)
    macro.debug('HM_ENCIN_CMD: %s', HM_ENCIN_CMD)

    ans = pool.SendToController([ctrlName, HM_CMD])
    if ans.startswith('HOME ERROR'):
        macro.error('Could not start icepap homing routine: %s', HM_CMD)
        macro.error('Icepap response: %s', ans)
        return False
    timeouts = 0
    try:
        while True:
            macro.checkPoint()
            ans = pool.SendToController([ctrlName, HM_STATUS_CMD])
            homeStats = ans.split()[0::2]
            macro.debug('Home stats: %s' % repr(homeStats))
            # updating motor info dictionaries
            for i, motInfo in enumerate(motorsInfoList):
                motor = motInfo['motor']
                motor_pos = motor.getAttribute('Position').getDisplayValue()
                motInfo['position'] = motor_pos
                macro.debug('Motor: %s, position: %s', motor.alias(),
                            motInfo['position'])
                homingStatus = homeStats[i]
                motInfo['status'] = homingStatus
                if homingStatus == 'FOUND':
                    motInfo['homed'] = True
                    axis = motor.getAxis()
                    ans = pool.SendToController([ctrlName,
                                                 HM_POS_CMD.format(axis)])
                    motInfo['home_ice_pos'] = ans
                    ans = pool.SendToController([ctrlName,
                                                 HM_ENCIN_CMD.format(axis)])
                    motInfo['encin'] = ans
            # refreshing output table
            output_homing_status(macro, motorsInfoList)

            # checking ending condition
            if not any([stat == 'MOVING' for stat in homeStats]):
                if any([stat == 'NOTFOUND' for stat in homeStats]):
                    return False
                else:
                    return True
            time.sleep(1)
            timeouts = 0
    except PyTango.DevFailed:
        timeouts += 1
        if timeouts > TIMEOUT_LIM:
            pool.SendToController([ctrlName, ABORT_CMD])
            macro.abort()


def home_group_strict(macro, motorsInfoList):
    return home(macro, motorsInfoList, group=True, strict=True)


def home_group(macro, motorsInfoList):
    return home(macro, motorsInfoList, group=True, strict=False)


def home_strict(macro, motorsInfoList):
    return home(macro, motorsInfoList, group=False, strict=True)


class ipap_homing(Macro):
    """
    This macro will execute an icepap homing routine for all motors passed
    as arguments in directions passes as arguments.
    Directions are considered in pool sense.

    Icepap homing routine is parametrizable in group and strict sense, so it
    has 4 possible configurations.
    Macro result depends on the configuration which you have chosen:
       - HOME (macro result is True if all the motors finds home, otherwise
         result is False)
       - HOME GROUP (macro result is True if all the motors finds home,
         otherwise result is False)
       - HOME STRICT (macro result is True when first motor finds home,
         otherwise result is False)
       - HOME GROUP STRICT (macro result is True when first motor finds home,
         otherwise result is False)
    """

    param_def = [
        ["group", Type.Boolean, False, "If performed group homing."],
        ["strict", Type.Boolean, False, "If performed strict homing."],
        ['motor_direction_list',
         [['motor', Type.Motor, None, 'Motor to be homed.'],
          ['direction', Type.Integer, None,
           'Direction of homing (in pool sense) <-1|1>'],
          {'min': 1}], None, 'List of motors and homing directions.']
    ]

    result_def = [
        ['homed', Type.Boolean, None, 'Motors homed state']
    ]

    def prepare(self, *args, **opts):
        self.group = args[0]
        self.strict = args[1]
        self.motors = []

        motors_directions = args[2]
        self.motorsInfoList = [create_motor_info_dict(m, d) for m, d in
                               motors_directions]

        # getting motion object for automatic aborting
        motorNames = [motorInfoDict['motor'].name for motorInfoDict in
                      self.motorsInfoList]
        self.getMotion(motorNames)

    def run(self, *args, **opts):
        return home(self, self.motorsInfoList, self.group, self.strict)
