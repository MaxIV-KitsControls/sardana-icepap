from PyTango import DeviceProxy
import time
import taurus
import json
from sardana.macroserver.macro import Macro, Type


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
                "WARNING!: read back from the controller (%s) didn't "
                "match the requested parameter (%s)" % (
                str(closed_loop_rb), str(closed_loop)))
            return False


# ---------------------------------------------------------------
# TODO: Check the use case: Does not work in firmware 3.17 

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
        self.info(result)
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
        cmd = self.sign_command % \
              (axis,
               "Steps per turn changed on user request from macro %s on "
               "%s" % (str(self.__class__.__name__), str(time.ctime())))
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
                "WARNING!: read back from the controller (%s) "
                "didn't match the requested parameter "
                "(%d)" % (str(result), steps))
            return False

# ---------------------------------------------------------------


class ipap_tg_config(Macro):
    """
    Macro to configure the MoveableOnInput of the trigger
    """
    param_def = [
        ["tg", Type.TriggerGate, None,
         "Icepap multiplexor trigger gate (axis=0)"]
    ]

    def run(self, tg):
        tg = DeviceProxy(tg.name)
        if int(tg.get_property('axis')['axis'][0]) != 0:
           raise ValueError('Only valid for multiplexor axis')

        ipap_ctrl_name_tg = tg.name().split('/')[1]
        ipap_ctrl_tg = self.getController(ipap_ctrl_name_tg)
        ipap_ctrl_name = \
        ipap_ctrl_tg.get_property('IcepapCtrlAlias')['IcepapCtrlAlias'][0]
        ipap_ctrl = self.getController(ipap_ctrl_name)

        config = {}
        for element in ipap_ctrl.elementlist:
            proxy = taurus.Device(element)
            fullname = proxy.name
            axis = int(proxy.get_property('axis')['axis'][0])
            config[fullname] = axis
        encoder = json.JSONEncoder()
        config_encoded = encoder.encode(config)
        tg.MoveableOnInput = config_encoded


