#!/usr/bin/env python

from tango import DevState, DevFailed
from time import sleep
from json import dumps
from sardana.macroserver.macro import Macro, Type
from sardana_icepap.macro.icepap_utils import isIcepapMotor


class ipap_esync(Macro):
    """
    Macro to ESYNC when the motor has an sync error and report it.
    The parameter should the the icepap motor
    """

    param_def = [
        ["motor", Type.Motor, None, "motor to reset (must be an IcePAP motor)"],
    ]

    def run(self, motor):
        """main method for send async"""

        try:
            self.motor = motor
            if not isIcepapMotor(self, motor):
                raise (
                    Exception(
                        "Motor: {motor} is not an Icepap motor.".format(
                            motor=self.motor.getName()
                        )
                    )
                )
            if self.motor.State() == DevState.ALARM:
                data = self.collect_data()
                self.send_esync()
                sleep(2)
                self.power_on_motor()
                self.create_log(data)
            else:
                raise (
                    Exception(
                        "Motors {motor} on ON state.".format(motor=self.motor.getName())
                    )
                )
        except Exception as e:
            self.error(str(e))

    def power_on_motor(self):
        """
        Set PowerOn attribute on motor
        """
        self.info("Power ON Motor")
        try:
            self.motor.write_attribute("PowerOn", 1)
        except Exception as e:
            self.info("ERROR while Powering On Motor: {error}".format(error=str(e)))

    def send_esync(self):
        """
        Sending 'ESYNC' command directly to controller
        """
        self.info("Sending ESYNC to {motor}".format(motor=self.motor.getName()))
        cmd = "ESYNC"
        self._send_cmd(cmd)
        self.info("ESYNC done")

    def create_log(self, data):
        """
        Create log with controller fault data

        Args:
            data (str): Data to be included in the log
        """
        self.warning(
            "macro:ipap_esync: motor {motor}: axis {axis}: {data}".format(
                motor=self.motor.getName(), axis=str(self.motor.getAxis()), data=data
            )
        )

    def collect_data(self):
        """
        Collecting controller attributes
        """
        self.info("Collecting data {motor}".format(motor=self.motor.getName()))
        data_obj = {
            "PosAxis": self._robust_attribute_read("PosAxis"),
            "PosTgtEnc": self._robust_attribute_read("PosTgtEnc"),
            "PosMotor": self._robust_attribute_read("PosMotor"),
            "EncTgtEnc": self._robust_attribute_read("EncTgtEnc"),
            "Position": self._robust_attribute_read("Position", write=True),
            "StatusAlive": self._robust_attribute_read("StatusAlive"),
            "Status5vpower": self._robust_attribute_read("Status5vpower"),
            "StatusDisable": self._robust_attribute_read("StatusDisable"),
            "StatusHome": self._robust_attribute_read("StatusHome"),
            "StatusIndexer": self._robust_attribute_read("StatusIndexer"),
            "StatusInfo": self._robust_attribute_read("StatusInfo"),
            "StatusLim-": self._robust_attribute_read("StatusLim-"),
            "StatusLim+": self._robust_attribute_read("StatusLim+"),
            "StatusMode": self._robust_attribute_read("StatusMode"),
            "StatusMoving": self._robust_attribute_read("StatusMoving"),
            "StatusOutOfWin": self._robust_attribute_read("StatusOutOfWin"),
            "StatusPowerOn": self._robust_attribute_read("StatusPowerOn"),
            "StatusPresent": self._robust_attribute_read("StatusPresent"),
            "StatusReady": self._robust_attribute_read("StatusReady"),
            "StatusSettling": self._robust_attribute_read("StatusSettling"),
            "StatusStopCode": self._robust_attribute_read("StatusStopCode"),
            "StatusWarning": self._robust_attribute_read("StatusWarning"),
        }
        self.info("Data collected")
        data = dumps(data_obj)
        return data

    def _send_cmd(self, cmd):
        """
        Helper function to send commands directly to controller

        Args:
            cmd (str): icepap command after the ":"
        """
        pool = self.motor.getPoolObj()
        ctrl_name = self.motor.getControllerName()
        axis = str(self.motor.getAxis())
        result = pool.SendToController([ctrl_name, axis + ":" + cmd])
        return result

    def _robust_attribute_read(self, attr_name, write=False):
        """
        Helper function to read attributes without reaising TangoErros.

        Args:
            attr_name (str): attribute name
            write (bool): False for value (read), True for w_value (write)
        """
        try:
            attr = self.motor.read_attribute(attr_name)
            value = attr.w_value if write else attr.value
        except DevFailed:
            value = None

        return value
