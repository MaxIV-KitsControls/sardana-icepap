import pytest
import json

from tango import DevState, DevFailed


class attribute:
    value = None
    w_value = None

    def __init__(self, attr_name):
        self.w_value = "w_value_" + attr_name
        self.value = "value_" + attr_name


@pytest.fixture
def motor(mocker):
    pool = mocker.MagicMock()
    pool.SendToController.side_effect = lambda x: x[1]

    motor = mocker.MagicMock()
    motor = motor.return_value
    motor.State.return_value = DevState.ALARM
    motor.controller = "ctrl"
    motor.getControllerName.return_value = "ctrl_name"
    motor.getAxis.return_value = 1
    motor.getName.return_value = "mtr_name"
    motor.read_attribute.side_effect = lambda x: attribute(x)
    motor.getPoolObj.return_value = pool

    return motor


@pytest.fixture
def esync(mocker):
    mocker.patch("sardana.macroserver.macro.Type")
    from sardana_icepap.macro.icepap_esync import ipap_esync

    mocker.patch.object(ipap_esync, "door", return_value="macro")
    mocker.patch.object(ipap_esync, "parent_macro", return_value="macro")

    controller = mocker.MagicMock()
    controller.getClassName.return_value = "IcepapController"

    door = mocker.MagicMock()
    door.return_value.get_macro.return_value = "esync"

    macro_mock = ipap_esync()
    macro_mock.getControllers = mocker.MagicMock()
    macro_mock.getControllers.return_value = {"ctrl": controller}
    macro_mock.warning = mocker.MagicMock()
    macro_mock.error = mocker.MagicMock()
    macro_mock.info = mocker.MagicMock()

    return macro_mock


def test_not_icepap(mocker, esync, motor):
    wrong_controller = mocker.MagicMock()
    wrong_controller.getClassName.return_value = "NotIcepapController"
    esync.getControllers.return_value = {"ctrl": wrong_controller}
    esync.motor = motor

    esync.run(motor)
    esync.error.assert_called_once_with("Motor: mtr_name is not an Icepap motor.")


def test_run(esync, motor):
    esync.run(motor)


def test_run_motor_on(esync, motor):
    esync.motor = motor
    esync.motor.State.return_value = DevState.ON
    esync.run(motor)
    esync.error.assert_called_once_with("Motors mtr_name on ON state.")


def test_power_on_motor(esync, motor):
    esync.motor = motor
    esync.power_on_motor()
    esync.motor.write_attribute.assert_called_once_with("PowerOn", 1)


def test_power_on_motor_fail(esync, motor):
    def write_attribute_error(*kargs):
        raise (Exception("Motor error"))

    esync.motor = motor
    esync.motor.write_attribute.side_effect = write_attribute_error
    esync.power_on_motor()
    esync.info.assert_called_with("ERROR while Powering On Motor: Motor error")


def test_send_esync(esync, motor):
    esync.motor = motor
    esync.send_esync()
    esync.motor.getPoolObj().SendToController.assert_called_once_with(
        ["ctrl_name", "1:ESYNC"]
    )


def test_create_log(esync, motor):
    data = "data"
    esync.motor = motor
    esync.create_log(data)
    esync.warning.assert_called_once_with(
        "macro:ipap_esync: motor mtr_name: axis 1: data"
    )


def test_collect_data(mocker, esync, motor):
    esync.motor = motor
    result = esync.collect_data()

    esync.motor.read_attribute.assert_has_calls(
        [
            mocker.call("PosAxis"),
            mocker.call("PosTgtEnc"),
            mocker.call("PosMotor"),
            mocker.call("EncTgtEnc"),
            mocker.call("Position"),
            mocker.call("StatusAlive"),
            mocker.call("Status5vpower"),
            mocker.call("StatusDisable"),
            mocker.call("StatusHome"),
            mocker.call("StatusIndexer"),
            mocker.call("StatusInfo"),
            mocker.call("StatusLim-"),
            mocker.call("StatusLim+"),
            mocker.call("StatusMode"),
            mocker.call("StatusMoving"),
            mocker.call("StatusOutOfWin"),
            mocker.call("StatusPowerOn"),
            mocker.call("StatusPresent"),
            mocker.call("StatusReady"),
            mocker.call("StatusSettling"),
            mocker.call("StatusStopCode"),
            mocker.call("StatusWarning"),
        ]
    )

    assert json.loads(result) == {
        "PosAxis": "value_PosAxis",
        "PosTgtEnc": "value_PosTgtEnc",
        "PosMotor": "value_PosMotor",
        "EncTgtEnc": "value_EncTgtEnc",
        "Position": "w_value_Position",
        "StatusAlive": "value_StatusAlive",
        "Status5vpower": "value_Status5vpower",
        "StatusDisable": "value_StatusDisable",
        "StatusHome": "value_StatusHome",
        "StatusIndexer": "value_StatusIndexer",
        "StatusInfo": "value_StatusInfo",
        "StatusLim-": "value_StatusLim-",
        "StatusLim+": "value_StatusLim+",
        "StatusMode": "value_StatusMode",
        "StatusMoving": "value_StatusMoving",
        "StatusOutOfWin": "value_StatusOutOfWin",
        "StatusPowerOn": "value_StatusPowerOn",
        "StatusPresent": "value_StatusPresent",
        "StatusReady": "value_StatusReady",
        "StatusSettling": "value_StatusSettling",
        "StatusStopCode": "value_StatusStopCode",
        "StatusWarning": "value_StatusWarning",
    }


def test_send_cmd(esync, motor):
    cmd = "command"

    esync.motor = motor
    result = esync._send_cmd(cmd)

    esync.motor.getPoolObj().SendToController.assert_called_once_with(
        ["ctrl_name", "1:command"]
    )

    assert result == "1:command"


@pytest.mark.parametrize(
    ["write", "expected_result", "fail"],
    (
        [False, "value_attribute", False],
        [True, "w_value_attribute", False],
        [False, None, True],
    ),
)
def test_robust_attribute_read(esync, motor, write, expected_result, fail):
    att_name = "attribute"

    esync.motor = motor
    if fail:

        def raise_DevFail(x):
            raise (DevFailed)

        esync.motor.read_attribute.side_effect = raise_DevFail
    result = esync._robust_attribute_read(att_name, write)

    esync.motor.read_attribute.assert_called_once_with(att_name)

    assert result == expected_result
