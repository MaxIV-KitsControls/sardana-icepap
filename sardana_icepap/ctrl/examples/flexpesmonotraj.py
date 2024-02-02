import numpy as np
from sardana.pool.controller import (
    Type,
    Description,
    DefaultValue,
    Access,
    DataAccess,
    FSet,
    FGet,
    Memorize,
    MemorizedNoInit,
)
from sardana import State
from icepaptrajctrl import IPAPTrajCtrl
import math
import time
from scipy import constants


DEFAULT_BEAMLINE_BRANCH = "a"
DEFAULT_GRATING = "pg400"
MIN_TRAJ_ENERGY = 1.0  # eV
MAX_TRAJ_ENERGY = 2000.0  # eV


class PGMTrajCtrl(IPAPTrajCtrl):

    axis_attributes = IPAPTrajCtrl.addAttributes(
        {
            "Cff": {
                Type: float,
                DefaultValue: 2.25,
                Access: DataAccess.ReadWrite,
                FGet: "getCff",
                FSet: "setCff",
                Memorize: MemorizedNoInit,
            },
            "LineDensity": {
                Type: float,
                DefaultValue: 300.0,
                Access: DataAccess.ReadWrite,
                FGet: "getLineDensity",
                FSet: "setLineDensity",
                Memorize: MemorizedNoInit,
            },
            "DiffrOrder": {
                Type: float,
                DefaultValue: 1.0,
                Access: DataAccess.ReadWrite,
                FGet: "getDiffrOrder",
                FSet: "setDiffrOrder",
                Memorize: MemorizedNoInit,
            },
            "OffsetGr": {
                Type: float,
                DefaultValue: 0.0,
                Access: DataAccess.ReadWrite,
                FGet: "getOffsetGr",
                FSet: "setOffsetGr",
                Memorize: MemorizedNoInit,
            },
            "OffsetM": {
                Type: float,
                DefaultValue: 0.0,
                Access: DataAccess.ReadWrite,
                FGet: "getOffsetM",
                FSet: "setOffsetM",
                Memorize: MemorizedNoInit,
            },
            "EnergyReal": {
                Type: float,
                Access: DataAccess.ReadOnly,
                FGet: "getEnergyReal",
            },
            "EnergyRaw": {
                Type: float,
                Access: DataAccess.ReadOnly,
                FGet: "getEnergyRaw",
            },
            "CffReal": {Type: float, Access: DataAccess.ReadOnly, FGet: "getCffReal"},
            "BeamlineBranch": {
                Type: str,
                Access: DataAccess.ReadWrite,
                FGet: "getBeamlineBranch",
                FSet: "setBeamlineBranch",
                Memorize: MemorizedNoInit,
            },
            "Grating": {
                Type: str,
                Access: DataAccess.ReadWrite,
                FGet: "getGrating",
                FSet: "setGrating",
                Memorize: MemorizedNoInit,
            },
            "MaxGratingAngle": {
                Type: float,
                DefaultValue: 200.0,
                Access: DataAccess.ReadWrite,
                FGet: "getMaxGratingAngle",
                FSet: "setMaxGratingAngle",
                Memorize: MemorizedNoInit,
            },
        }
    )

    extra_properties = {
        "GrazingAngleSign": {
            Description: "The sign should be negative if the angles get larger as the motors \
                                 are moving in the negative direction.",
            Type: float,
            DefaultValue: 1.0,
        },
        "AngleConversion": {
            Description: "milliradians if 1000.0, degrees if 57.29577 (360/(2*math.pi)",
            Type: float,
            DefaultValue: 1.0,
        },
        "MaximumRatio": {
            Description: "Maximum ratio between grating and mirror positions",
            Type: float,
            DefaultValue: 1.84,
        },
        "mono_equations_a_branch_pg400": {
            Description: "The terms describing the mono energy equations for A-branch and Grating 400, Ecorr = a0 + a1*E + a2*E^2 + a3*E^3",
            Type: (float,),
            DefaultValue: [-0.19013, 1.0023, -4.2083e-05, 5.5158e-09],
        },
        "mono_equations_b_branch_pg400": {
            Description: "The terms describing the mono energy equations for B-branch and Grating 400, Ecorr = a0 + a1*E + a2*E^2 + a3*E^3",
            Type: (float,),
            DefaultValue: [-0.06236, 1.0024, -4.4037e-05, 6.3212e-09],
        },
        "mono_equations_a_branch_pg1221": {
            Description: "The terms describing the mono energy equations for A-branch Grating 1221, Ecorr = a0 + a1*E + a2*E^2 + a3*E^3",
            Type: (float,),
            DefaultValue: [-0.19013, 1.0023, -4.2083e-05, 5.5158e-09],
        },
        "mono_equations_b_branch_pg1221": {
            Description: "The terms describing the mono energy equations for B-branch Grating 1221, Ecorr = a0 + a1*E + a2*E^2 + a3*E^3",
            Type: (float,),
            DefaultValue: [-0.06236, 1.0024, -4.4037e-05, 6.3212e-09],
        },
    }

    trajectory_parameters = (
        "Cff",
        "LineDensity",
        "DiffrOrder",
        "OffsetGr",
        "OffsetM",
        "MaxGratingAngle",
    )

    motor_roles = ("mirror", "grating")
    # ctrl_properties = IPAPTrajCtrl.addMotorRoles(motor_roles)
    master_motor = "mirror"

    extra_properties.update(IPAPTrajCtrl.makeMotorRoleProps(motor_roles))
    ctrl_properties = IPAPTrajCtrl.addProperties(extra_properties)

    hc = (
        constants.h * constants.c / constants.elementary_charge
    )  # Planck * speed of light, Unit: meter electronvolts

    def __init__(self, inst, props, *args, **kwargs):
        self._beamline_branch = None
        self._grating = None
        self._prev_position_time = 0
        IPAPTrajCtrl.__init__(self, inst, props, *args, **kwargs)
        self._log.debug("Init done")

    def _calc_trajectory_coords(self, param_min, param_max, nbrpoints):
        return np.logspace(np.log10(param_min), np.log10(param_max), num=nbrpoints)

    def calc_positions(self, energy, parameters):
        """
        From a given calibrated energy, we calculate the physical
        position  for the real motors using the desired Cff value.
        """
        # print "energy: ", energy
        cff, grx_mm, diff_ord, ofs_g, ofs_m, _gratinglimit = parameters
        # print parameters
        grx = grx_mm * 1000  # Convert from lines/mm to lines/m
        # print "lkla"
        mono_energy = self.calc_energy_raw(energy)

        try:
            m, g = self.calculate_mirrorgrating_angles(
                mono_energy,
                cff,
                grx,
                offsetg=ofs_g,
                offsetm=ofs_m,
                grazinganglesign=self.GrazingAngleSign,
                diffrorder=diff_ord,
                angleconv=self.AngleConversion,
            )
        except Exception as e:
            self._log.error("error calculating positions: {}".format(e))
            raise
        # print "m, g: ", m, g
        return (m, g)

    def calc_energy_raw(self, energy):
        """
        From the calibrated energy pseudomotor position, we calculate the non-calibrated energy pseudomotor position
        """
        mono_terms = self.get_mono_terms()

        mono_energy = (
            mono_terms[0]
            + mono_terms[1] * math.pow((energy), 1)
            + mono_terms[2] * math.pow((energy), 2)
            + mono_terms[3] * math.pow((energy), 3)
        )

        return mono_energy

    def calc_pseudo_raw(self, parameters_dict):
        """
        From the real motor positions, we calculate the non-calibrated energy pseudomotor position and the real Cff value.
        """
        mirror = self._mirror_pos
        grating = self._grating_pos
        cff = parameters_dict["Cff"]
        grx = parameters_dict["LineDensity"] * 1000.0
        diff_ord = parameters_dict["DiffrOrder"]
        ofs_g = parameters_dict["OffsetGr"]
        ofs_m = parameters_dict["OffsetM"]

        energy, cff = self.calculate_energycff(
            mirror,
            grating,
            grx,
            offsetg=ofs_g,
            offsetm=ofs_m,
            grazinganglesign=self.GrazingAngleSign,
            diffrorder=diff_ord,
            angleconv=self.AngleConversion,
        )
        return (energy, cff)

    def calc_pseudo(self, mono_energy):
        """
        From the non-calibrated energy pseudomotor position, we calculate the calibrated energy pseudomotor position
        """
        mono_terms = self.get_mono_terms()

        coeffs = (
            mono_terms[3],
            mono_terms[2],
            mono_terms[1],
            mono_terms[0] - mono_energy,
        )

        roots = np.roots(np.asarray(coeffs))
        for root in roots:
            if np.iscomplex(root):  # disregard complex roots
                continue

            energy = root.real

        return energy

    @staticmethod
    def calculate_mirrorgrating_angles(
        energy,
        cff,
        grx,
        offsetm=0.0,
        offsetg=0.0,
        grazinganglesign=1.0,
        diffrorder=1.0,
        angleconv=1.0,
    ):
        """
        Calculate the mirror and grating positions from the cff, energy (eV) and grating line density = grx (l/m).
        The grazinganglesign parameter signifies the sign of the angles which depends on the
        direction the motors move to increase the angle of the mirror and grating.
        In cases of Toyama monochromators, such as Hippie and Veritas, the angles should be positive
        while for Species, manufactured by FMB, the angles are converted to negative (grazinganglesign set to -1.0)
        as those motors approach negative as they move towards the floor.
        The parameter angleconv specifies the angleformat: milliradians (angleconv=1000) or degrees (angleconv=57.2958=PI/180).
        The calculations are done in radians.
        """
        # assert type(energy) == float
        # assert type(cff) == float
        # assert type(grx) == float

        if energy == 0.0:
            wavelength = 0.0
        else:
            wavelength = PGMTrajCtrl.hc / energy

        f1 = cff**2 + 1
        f2 = 1 - cff**2
        K = diffrorder * wavelength * grx

        CosAlpha = math.sqrt(
            -1 * K**2 * f1 + 2 * math.fabs(K) * math.sqrt(f2**2 + cff**2 * K**2)
        ) / math.fabs(f2)
        alpha = math.acos(CosAlpha)

        CosBeta = cff * CosAlpha
        beta = -math.acos(CosBeta)

        theta = (alpha - beta) * 0.5

        m = ((math.pi / 2.0 - theta) * angleconv + offsetm) * grazinganglesign
        g = ((beta + math.pi / 2.0) * angleconv + offsetg) * grazinganglesign

        return m, g

    @staticmethod
    def calculate_energycff(
        mirror,
        grating,
        grx,
        offsetm=0.0,
        offsetg=0.0,
        grazinganglesign=1.0,
        diffrorder=1.0,
        angleconv=1.0,
    ):
        """
        Calculate the energy from mirror and grating positions and grating line density = grx (l/m).
        Also calculates the fixed focus condition (Cff)
        The grazinganglesign parameter signifies the sign of the angles which depends on the
        direction the motors move to increase the angle of the mirror and grating.
        In cases of Toyama monochromators, such as Hippie and Veritas, the angles should be positive
        while for Species, manufactured by FMB, the angles are converted to negative (grazinganglesign set to -1.0)
        as those motors approach negative as they move towards the floor.
        The parameter angleconv specifies the angleformat: milliradians (angleconv=1000) or degrees (angleconv=57.2958=PI/180).
        The calculations are done in radians.
        """
        # print("mirror {}, grating {}, grx {}, offsetm {}, offsetg {}, grazinganglesign {}, diffrorder {}, angleconv {}".format(mirror, grating, grx, offsetm, offsetg, grazinganglesign, diffrorder, angleconv))
        beta = (
            grazinganglesign * grating / angleconv - math.pi / 2.0 - offsetg / angleconv
        )
        theta = (
            math.pi / 2.0 - grazinganglesign * mirror / angleconv + offsetm / angleconv
        )
        alpha = 2.0 * theta + beta
        wavelength = (math.sin(alpha) + math.sin(beta)) / (diffrorder * grx)

        if wavelength == 0.0:
            energy = 0.0
        else:
            energy = math.fabs(PGMTrajCtrl.hc / wavelength)

        cff = math.fabs(math.cos(beta) / math.cos(alpha))
        return energy, cff

    def calc_limits(self, parameters_dict):
        """
        Update the parameter limits for self.active_linedensity
        and self.active_focus.
        """
        cff = parameters_dict["Cff"]
        grx = parameters_dict["LineDensity"] * 1000.0
        diff_ord = parameters_dict["DiffrOrder"]
        ofs_g = parameters_dict["OffsetGr"]
        ofs_m = parameters_dict["OffsetM"]
        max_grating_ang = parameters_dict["MaxGratingAngle"]
        # cff, grx_mm, diff_ord, ofs_g, ofs_m = parameters
        # grx = grx_mm*1000 #Convert from lines/mm to lines/m
        # walk around to find max energy
        traj_coord_max = MAX_TRAJ_ENERGY
        traj_coord_min = MIN_TRAJ_ENERGY
        startenergy = traj_coord_min
        stopenergy = traj_coord_max

        # TODO add check for max angle

        min_range = 1e-9
        try:
            for k in range(100):
                angle_limit_energy = (startenergy + stopenergy) / 2
                m, g = self.calculate_mirrorgrating_angles(
                    angle_limit_energy,
                    cff,
                    grx,
                    offsetg=ofs_g,
                    offsetm=ofs_m,
                    grazinganglesign=self.GrazingAngleSign,
                    diffrorder=diff_ord,
                    angleconv=self.AngleConversion,
                )
                # print "mid: ", angle_limit_energy, g
                if g > max_grating_ang:
                    startenergy = angle_limit_energy
                else:
                    stopenergy = angle_limit_energy
                if (stopenergy - startenergy) < min_range:
                    break
        except Exception as e:
            self._log.error("Error calculating limits: {}".format(e))

        startenergy = traj_coord_min
        stopenergy = traj_coord_max
        try:
            for k in range(100):
                cff_limit_energy = (startenergy + stopenergy) / 2
                m, g = self.calculate_mirrorgrating_angles(
                    cff_limit_energy,
                    cff,
                    grx,
                    offsetg=ofs_g,
                    offsetm=ofs_m,
                    grazinganglesign=self.GrazingAngleSign,
                    diffrorder=diff_ord,
                    angleconv=self.AngleConversion,
                )
                new_ratio = g / m
                # print "mid: ", cff_limit_energy, new_ratio
                if new_ratio > self.MaximumRatio:
                    startenergy = cff_limit_energy
                else:
                    stopenergy = cff_limit_energy
                if (stopenergy - startenergy) < min_range:
                    break
        except Exception as e:
            self._log.error("Error calculating limits: {}".format(e))
        traj_coord_min = max((cff_limit_energy, angle_limit_energy)) + 0.1
        # add 0.1eV to avoid including the limit
        self._log.debug(
            "Calculated limits: {} - {}".format(traj_coord_min, traj_coord_max)
        )
        return traj_coord_min, traj_coord_max

    def calc_parameter_limits(self):
        """
        Get a dictionary with the limits for all parameter values.
        The default implementation returs None for both min and max, override as needed.
        """
        limits = super().calc_parameter_limits()
        cff_max = self.calc_cff_max(self.active_parameters_dict)
        limits["Cff"]["max"] = cff_max
        limits["Cff"]["min"] = 1.4
        return limits

    def check_positions_safe(self, positions):
        """
        Check if a combination of positions is safe.
        Default is always safe, override if needed!
        """
        m, g = positions
        new_ratio = g / m
        safe = new_ratio < self.MaximumRatio
        return safe

    def calc_cff_max(self, parameters_dict):
        """
        Update the parameter limits for self.active_linedensity
        and self.active_focus.
        """
        self._log.trace("Parameters: {}".format(parameters_dict))
        # cff = parameters_dict["Cff"]
        grx = parameters_dict["LineDensity"] * 1000.0
        diff_ord = parameters_dict["DiffrOrder"]
        ofs_g = parameters_dict["OffsetGr"]
        ofs_m = parameters_dict["OffsetM"]
        # cff, grx_mm, diff_ord, ofs_g, ofs_m = parameters
        # grx = grx_mm*1000 #Convert from lines/mm to lines/m
        # walk around to find max energy
        startcff = 1.01
        stopcff = 1000
        energy = self.ReadOne(1)
        self._log.trace("Energy: {}".format(energy))
        for k in range(100):
            midcff = (startcff + stopcff) / 2
            m, g = self.calculate_mirrorgrating_angles(
                energy,
                midcff,
                grx,
                offsetg=ofs_g,
                offsetm=ofs_m,
                grazinganglesign=self.GrazingAngleSign,
                diffrorder=diff_ord,
                angleconv=self.AngleConversion,
            )
            new_ratio = g / m
            self._log.trace("Mid: {}, ratio: {}".format(midcff, new_ratio))
            if new_ratio < self.MaximumRatio:
                startcff = midcff
            else:
                stopcff = midcff
        return midcff

    def getLineDensity(self, axis):
        return self.read_traj_parameter("LineDensity", axis=axis)

    def setLineDensity(self, axis, value):
        # print "read linedensity"
        self.write_traj_parameter("LineDensity", value, axis=axis, keep_traj_pos=False)

    def getCff(self, axis):
        return self.read_traj_parameter("Cff", axis=axis)

    def setCff(self, axis, value):
        if axis == 1:
            state, status, switches = self.StateOne(axis)
            if state == State.On:
                cff_max = self.calc_cff_max(self.active_parameters_dict)
                if value < cff_max:
                    old_cff = self.getCff(axis)
                    self.next_parameters_dict["Cff"] = value
                    try:
                        energy_min, energy_max = self.calc_limits(
                            self.next_parameters_dict
                        )
                        if self.ReadOne(axis) <= energy_min:
                            self.next_parameters_dict["Cff"] = old_cff
                            raise Exception(
                                f"It is currently not allowed to write Cff with the current energy. Move the energy above: {energy_min}"
                            )
                        # move to another trajectory!
                        self.move_to_new_trajectory(
                            self.next_parameters_dict, keep_traj_pos=True
                        )
                    except Exception as e:
                        raise Exception(
                            "Fail to change cff while moving to new trajectory, "
                            + "check the energy you are and what will be the min and max grating"
                            + "and mirror angles.\n{}".format(str(e))
                        )
                else:
                    raise ValueError("Too large Cff, current max is {}".format(cff_max))
            elif state == State.Alarm:
                # update parameter for DefinePosition
                self.active_parameters_dict["Cff"] = value
            else:
                raise Exception(
                    "It is currently not allowed to write Cff, because the state is different than ON or ALARM"
                )

    def getDiffrOrder(self, axis):
        return self.read_traj_parameter("DiffrOrder", axis=axis)

    def setDiffrOrder(self, axis, value):
        self.write_traj_parameter("DiffrOrder", value, axis=axis, keep_traj_pos=False)

    def getMaxGratingAngle(self, axis):
        return self.read_traj_parameter("MaxGratingAngle", axis=axis)

    def setMaxGratingAngle(self, axis, value):
        self.write_traj_parameter(
            "MaxGratingAngle", value, axis=axis, keep_traj_pos=True
        )

    def getOffsetGr(self, axis):
        return self.read_traj_parameter("OffsetGr", axis=axis)

    def setOffsetGr(self, axis, value):
        self.write_traj_parameter("OffsetGr", value, axis=axis, keep_traj_pos=True)

    def getOffsetM(self, axis):
        return self.read_traj_parameter("OffsetM", axis=axis)

    def setOffsetM(self, axis, value):
        self.write_traj_parameter("OffsetM", value, axis=1, keep_traj_pos=True)

    def getCffReal(self, axis):
        # print "write OffsetM"
        if axis == 1:
            if time.time() - self._prev_position_time > 1:
                with self.lock:
                    m_steps, g_steps = self._ipap.get_pos(list(self.motor_roles))
                    self._grating_pos = self._steps2units("grating", g_steps)
                    self._mirror_pos = self._steps2units("mirror", m_steps)
                    self._log.trace(
                        "Grating: {}, mirror: {}".format(
                            self._grating_pos, self._mirror_pos
                        )
                    )
                self._prev_position_time = time.time()
                self._energy_raw, self._cff = self.calc_pseudo_raw(
                    self.active_parameters_dict
                )
            return self._cff

    def getEnergyReal(self, axis):
        # print "write OffsetM"
        if axis == 1:
            if time.time() - self._prev_position_time > 1:
                with self.lock:
                    m_steps, g_steps = self._ipap.get_pos(list(self.motor_roles))
                    self._grating_pos = self._steps2units("grating", g_steps)
                    self._mirror_pos = self._steps2units("mirror", m_steps)
                self._prev_position_time = time.time()
                self._energy_raw, self._cff = self.calc_pseudo_raw(
                    self.active_parameters_dict
                )
            self._energy = self.calc_pseudo(self._energy_raw)
            return self._energy

    def getEnergyRaw(self, axis):
        if axis == 1:
            if time.time() - self._prev_position_time > 1:
                with self.lock:
                    m_steps, g_steps = self._ipap.get_pos(list(self.motor_roles))
                    self._grating_pos = self._steps2units("grating", g_steps)
                    self._mirror_pos = self._steps2units("mirror", m_steps)
                self._prev_position_time = time.time()
                self._energy_raw, self._cff = self.calc_pseudo_raw(
                    self.active_parameters_dict
                )
            return self._energy_raw

    def getBeamlineBranch(self, axis):
        return self.beamline_branch

    def setBeamlineBranch(self, axis, value):
        if value.lower() in {"a", "b"}:
            state, status, switches = self.StateOne(axis)
            if state == State.On:
                self.beamline_branch = value.lower()
                self.move_to_new_trajectory(
                    self.active_parameters_dict, keep_traj_pos=True
                )
            else:
                raise ValueError(
                    f"Only allowed to switch branches in ON state.\n"
                    f"State {state}\nStatus {status}\nSwitches {switches}\n"
                )
        else:
            raise ValueError(f"Invalid branch, only A and B allowed (got {value!r})")

    def getGrating(self, axis):
        return self.grating

    def setGrating(self, axis, value):
        if value.lower() in {"pg400", "pg1221"}:
            state, status, switches = self.StateOne(axis)
            if state == State.On:
                self.grating = value.lower()
                self.move_to_new_trajectory(
                    self.active_parameters_dict, keep_traj_pos=True
                )
            else:
                raise ValueError(
                    f"Only allowed to switch grating in ON state.\n"
                    f"State {state}\nStatus {status}\nSwitches {switches}\n"
                )
        else:
            raise ValueError(
                f"Invalid grating, only pg400 and pg1221 allowed (got {value!r})"
            )

    def get_mono_terms(self):
        """
        Select mono terms according to `BeamlineBranch` and `Grating` attributes
        """
        mono_terms_map = {
            "a-pg400": self.mono_equations_a_branch_pg400,
            "b-pg400": self.mono_equations_b_branch_pg400,
            "a-pg1221": self.mono_equations_a_branch_pg1221,
            "b-pg1221": self.mono_equations_b_branch_pg1221,
        }
        branch = self.beamline_branch.lower()
        grating = self.grating.lower()
        try:
            mono_terms = mono_terms_map[f"{branch}-{grating}"]
        except KeyError:
            raise ValueError(
                f"Cannot get terms describing the mono energy equations for "
                f"current Branch: {self.beamline_branch} and Grating: {self.grating}"
            )
        return mono_terms

    @property
    def beamline_branch(self):
        if self._beamline_branch is None:
            try:
                self._beamline_branch = self._get_memorized_attribute("BeamlineBranch")
                self._log.debug(
                    f"using memorized value for BeamlineBranch: {self._beamline_branch!r}"
                )
            except Exception as e:
                self._beamline_branch = DEFAULT_BEAMLINE_BRANCH
                self._log.warning(
                    f"Could not get memorized value for BeamlineBranch attribute: {e}. "
                    f"Defaulted to {self._beamline_branch!r}."
                )
        return self._beamline_branch

    @beamline_branch.setter
    def beamline_branch(self, value):
        self._beamline_branch = value

    @property
    def grating(self):
        if self._grating is None:
            try:
                self._grating = self._get_memorized_attribute("Grating")
                self._log.debug(f"using memorized value for Grating: {self._grating!r}")
            except Exception as e:
                self._grating = DEFAULT_GRATING
                self._log.warning(
                    f"Could not get memorized value for Grating attribute: {e}. "
                    f"Defaulted to {self._grating!r}."
                )
        return self._grating

    @grating.setter
    def grating(self, value):
        self._grating = value
