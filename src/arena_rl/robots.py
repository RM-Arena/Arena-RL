"""Public robot descriptors: one learned contract conditioned on the deployed chassis.

Everything here comes from the public design catalog that Arena sends with each
observation (`robot_models`) and from the controlled robot's own telemetry. No
hidden simulator state, physics edit or per-model reward is introduced.
"""
from dataclasses import dataclass
import math
import numpy as np

ROLES=('infantry','heavy','sentry')
DRIVES=('wheel_legged','mecanum','omni','swerve')
HOLONOMIC=('mecanum','omni','swerve')
# 2027-legal ground models, grouped by the chassis class a learned policy must handle.
GROUND_MODELS_2027={'infantry':(3,4,7,301,302,303,304,305,2731,2732),'heavy':(2711,2712,2713),'sentry':(2771,2772)}
AERIAL_MODELS_2027=(2761,2762,2763)
DESCRIPTOR=(['role_'+r for r in ROLES]+['drive_'+d for d in DRIVES]+
            ['holonomic','caliber_42','can_jump','self_righting','tunnel','step_climb','speed_reference','mass','radius','clearance','turret_rate','legs'])


@dataclass(frozen=True)
class RobotProfile:
    model_id:int
    role:str
    drive:str
    caliber:str
    jump:bool
    self_righting:bool
    tunnel:bool
    step_climb_m:float
    speed_reference_mps:float
    mass_kg:float
    radius_m:float
    clearance_m:float
    turret_rate:float
    leg_min_m:float|None
    leg_max_m:float|None
    serial_legs:bool

    @property
    def legged(self):return self.drive=='wheel_legged'
    @property
    def holonomic(self):return self.drive in HOLONOMIC
    @property
    def heavy_launcher(self):return self.role=='heavy'
    @property
    def ammo_scale(self):return 40. if self.caliber=='42mm' else 300.

    def descriptor(self):
        values=[float(self.role==r) for r in ROLES]+[float(self.drive==d) for d in DRIVES]
        values+=[float(self.holonomic),float(self.caliber=='42mm'),float(self.jump),float(self.self_righting),float(self.tunnel),
                 self.step_climb_m/.2,self.speed_reference_mps/5.,self.mass_kg/40.,self.radius_m/.5,self.clearance_m/.12,
                 self.turret_rate/8.,float(self.legged)]
        return np.asarray(values,dtype=np.float32)

    def leg_height(self,level):
        """Low/nominal/high ride height inside this chassis' public leg travel."""
        low,high=self.leg_min_m or .12,self.leg_max_m or .34
        nominal=min(max(.245,low+.04),high-.02)
        return [max(low+.06,.18),nominal,min(high-.03,.31)][int(level)]

    def json(self):return {k:getattr(self,k) for k in self.__dataclass_fields__}|{'legged':self.legged,'holonomic':self.holonomic}


def profile(robot,observation):
    """Describe the robot from public design data; telemetry fills fields absent from the catalog."""
    models=observation.get('robot_models') or {}
    design=models.get(str(robot['model_slot']),{})
    caps=design.get('capabilities',{})
    drive=caps.get('drive',design.get('chassis_type',robot.get('chassis_type','mecanum')))
    if drive not in DRIVES:raise ValueError(f'Ground policy cannot drive chassis {drive}')
    role=robot['role']
    if role not in ROLES:raise ValueError(f'Ground policy has no role {role}')
    size=robot.get('chassis_size_m',design.get('chassis_size_m',[.5,.5,.1]))
    return RobotProfile(int(robot['model_slot']),role,drive,'42mm' if role=='heavy' else '17mm',bool(caps.get('jump',drive=='wheel_legged')),
        bool(caps.get('self_righting',False)),bool(caps.get('tunnel',False)),float(caps.get('step_climb_m',0.) or 0.),
        float(robot.get('speed_reference_mps',design.get('speed_reference_mps',4.))),float(robot.get('mass_kg',design.get('mass_kg',25.))),
        float(robot.get('radius',math.hypot(size[0],size[1])/2)),float(design.get('ground_clearance_m',.05)),
        float(design.get('turret_yaw_rate_rad_s',6.)),design.get('leg_min_m'),design.get('leg_max_m'),design.get('leg_type')=='serial')


def can_fire(robot,profile):
    """Referee fire gates visible to the robot itself; the engine still enforces every rule."""
    if not robot['alive'] or robot.get('weak') or robot.get('overheated') or robot.get('heat_permanent_lock'):return False
    if profile.role=='heavy' and robot.get('launcher') not in ('fitted','installed'):return False
    heat_per_shot=100. if profile.caliber=='42mm' else 10.
    return robot.get('max_heat',0)<=0 and profile.role!='heavy' or robot.get('heat',0)+heat_per_shot<=robot.get('max_heat',0)+1e-6


def roster_profiles(observation):
    return {r['id']:profile(r,observation) for r in observation['robots'] if r['role'] in ROLES}
