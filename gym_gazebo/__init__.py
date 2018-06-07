import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------


# barret wam

register(
    id='GazeboWAMemptyEnv-v0',
    entry_point='gym_gazebo.envs.barret_wam:GazeboWAMemptyEnv',
)

register(
    id='GazeboWAMemptyEnv-v1',
    entry_point='gym_gazebo.envs.barret_wam:GazeboWAMemptyEnvv1',
)

register(
    id='GazeboWAMemptyEnv-v2',
    entry_point='gym_gazebo.envs.barret_wam:GazeboWAMemptyEnvv2',
)
