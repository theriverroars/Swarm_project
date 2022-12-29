from gym.envs.registration import register

register(
    id='ctrl-aviary-v0',
    entry_point='gym_cbf.envs:CtrlAviary',
)

register(
    id='dyn-aviary-v0',
    entry_point='gym_cbf.envs:DynAviary',
)

register(
    id='velocity-aviary-v0',
    entry_point='gym_cbf.envs:VelocityAviary',
)

register(
    id='vision-aviary-v0',
    entry_point='gym_cbf.envs:VisionAviary',
)




register(
    id='takeoff-aviary-v0',
    entry_point='gym_cbf.envs.single_agent_rl:TakeoffAviary',
)

register(
    id='hover-aviary-v0',
    entry_point='gym_cbf.envs.single_agent_rl:HoverAviary',
)

register(
    id='flythrugate-aviary-v0',
    entry_point='gym_cbf.envs.single_agent_rl:FlyThruGateAviary',
)

register(
    id='tune-aviary-v0',
    entry_point='gym_cbf.envs.single_agent_rl:TuneAviary',
)




register(
    id='flock-aviary-v0',
    entry_point='gym_cbf.envs.multi_agent_rl:FlockAviary',
)

register(
    id='leaderfollower-aviary-v0',
    entry_point='gym_cbf.envs.multi_agent_rl:LeaderFollowerAviary',
)

register(
    id='meetup-aviary-v0',
    entry_point='gym_cbf.envs.multi_agent_rl:MeetupAviary',
)