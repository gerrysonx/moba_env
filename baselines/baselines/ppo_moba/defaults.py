def gym_moba():
    return dict(
        nsteps=2048*8, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=0.01,
        vf_coef=1.0,
        lr=lambda f : f * 3e-3,
        cliprange=0.2,
    )