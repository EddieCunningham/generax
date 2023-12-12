# Base

::: generax.flows.base.BijectiveTransform
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse
            - get_inverse

::: generax.flows.base.TimeDependentBijectiveTransform
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse
            - vector_field

::: generax.flows.base.Sequential
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse

::: generax.flows.base.InjectiveTransform
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse
            - project
            - log_determinant
            - log_determinant_surrogate

::: generax.flows.base.InjectiveSequential
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse
