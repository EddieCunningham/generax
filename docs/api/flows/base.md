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

::: generax.flows.base.Repeat
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse
            - to_sequential

::: generax.flows.base.Sequential
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse

::: generax.flows.base.TimeDependentSequential
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse
            - vector_field

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

::: generax.flows.coupling.TimeDependentWrapper
    selection:
        members:
            - __init__
            - __call__
            - data_dependent_init
            - inverse
            - vector_field
