from .policy_RNN import ModelFreeOffPolicy_Separate_RNN as Policy_Separate_RNN

AGENT_CLASSES = {
    "Policy_Separate_RNN": Policy_Separate_RNN,
}



from enum import Enum


class AGENT_ARCHS(str, Enum):
    # inherit from str to allow comparison with str
    Memory = Policy_Separate_RNN.ARCH