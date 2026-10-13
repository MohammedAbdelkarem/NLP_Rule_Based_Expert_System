from engine import NeuralNetworkExpertSystem
from utils.console_control import install_console_controls, run_controlled_engine


engine = NeuralNetworkExpertSystem()
engine.reset()
install_console_controls()
run_controlled_engine(engine)
