from ModelUtils.Models.LinearModel import *
from ModelUtils.Models.structurer.LinearStructurer import *
from ModelUtils.Tester.ModelTester import train_model
from ModelUtils.Models.structurer.ModelName import *

if __name__ == '__main__':
    linear_model_struct = LinearStructurer()
    linear_model_struct.nb_classes = 572
    linear_model_struct.input_shape = (192, 108, 3)

    model = create_linear_model(linear_model_struct)
    description = getLinStructAsString(linear_model_struct)
    train_model(LINEAR_MODEL, model, description, 2)
