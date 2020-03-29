from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from Models.structurer.LstmStructurer import LstmStructurer



################################################################## Beginin of MLP Part ##########################################################################################

def create_lstm(lstm_struct: LstmStructurer) -> Model:

    input_tensor = Input((32, 96))

    lstm_tensor = input_tensor
    for i in range(lstm_struct.nb_layers - 1):
        lstm_tensor = LSTM(units= lstm_struct.units,
                           kernel_regularizer=lstm_struct.kernel_regularizer,
                           recurrent_regularizer=lstm_struct.recurrent_regularizer,
                           dropout=lstm_struct.dropout_value,
                           recurrent_dropout=lstm_struct.recurrent_dropout_value,
                           return_sequences=True
                           )(lstm_tensor)

    lstm_tensor = LSTM(units=lstm_struct.units,
                       kernel_regularizer=lstm_struct.kernel_regularizer,
                       recurrent_regularizer=lstm_struct.recurrent_regularizer,
                       dropout=lstm_struct.dropout_value,
                       recurrent_dropout=lstm_struct.recurrent_dropout_value,
                       return_sequences=False
                       )(lstm_tensor)

    output_tensor = Dense(10, activation=lstm_struct.output_activation, kernel_regularizer=lstm_struct.output_regularizer)(lstm_tensor)

    model = Model(input_tensor, output_tensor)

    model.compile(loss=lstm_struct.loss, optimizer=lstm_struct.optimizer, metrics=lstm_struct.metrics)

    return model

def getLstmStructAsString(lstm_structurer: LstmStructurer) -> str:
    return "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(lstm_structurer.nb_layers,
                                                                       lstm_structurer.units,
                                                                       lstm_structurer.activation,
                                                                       lstm_structurer.recurrent_activation,
                                                                       lstm_structurer.output_activation,
                                                                       lstm_structurer.dropout_value,
                                                                       lstm_structurer.recurrent_dropout_value,
                                                                       lstm_structurer.kernel_regularizer.__class__.__name__,
                                                                       lstm_structurer.recurrent_regularizer.__class__.__name__,
                                                                       lstm_structurer.output_regularizer.__class__.__name__,
                                                                       lstm_structurer.l1_value,
                                                                       lstm_structurer.l2_value,
                                                                       lstm_structurer.loss,
                                                                       lstm_structurer.optimizer,
                                                                       " ".join(lstm_structurer.metrics))
