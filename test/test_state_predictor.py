from model_based import State_predictor
import torch

import matplotlib.pyplot as plt

def test_sample():

    memory_path = "result/191119_214214/memories.pkl"

    sp = State_predictor(4, 4, memory_path)
    states, actions, state_s = sp._sample_batch()
    print(states.shape, actions.shape, state_s.shape)

    fig, axes = plt.subplots(1, 5)

    axes[0].imshow(states[0, :, :, -1], interpolation="nearest")
    axes[0].set_title("moto")
    for i in range(4):
        axes[1 + i].imshow(state_s[0, i, :, :], interpolation="nearest")
        axes[i + 1].set_title(str(i + 1))
    plt.show()

def test_predict(memory_path="result/191119_214214/memories.pkl", model_path="result_WORLD/191122_175903/models/model.pt"):
    sp = State_predictor(4, memory_path=memory_path, model_path=model_path)

    states, actions, state_s = sp._sample_batch(is_test=True)
    print(actions.size(), state_s.size())
    output = sp.predict(states, actions[:, 0, :])

    state_s = state_s.data.cpu().numpy()
    states = states.data.cpu().numpy()

    for b in range(32):
        fig, axes = plt.subplots(1, 5)

        for i in range(3):
            axes[i].imshow(states[b, i + 1, :, :], interpolation="nearest")
            axes[i].set_title(str(i))

        axes[3].imshow(output[b, 0, :, :], interpolation="nearest")
        axes[3].set_title("3 (predicted)")
        axes[4].imshow(state_s[b, 0, :, :], interpolation="nearest")
        axes[4].set_title("3 (real)")

        plt.show()


def test_predict_multi(memory_path="result/191119_214214/memories.pkl", model_path="result_WORLD/191122_175903/models/model.pt", prediction_step=3):
    sp = State_predictor(4, memory_path=memory_path, model_path=model_path)

    states, actions, state_s = sp._sample_batch(
        prediction_step=prediction_step, is_test=True)

    input_states = states
    outputs = []
    for step in range(prediction_step):
        output = sp.model(input_states, actions[:, step, :])
        outputs.append(output)
        input_states = torch.cat((input_states[:, :3, :, :], output), dim=1)

    outputs = torch.cat(outputs, dim=1)

    state_s = sp._post_process_states(state_s)
    outputs = sp._post_process_states(outputs)
    for i in range(32):
        fig, axes = plt.subplots(2, prediction_step)
        for step in range(prediction_step):
            axes[0, step].imshow(outputs[i, step, :, :],
                                 interpolation='nearest')
            axes[0, step].set_title('predict_{}'.format(str(step)))
            axes[1, step].imshow(state_s[i, step, :, :],
                                 interpolation='nearest')
            axes[1, step].set_title('real_{}'.format(str(step)))
        plt.show()

def test_predict_multi_with_models(model_path_1,model_path_2,memory_path="result/191119_214214/memories.pkl",prediction_step=3):
    sp_1 = State_predictor(4, memory_path=memory_path, model_path=model_path_1)
    sp_2 = State_predictor(4, memory_path=memory_path, model_path=model_path_2)

    states, actions, state_s = sp_1._sample_batch(
        prediction_step=prediction_step, is_test=True)

    input_states_1,input_states_2 = states,states
    outputs_1 = []
    outputs_2 = []
    for step in range(prediction_step):
        output_1 = sp_1.model(input_states_1, actions[:, step, :])
        output_2 = sp_2.model(input_states_2, actions[:, step, :])
        outputs_1.append(output_1)
        outputs_2.append(output_2)
        input_states_1 = torch.cat((input_states_1[:, :3, :, :], output_1), dim=1)
        input_states_2 = torch.cat((input_states_2[:, :3, :, :], output_2), dim=1)

    outputs_1 = torch.cat(outputs_1, dim=1)
    outputs_2 = torch.cat(outputs_2, dim=1)

    state_s = sp_1._post_process_states(state_s)
    outputs_1 = sp_1._post_process_states(outputs_1)
    outputs_2 = sp_1._post_process_states(outputs_2)
    for i in range(32):
        fig, axes = plt.subplots(3, prediction_step)
        for step in range(prediction_step):
            axes[0, step].imshow(outputs_1[i, step, :, :],
                                 interpolation='nearest')
            axes[0, step].set_title('predict_{}_model({})'.format(str(step),'1'))
            axes[1, step].imshow(outputs_2[i, step, :, :],
                                 interpolation='nearest')
            axes[1, step].set_title('predict_{}_model({})'.format(str(step),'2'))
            axes[2, step].imshow(state_s[i, step, :, :],
                                 interpolation='nearest')
            axes[2, step].set_title('real_{}'.format(str(step)))
        plt.show()