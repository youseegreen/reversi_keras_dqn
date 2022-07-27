import numpy as np

import argparse
from reversi import Reversi
from dqn_agent import DQNAgent
from random_agent import RANDOMAgent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", help='Path ot the model files')
    parser.add_argument("-l", "--load", dest="load", action="store_true",
                        default=False, help='Load trained model (default: off)')
    parser.add_argument("-e", "--epoch-num", dest="n_epochs", default=30,
                        type=int, help='Numpber of training epochs (default: 1000)')
    parser.add_argument("--simple", dest="is_simple", action="store_true", default=True,
                        help='Train simple model without cnn (8 x 8) (default: true)')
    parser.add_argument("-g", "--graves", dest="graves", action="store_true",
                        default=False, help='Use RmpropGraves (default: off)')
    parser.add_argument("-d", "--ddqn", dest="ddqn", action="store_true",
                        default=False, help='Use Double DQN (default: off)')
    parser.add_argument("-s", "--save-interval", dest="save_interval", default=15, type=int)  # 1000
    args = parser.parse_args()

    # parameters
    n_epochs = args.n_epochs

    # environment, agent
    env = Reversi()

    # playerID
    playerID = [env.Black, env.White, env.Black]

    # player agent
    players = []
    # player[0] = env.Black
    agent = DQNAgent(env.enable_actions, env.name, color="black", ddqn=args.ddqn)
    if args.load:
        agent.load_model(args.model_path)
    else:
        agent.init_model()
    players.append(agent)

    # player[1] = env.White
    agent = RANDOMAgent(env.enable_actions, env.name, color="white")
    players.append(agent)

    # # variables
    wins = [0, 0]
    e = 0          # エポック数
    
    while e < n_epochs:
        # reset
        env.reset()

        state_ts = [None, None]
        action_ts = [None, None]
        state_t_1s = [None, None]
        reward_ts = [None, None]
        terminals = [None, None]
        
        while not env.isEnd():

            # 次の手番の人をゲットする
            # ゲームが終わっていない以上、BlackかWhiteしか出ない
            next_player_color = env.get_next_player()
            idx = 0 if next_player_color == env.Black else 1 if next_player_color == env.White else None   # Noneだとプログラムおかしい

            # observe environment
            state_t_1s[idx], reward_ts[idx], terminals[idx] = env.observe(next_player_color)

            # 1時刻前の結果をstoreにpushしておく
            # 本来はenv.execute_action(action_ts[col], playerID[col])の後に、env.observe()とstore_experienceを実行したいが、
            # 相手番との兼ね合いがあるので、また自分の手番になった際に代入するようにしている
            if action_ts[idx] != None:
                players[idx].store_experience([state_ts[idx]], action_ts[idx], reward_ts[idx], [state_t_1s[idx]], terminals[idx])

            # 着手する盤面をstate_tに代入
            state_ts[idx] = state_t_1s[idx]  
            # 行動を選択
            action_ts[idx] = players[idx].select_action([state_ts[idx]], players[idx].exploration)
            # 実際に環境に対してアクションを起こす
            env.execute_action(action_ts[idx], playerID[idx])

        # whileを抜けたということはゲームが終わったということ
        # 最後のstore_experienceを代入する
        for i, color in enumerate([env.Black, env.White]):
            state_t_1s[i], reward_ts[i], terminals[i] = env.observe(color)
            players[i].store_experience([state_ts[i]], action_ts[i], reward_ts[i], [state_t_1s[i]], terminals[i])

            if reward_ts[i] == 1:
                wins[i] += 1
        
        # 1試合終わったら学習を開始する ⇒ N試合終わったら学習を開始するに変更する

        # εの値を小さくする
        players[0].update_exploration(e)
        players[1].update_exploration(e)

        # 学習
        players[0].experience_replay(e)
        players[1].experience_replay(e)

        # 10試合でtarget_modelもupdateする
        if e % 10:
            players[0].update_target_model
            players[1].update_target_model
            players[0].reset_experience()
            players[1].reset_experience()

        print(f"EPOCH: {e:03d}/{n_epochs - 1:03d} | BLACK_WIN: {wins[0]:03d} | WHITE_WIN: {wins[1]:03d}")
        if e > 0 and e % args.save_interval == 0:
            players[0].save_model(e)
            players[0].save_model()
            players[1].save_model(e)
            players[1].save_model()
        e += 1

    # save model
    players[0].save_model()
    players[1].save_model()
