import os
from xml.etree.ElementTree import iselement
import numpy as np
is_cv2 = False
try:
    import cv2    
    is_cv2 = True
except:
    print('You do not have opencv-python')


class Reversi:

    def __init__(self):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.Blank = 0
        self.Black = 1
        self.White = 2
        self.screen_n_rows = 8
        self.screen_n_cols = 8
        self.enable_actions = np.arange(self.screen_n_rows*self.screen_n_cols)
        # variables
        self.reset()

    def update(self, action, color):
        """
        action:石を置く位置 0〜63
        """

        self.terminal = False
        self.reward = 0

        # そのマスにおいた場合の取れる数
        n = self.put_piece(action, color, False)
        if n < 1:
            # 置けない場所に置こうとした
            self.reward = -1
            self.next_color = color # 次も同じ人の手番
            return

        # 1枚以上捲れる
        # 実際に捲る screenが変わる
        self.put_piece(action, color)

        # 次の人が捲れるかを判定する
        e1 = self.get_enables(self.Black if color == self.White else self.White)

        if len(e1) > 0:  # 次の人はおける
            # 次の人に手番を渡す
            self.next_color = self.Black if color == self.White else self.White  
            return

        # 次の人はおけない時
        # 今置いた人はおける
        if len(self.get_enables(color)) > 0:  
            self.next_color = color 
            return        
        # 両者おけない
        else:
            self.next_color = 0
            self.terminal = True
            self.reward = 0  # dummy
            return

    def get_next_player(self):
        return self.next_color


    def draw(self):
        SIZE = 50  # px
        # 緑色の画像を作成
        img = np.zeros((SIZE * self.screen_n_rows + 1, SIZE * self.screen_n_rows + 1, 3), np.uint8)
        img[:, :,] = (0, 255, 0)
        # 枠線を引く
        for y in range(self.screen_n_rows + 1):
            img[y * SIZE: y * SIZE + 1, :, ] = (0, 0, 0)
        for x in range(self.screen_n_cols + 1):
            img[:, x * SIZE: x * SIZE + 1, ] = (0, 0, 0)
        i = 0
        for y in range(self.screen_n_rows):
            for x in range(self.screen_n_cols):
                cx = x * SIZE + (int)(SIZE / 2)
                cy = y * SIZE + (int)(SIZE / 2)
                if self.screen[y][x] == self.Black:
                    cv2.circle(img, (cx, cy), (int)(SIZE / 2 - 3), (0, 0, 0), -1)
                elif self.screen[y][x] == self.White:
                    cv2.circle(img, (cx, cy), (int)(SIZE / 2 - 3), (255, 255, 255), -1)
        cv2.imshow("board", img)
        cv2.waitKey(1)

    """ screen_t+1, reward_t, terminal_tを返す """
    def observe(self, color):
        self.draw()
        
        # 試合が終わっている場合はcolorに応じて報酬を出す
        if self.isEnd():
            if self.winner() == color:
                self.reward = 1
            elif self.winner() == 0:
                self.reward = 0
            else:
                self.reward = -1

        return self.screen, self.reward, self.terminal
        
    def execute_action(self, action, color):
        self.update(action, color)


    """ 盤面の初期化 """
    def reset(self):
        # reset board
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))
        self.set_cells(27, self.White)
        self.set_cells(28, self.Black)
        self.set_cells(35, self.Black)
        self.set_cells(36, self.White)
        self.next_color = self.Black
        self.reward = 0
        self.terminal = False

    def get_cells(self, i):
        r = int(i / self.screen_n_cols)
        c = int(i - ( r * self.screen_n_cols))
        return self.screen[r][c]  
               
    def set_cells(self, i, value):
        r = int(i / self.screen_n_cols)
        c = int(i - ( r * self.screen_n_cols))
        self.screen[r][c] = value 

    """ 盤面の出力 """
    def print_screen(self):
        if is_cv2:
            self.show_screen()    
            return

        i = 0
        for r in range(self.screen_n_rows):
            s1 = ''
            for c in range(self.screen_n_cols):
                s2 = ''
                if self.screen[r][c] == self.Blank:
                    s2 = '{0:2d}'.format(self.enable_actions[i])
                elif self.screen[r][c] == self.Black:
                    s2 = '●'
                elif self.screen[r][c] == self.White:
                    s2 = '○'
                s1 = s1 + ' ' + s2
                i += 1
            print(s1)


    def put_piece(self, action, color, puton=True):
        """自駒color(1 or 2)を位置action(0～63)に置く関数 """
         
        if self.get_cells(action) != self.Blank:
            return -1

        """ ---------------------------------------------------------
           縦横斜めの8通りは、1次元データなので、
           現在位置から[-9, -8, -7, -1, 1, 7, 8, 9] 
           ずれた方向を見ます。
           これは、[-1, 0, 1]と[-8, 0, 8]の組合せで調べます
           (0と0のペアは除く)。
        """
        t, x, y, l = 0, action%8, action//8, []
        for di, fi in zip([-1, 0, 1], [x, 7, 7-x]):
            for dj, fj in zip([-8, 0, 8], [y, 7, 7-y]):
                
                if not di == dj == 0:
                    b, j, k, m, n =[], 0, 0, [], 0                    
                    """a:対象位置のid リスト"""
                    a = self.enable_actions[action+di+dj::di+dj][:min(fi, fj)]
                    """b:対象位置の駒id リスト"""
                    for i in a: 
                        b.append(self.get_cells(i))
                    
                    #print("a={:}".format(a))
                    #print("b={:}".format(b))
                    for i in b:
                        if i == 0: #空白
                            break  
                        elif i == color: #自駒があればその間の相手の駒を取れる
                            """ 取れる数を確定する """ 
                            n = k
                            """ ひっくり返す駒を確定する """ 
                            l += m
                            """ その方向の探査終了 """
                            break
                        else: #相手の駒
                            k += 1
                            """ ひっくり返す位置をストックする """ 
                            m.insert(0, a[j]) 
                        j += 1
                    #print("n={:}".format(n))    
                    t += n 
                    
        #print("t={:}".format(t))            
        #print("l={:}".format(l))            
        if t == 0:
            return 0
            
        if puton:
            """ ひっくり返す石を登録する """
            for i in l:
                self.set_cells(i, color)
            """ 今置いた石を追加する """ 
            self.set_cells(action, color)
            
            
        return t
        
    def winner(self):
        """ 勝ったほうを返す """
        Black_score = self.get_score(self.Black)
        White_score = self.get_score(self.White)
            
        if Black_score == White_score:
            return 0 # 引き分け
        elif Black_score > White_score:
            return self.Black # Blackの勝ち
        elif Black_score < White_score:
            return self.White # Whiteの勝ち
        
    def get_score(self, color):
        """ 指定した色の現在のスコアを返す """
        score = 0
        for i in self.enable_actions:
            if self.get_cells(i) == color:
                score += 1
        return score

    def get_enables(self, color):
        result = []
        """ 置ける位置のリストを返す """
        for action in self.enable_actions:
            if self.get_cells(action) == self.Blank:
                """ 空白の位置 """
                if self.put_piece(action, color, False) > 0:
                    """ ここ置ける!! """
                    result.insert(0, action)
        return result
                    
    def isEnd(self):
        e1 = self.get_enables(self.Black)        
        e2 = self.get_enables(self.White)  
        if len(e1) == 0 and len(e2) == 0:
            #双方置けなくなったらゲーム終了
            return True

        return False
            
        # for action in self.enable_actions:
        #     if self.get_cells(action) == self.Blank:
        #         return False

        # return True


 
if __name__ == "__main__":
   # game
   print("まだ実装していません")
