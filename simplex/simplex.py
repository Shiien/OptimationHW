# -*- coding: utf-8 -*-
import numpy as np
import argparse
import ast
import copy
import sys
from typing import *


class Simplex:
    def __init__(self, args, path: Optional[str] = None, method: str = 'F'):
        self.method = method
        self.myeps = 1e-6
        if args is None:
            return
        b = self.args_to_array(args.b)
        c = self.args_to_array(args.c)
        if args.max or args.min:
            self.opp = args.max
            self.max = args.max
        else:
            assert 0, 'you should set --max or --min'
        Al = self.args_to_array(args.Al)
        Ag = self.args_to_array(args.Ag)
        Ae = self.args_to_array(args.Ae)
        if path is not None:
            f = open(path, 'w+', encoding='utf-8')
        else:
            f = sys.stdout
        self.outfile = f
        s = ['\\leq' for _ in Al if Al.size > 0]
        s.extend(['\\geq' for _ in Ag if Ag.size > 0])
        s.extend(['=' for _ in Ae if Ae.size > 0])
        print(u'原问题可以表示为', file=self.outfile)
        self.show_problem(np.concatenate([k for k in (Al, Ag, Ae) if k.size != 0], axis=0), b, c, s, f)
        x_num = 0
        row_num = 0
        if Al.size != 0:
            x_num = max(x_num, Al.shape[1])
            row_num += Al.shape[0]
        if Ag.size != 0:
            x_num = max(x_num, Ag.shape[1])
            row_num += Ag.shape[0]
        if Ae.size != 0:
            x_num = max(x_num, Ae.shape[1])
        A = copy.deepcopy(np.zeros([b.size, c.size + row_num], dtype=np.float))
        try:
            last_col = 0
            if Al.size != 0:
                A[0:Al.shape[0], 0:Al.shape[1]] = Al
                A[0:Al.shape[0], Al.shape[1]:(Al.shape[1] + Al.shape[0])] = np.eye(Al.shape[0])
                x_num = x_num + Al.shape[0]
                last_col = Al.shape[0]
            if Ag.size != 0:
                A[last_col:last_col + Ag.shape[0], 0:Ag.shape[1]] = Ag
                A[last_col:last_col + Ag.shape[0], x_num:x_num + Ag.shape[0]] = np.eye(Ag.shape[0]) * -1
                last_col = last_col + Ag.shape[0]
            if Ae.size != 0:
                A[last_col:last_col + Ae.shape[0], 0:Ae.shape[1]] = Ae
            self.A = A
            self.b = b
            for i in range(len(self.b)):
                if self.b[i] < 0:
                    self.b[i] = -self.b[i]
                    self.A[i] = -self.A[i]
            c = np.append(c, np.zeros(row_num))
            if args.max:
                self.c = -c
            else:
                self.c = c
            self.max = False
            self.M = self.A.shape[0]
            self.N = self.c.shape[0]
            print(u'化为标准型', file=self.outfile)
            self.show_problem(self.A, self.b, self.c, ['=' for _ in self.b], f)
        except:
            print('formation error')
            # assert 0

    def calculate(self):
        if self.method == 'T':
            self._two_cal()
        elif self.method == 'F':
            self._force_cal()
        elif self.method == 'M':
            self._M_cal()
        else:
            assert 0, 'error'
        pass

    def _is_greater_than_0(self, v):
        if isinstance(v, tuple):
            return v[0] > self.myeps and v[1] > self.myeps
        return v > self.myeps

    def _cal(self):
        times = 0
        while True:
            times += 1
            self.show_table()
            pl, v = self.chooseL()
            if self._is_greater_than_0(v) or pl in self.baseX:
                print('结束')
                if self.method=='M':
                    for idx, i in enumerate(self.baseX):
                        if self.cm[i-1] >0:
                            print(f'无解', file=self.outfile)
                            return None,None,None
                ans = 0.0
                for idx, i in enumerate(self.baseX):
                    ans += self.c[i - 1] * self.Binvb[idx]
                if self.opp:
                    ans = -ans
                print(f'答案是{ans}', file=self.outfile)
                return ans, self.baseX, self.Binvb

            print(u'选择非基变量' + f'$x_{{{pl}}}$', file=self.outfile)
            pr = self.chooseR(pl)
            if pr == -1:
                print('无界', file=self.outfile)
                return None, None, None
            if times >= 10000000:
                print('超时')
                return None, None, None
            print(u'选择基变量' + f'$x_{{{self.baseX[pr]}}}$', file=self.outfile)
            self.pivot(pl, pr)

    def pivot(self, l, r):
        for i in range(self.M):
            if i == r:
                continue
            kk = self.BinvA[i][l - 1] / self.BinvA[r][l - 1]
            self.BinvA[i] = self.BinvA[i] - kk * self.BinvA[r]
            self.Binvb[i] = self.Binvb[i] - kk * self.Binvb[r]
        self.d = self.d - (self.d[l - 1] / self.BinvA[r][l - 1]) * self.BinvA[r]
        if self.method == 'M':
            self.dm = self.dm - (self.dm[l - 1] / self.BinvA[r][l - 1]) * self.BinvA[r]
        self.Binvb[r] /= self.BinvA[r][l - 1]
        self.BinvA[r] = self.BinvA[r] / self.BinvA[r][l - 1]
        self.baseX[r] = l

    def _M_cal(self):
        self.c = np.append(self.c, np.zeros(self.A.shape[0]))
        self.cm = np.zeros_like(self.c)
        self.cm[self.A.shape[1]:] = 1
        self.A = np.concatenate((self.A, np.eye(self.A.shape[0])), axis=1)
        self.M = self.A.shape[0]
        self.N = self.c.shape[0]
        self.show_M_problem(self.A, self.b, self.c, self.outfile)
        print(u'画出初始单纯型表', file=self.outfile)
        self.baseX = np.array([i for i in range(self.c.shape[0] - self.M + 1, self.c.shape[0] + 1)], dtype=np.int)
        self.nX = np.array([i for i in range(1, self.c.shape[0] - self.M + 1)], dtype=np.int)
        self.cb = np.array([0 for _ in range(self.c.shape[0] - self.M + 1, self.c.shape[0] + 1)])
        self.cbm = np.array([1 for _ in range(self.c.shape[0] - self.M + 1, self.c.shape[0] + 1)])
        self.cn = copy.deepcopy(self.c[0:self.c.shape[0] - self.M])
        self.cnm = np.zeros_like(self.cn)
        self.Binv = np.eye(self.A.shape[0])
        self.Binvb = copy.deepcopy(self.b)
        self.BinvA = copy.deepcopy(self.A)
        self.d, self.dm = self.cal_delta()
        return self._cal()

    def _force_cal(self):
        # self.show_problem(self.A, self.b, self.c,['=' for _ in range(self.M)], self.outfile)
        print(u'画出初始单纯型表', file=self.outfile)
        times = 0
        self.baseX = np.array([i for i in range(self.N - self.M + 1, self.N + 1)], dtype=np.int)
        self.nX = np.array([i for i in range(1, self.N - self.M + 1)], dtype=np.int)
        while times < 1000000:
            self.cb = copy.deepcopy(self.c[self.baseX - 1])
            self.cn = copy.deepcopy(self.c[self.nX - 1])
            self.Binv = self.A[:, (self.baseX - 1)]
            times += 1
            flag = 1
            if np.fabs(np.linalg.det(self.Binv)) > self.myeps:
                self.Binv = np.linalg.inv(self.Binv)
                if ((self.Binv @ self.b) < 0).sum() > 0:
                    flag = 0
            else:
                flag = 0
            if flag == 1:
                self.Binvb = self.Binv @ self.b
                self.BinvA = self.Binv @ self.A
                self.d, _ = self.cal_delta()
                return self._cal()
            else:
                l = np.random.randint(0, len(self.baseX))
                r = np.random.randint(0, len(self.nX))
                self.baseX[l], self.nX[r] = self.nX[r], self.baseX[l]
        print('选择初始基变量超时',file=self.outfile)
    def _two_cal(self):
        p = self.create_another(self)
        p.c = np.concatenate((np.zeros_like(p.c), np.ones(self.M)))
        print(p.c)
        p.A = np.concatenate((p.A, np.eye(self.M)), axis=1)
        p.M = p.A.shape[0]
        p.N = p.c.shape[0]
        ans, baseX, bb = p._force_cal()
        if (ans is None) or (np.fabs(ans) > self.myeps):
            print('无解')
            print('第一阶段答案不为0，无解', file=self.outfile)
            return None, None, None
        print('第一阶段答案为0，进入第二阶段', file=self.outfile)
        self.baseX = baseX
        baseX_set = set(baseX)
        self.nX = np.array([i for i in range(1, p.N - p.M + 1) if i not in baseX_set])
        jj = 0
        for idx, i in enumerate(self.baseX):
            if i >= p.N - p.M:
                print('交换变量', file=self.outfile)
                p.pivot(self.nX[jj], idx)
                jj += 1
        self.baseX = p.baseX
        baseX_set = set(self.baseX)
        self.nX = np.array([i for i in range(1, p.N - p.M + 1) if i not in baseX_set], dtype=np.int)
        self.cb = copy.deepcopy(self.c[self.baseX - 1])
        self.cn = copy.deepcopy(self.c[self.nX - 1])
        self.Binv = p.BinvA[:, (self.baseX - 1)]
        self.BinvA = p.BinvA[:, 0:p.N - p.M]
        self.Binvb = p.Binvb
        self.N = p.N - p.M
        self.d, _ = self.cal_delta()
        print('构造新的初始单纯型表', file=self.outfile)
        return self._cal()

    def chooseL_M(self):
        idx = 0
        ans = (self.dm[0], self.d[0])
        for i in range(1, self.N):
            if self.dm[i] < ans[0] - self.myeps:
                ans = (self.dm[i], self.d[i])
                idx = i
            elif np.fabs(self.dm[i] - ans[0]) < self.myeps:
                if self.d[i] < ans[1] - self.myeps:
                    ans = (self.dm[i], self.d[i])
                    idx = i
        return idx + 1, ans

    def chooseR(self, col):
        ans = None
        flag = -1
        for idx, i in enumerate(self.BinvA[:, col - 1]):
            if i > 0:
                if ans is None or ans > self.Binvb[idx] / i:
                    ans = self.Binvb[idx] / i
                    flag = idx
        return flag

    def chooseL(self):
        if self.method == 'M':
            return self.chooseL_M()
        idx = 0
        ans = self.d[0]
        for i in range(1, self.N):
            if self.d[i] < ans:
                ans = self.d[i]
                idx = i
        return idx + 1, ans

    @staticmethod
    def create_another(another: "Simplex"):
        p = Simplex(None)
        p.A = another.A
        p.b = another.b
        p.c = another.c
        p.opp = False
        p.M = another.M
        p.N = another.N
        p.outfile = another.outfile
        return p

    def cal_delta(self):
        d = np.zeros_like(self.BinvA[0])
        d[self.nX - 1] = self.cn - self.cb @ self.BinvA[:, self.nX - 1]
        dm = None
        if self.method == 'M':
            dm = np.zeros_like(self.BinvA[0])
            dm[self.nX - 1] = self.cnm - self.cbm @ self.BinvA[:, self.nX - 1]
        return d, dm

    def show_table(self):
        if self.method == 'M':
            self.show_table_M()
            return
        cc = 'c' * (self.N + 2)
        some_var = '&'.join([f'$x_{{{i}}}$' for i in range(1, self.N + 1)])
        some_var = '&' + some_var + '& \\\\ \\hline'
        c_line = ''
        # print(self.d)
        for i in range(self.N):
            c_line += f'& ${round(self.d[i], 3)}$'
        c_line += '&$B^{-1}b$ \\\\ \\hline'
        A_line = ''
        for i in range(self.M):
            A_line += f'$x_{{{self.baseX[i]}}}$'
            A_line += (''.join([f'&{round(self.BinvA[i][j], 3)}' for j in range(self.N)]))
            A_line += f'& {round(self.Binvb[i], 3)} \\\\ '
        tamplete = f'''
\\begin{{table}}[H]
\\centering
\\resizebox{{\\textwidth}}{{14mm}}{{
\\begin{{tabular}}{{{cc}}}
        {some_var}
        {c_line}
        {A_line}
\end{{tabular}} }}
\end{{table}}
'''
        print(tamplete, file=self.outfile)

    def show_table_M(self):
        cc = 'c' * (self.N + 2)
        some_var = '&'.join([f'$x_{{{i}}}$' for i in range(1, self.N + 1)])
        some_var = '&' + some_var + '& \\\\ \\hline'
        c_line = ''
        for i in range(self.N):
            c_tmp = ''
            c_tmp += '& '
            c_tmp += f'${round(self.dm[i], 3)}M+{round(self.d[i], 3)}$'
            c_line += c_tmp
        c_line += '&$B^{-1}b$ \\\\ \\hline'
        A_line = ''
        for i in range(self.M):
            A_line += f'$x_{{{self.baseX[i]}}}$'
            for j in range(self.N):
                A_line += f'&{round(self.BinvA[i][j], 3)}'
            A_line += f'& {round(self.Binvb[i], 3)} \\\\ '
        tamplete = f'''
\\begin{{table}}[H]
\\centering
\\resizebox{{\\textwidth}}{{14mm}}{{
\\begin{{tabular}}{{{cc}}}
        {some_var}
        {c_line}
        {A_line}
\end{{tabular}} }}
\end{{table}}
'''
        print(tamplete, file=self.outfile)

    def show_problem(self, A, b, c, pp, f=sys.stdout):
        if self.max:
            mm = r'\max'
        else:
            mm = r'\min'
        C_line = ''
        for i, j in enumerate(c):
            if i != 0:
                C_line += f'+ {j}x_{{{i + 1}}}'
            else:
                C_line += f' {j}x_{{{i + 1}}}'
        A_limites = []
        b_limites = []
        op_limites = []
        for i, a in enumerate(A):
            tmp_a = ''
            for j, bb in enumerate(a):
                if j != 0:
                    tmp_a += f'+ {bb}x_{{{j + 1}}}'
                else:
                    tmp_a += f' {bb}x_{{{j + 1}}}'
            A_limites.append(tmp_a)
            op_limites.append(pp[i])
            b_limites.append(f' {b[i]}')
        A_line = '\n'.join([f'&{limite}{op}{b} \\\\' for limite, op, b in zip(A_limites, op_limites, b_limites)])
        items = ','.join([f'x_{{{i}}}' for i in range(1, len(A[0]) + 1)])
        last_line = f'&{items}\geq 0'
        templete = f'''
\\begin{{align*}}
{mm} \quad& {C_line} \\\\
\mbox{{s.t.}}\quad 
{A_line}
{last_line}
\end{{align*}}'''
        print(templete, file=f)
        print(file=f)
        # print(a.shape, b.shape, c.shape, file=f)
        # print(a, b, c, file=f)

    def show_M_problem(self, A, b, c, f=sys.stdout):
        print(u'使用大M法', file=f)
        C_line = ''
        for i, j in enumerate(c):
            if i != 0:
                if i >= len(c) - self.M:
                    C_line += f'+ Mx_{{{i + 1}}}'
                else:
                    C_line += f' + {j}x_{{{i + 1}}}'
            else:
                C_line += f' {j}x_{{{i + 1}}}'
        A_limites = []
        b_limites = []
        op_limites = []
        for i, a in enumerate(A):
            tmp_a = ''
            for j, bb in enumerate(a):
                if j != 0:
                    tmp_a += f'+ {bb}x_{{{j + 1}}}'
                else:
                    tmp_a += f' {bb}x_{{{j + 1}}}'
            A_limites.append(tmp_a)
            op_limites.append('=')
            b_limites.append(f' {b[i]}')
        A_line = '\n'.join([f'&{limite}{op}{b} \\\\' for limite, op, b in zip(A_limites, op_limites, b_limites)])
        items = ','.join([f'x_{{{i}}}' for i in range(1, len(A[0]) + 1)])
        last_line = f'&{items}\geq 0'
        templete = f'''
        \\begin{{align*}}
        \min \quad& {C_line} \\\\
        \mbox{{s.t.}}\quad 
        {A_line}
        {last_line}
        \end{{align*}}'''
        print(templete, file=f)
        print(file=f)

    @staticmethod
    def args_to_array(x):
        try:
            ans = np.array(ast.literal_eval(x), dtype=np.float)
            return ans
        except:
            print('formation error')


# 9.1
# python simplex.py --Al [[-1,2],[1,1]] -b [6,5] -c [1,-3] --min
# python simplex.py --Al [[1,-2,1]] --Ag [[-4,1,2]] --Ae [[-2,0,1]] -b [11,3,1] -c [-3,1,1] --min
# 9.3
# python simplex.py --Al [[1,3,0,1],[2,1,0,0],[0,1,4,1]] -b [4,3,3] -c [-2,-4,-1,-1] --min
# python simplex.py --Al [[1,4],[1,1]] -b [8,12] -c [1,2] --max
# python simplex.py --Al [[2,-4,1],[2,3,-1],[6,-1,3]] -b [42,42,42] -c [5,2,8] --max
# 11
# python simplex.py --Ag [[2,0,2],[4,3,1]] -b [4,11] -c [4,2,3] --min
# 10.1
# python simplex.py --Al [[1,2,3]] --Ag [[2,4,0]] --Ae [[1,-1,2]] -b [40,20,15] -c [4,5,3] --max
# 10.2
# python simplex.py --Ag [[1,2,4]] --Ae [[2,1,0]] -b [10,12] -c [24,48,36] --min
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='-A <matrix> -b <vector> -c <vector> [--min | --max ]<None> ')
    parser.add_argument('--Al', type=str, default='[[]]',
                        help='A: Matrix that represents coefficients of constraints.(Ax<=b)')
    parser.add_argument('--Ag', type=str, default='[[]]',
                        help='A: Matrix that represents coefficients of constraints.(Ax>=b)')
    parser.add_argument('--Ae', type=str, default='[[]]',
                        help='A: Matrix that represents coefficients of constraints.(Ax=b)')
    parser.add_argument('-b', type=str,
                        help='b: Ax constrain by b')
    parser.add_argument('-c', type=str,
                        help='c: Coefficients of objective function.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--max", action="store_true", default=False, help='max objective function')
    group.add_argument("--min", action="store_true", default=False, help='min objective function')
    args = parser.parse_args()
    S = Simplex(args, 'test.md')
    S.calculate()
