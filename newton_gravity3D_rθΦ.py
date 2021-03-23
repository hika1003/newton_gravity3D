
import numpy as np

# 物理定数, 時刻
G_par = 1
M_par = 2
tmin = 0
tmax = 10
dt = 0.001
Nt = int((tmax - tmin)/dt) + 1


dt = 0.1
Nt = (10**4)*1.4
Nt = int(Nt)
tmin = 0
tmax = dt*Nt


# 時間二階微分以外の項を右辺に移行した後の運動方程式の右辺
# 球座標(r, θ, Φ)での運動方程式

def f_1(M, G, x, xdot):
    r = x[0]
    theta = x[1]
    theta_dot = xdot[1]
    phi_dot = xdot[2]
    #
    return r*theta_dot**2 + r*(phi_dot*np.sin(theta))**2 - (G*M)/(r**2)


def f_2(M, G, x, xdot):
    r = x[0]
    theta = x[1]
    r_dot = xdot[0]
    theta_dot = xdot[1]
    phi_dot = xdot[2]
    #
    return (-2/r)*r_dot*theta_dot + (phi_dot**2)*np.sin(theta)*np.cos(theta)



def f_3(M, G, x, xdot):
    r = x[0]
    theta = x[1]
    r_dot = xdot[0]
    theta_dot = xdot[1]
    phi_dot = xdot[2]
    #
    return -2*((theta_dot*phi_dot)/np.tan(theta) + (r_dot*phi_dot)/r)


def f_vec(M, G, x, xdot):
    lis = [
        f_1(M, G, x, xdot),
        f_2(M, G, x, xdot),
        f_3(M, G, x, xdot),
    ]
    return np.array(lis)


# 変数
x_arr = np.zeros((3, Nt))
x_dot_arr = np.zeros((3, Nt))
t_arr = np.linspace(tmin, tmax, Nt)

# k_[i, RK添え字]
k_x = np.zeros((3, 4))
k_v = np.zeros((3, 4))

# 初期値
x_dot_arr[:, 0] = np.array([-0.02886728, -0.00824957,  0.01750001]) 
x_arr[:, 0] = np.array([17.32050808,  0.95531662, -0.78539816]) 


# データ数削減
sb_dt = 3
sb_Nt = int((tmax - tmin)/sb_dt) + 1
sb_x_arr = np.zeros((3, sb_Nt))
cat_val = int(Nt/sb_Nt)
# 初期化
sb_x_arr[:, 0] = x_arr[:, 0]
sb_i = 0
sb_time = [tmin]


i_val = 1
for i in range(1, Nt):
    x = x_arr[:, i-1]
    xdot = x_dot_arr[:, i-1]
    #
    # ルンゲクッタ
    k_v[:, 0] = f_vec(M_par, G_par, x, xdot)
    #
    k_x[:, 0] = xdot
    #
    for ii in range(1, 4):
        # 係数決定
        if ii != 3:
            delta = dt/2
        else:
            delta = dt
        # 足す
        add_x = delta * k_x[:, ii-1]
        add_v = delta * k_v[:, ii-1]
        #
        k_v[:, 0] = f_vec(M_par, G_par, x + add_x, xdot + add_v)
        k_x[:, 0] = xdot + add_v
    #
    # 漸化式
    x_dot_arr[:, i] = x_dot_arr[:, i-1] + (dt/6)*(k_v[:, 0] + 2*k_v[:, 1] + 2*k_v[:, 2] + k_v[:, 3])
    x_arr[:, i] = x_arr[:, i-1] + (dt/6)*(k_x[:, 0] + 2*k_x[:, 1] + 2*k_x[:, 2] + k_x[:, 3])
    #
    # モニタリング
    print(' r( ', t_arr[i], ' ) = ', x_arr[:, i])
    print('\n')
    i_val += 1
    #
    # サブデータ
    if i % cat_val == 0 and sb_i < sb_Nt:
        sb_x_arr[:, sb_i] = x_arr[:, i]
        sb_time.append(t_arr[i])
        sb_i += 1
    # rが中心近づく
    if x_arr[0, i] <= 0.1:
        break

# 直交座標(x, y, z)に変換
X = sb_x_arr[0, 0:i_val]*np.sin(sb_x_arr[1, 0:i_val])*np.cos(sb_x_arr[2, 0:i_val])
Y = sb_x_arr[0, 0:i_val]*np.sin(sb_x_arr[1, 0:i_val])*np.sin(sb_x_arr[2, 0:i_val])
Z = sb_x_arr[0, 0:i_val]*np.cos(sb_x_arr[1, 0:i_val])


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig.set_dpi(100)
ax = Axes3D(fig)
'''
# 点
ax.scatter3D(0, 0, 0, c='black')
ax.plot(X, Y, Z)

plt.show()
'''



import matplotlib.animation as animation
    
    
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()


def animate(i):
    ax.clear()
    # リム
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    # ラベル
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    # プロット
    ax.scatter3D(0, 0, 0, s=150, c='#ff5613')
    ax.scatter3D(X[i], Y[i], Z[i], c='#00c4b3', s=100)
    ax.plot(X[0:i], Y[0:i], Z[0:i], '--', c='#fcf59c')
    #
    #
    # 数式
    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.rcParams['mathtext.default'] = 'it'
    mathst = r"$\frac{d^2}{d t^2}\vec{r} = \frac{-GM}{|\vec{r}|^2}  \frac{\vec{r}}{|\vec{r}|} $ "
    mathcolor = 'white'
    ax.text3D(0, 10, 10, mathst, size=25, style='italic', c=mathcolor)
    #
    #
    # 背景などのカスタマイズ
    #
    backgoundcolor = 'black'
    #
    #図の色
    fig.patch.set_facecolor(backgoundcolor)  # 図全体の背景色
    fig.patch.set_alpha(0.5)  # 図全体の背景透明度
    ax.patch.set_facecolor(backgoundcolor)  # subplotの背景色
    ax.patch.set_alpha(0.9)  # subplotの背景透明度
    #軸などを消す
    ax.axis("off")



kaisu = sb_Nt - 1
anim = animation.FuncAnimation(fig,animate,frames=kaisu,interval=1)
#anim.save("wave2D-N2.gif", writer="imagemagick")
plt.show()