import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import shapefile
import numpy as np
import matplotlib.pyplot as plt 
from pykrige.ok import OrdinaryKriging
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")

def add_north(ax, labelsize=18, loc_x=0.88, loc_y=0.85, width=0.06, height=0.09, pad=0.14):
    """
    画一个比例尺带'N'文字注释
    主要参数如下
    :param ax: 要画的坐标区域 Axes实例 plt.gca()获取即可
    :param labelsize: 显示'N'文字的大小
    :param loc_x: 以文字下部为中心的占整个ax横向比例
    :param loc_y: 以文字下部为中心的占整个ax纵向比例
    :param width: 指南针占ax比例宽度
    :param height: 指南针占ax比例高度
    :param pad: 文字符号占ax比例间隙
    :return: None
    """
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen*(loc_x - width*.5), miny + ylen*(loc_y - pad)]
    right = [minx + xlen*(loc_x + width*.5), miny + ylen*(loc_y - pad)]
    top = [minx + xlen*loc_x, miny + ylen*(loc_y - pad + height)]
    center = [minx + xlen*loc_x, left[1] + (top[1] - left[1])*.4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen*loc_x,
            y=miny + ylen*(loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.add_patch(triangle)
    
def add_scalebar(ax,lon0,lat0,length,size=0.45):
    '''
    ax: 坐标轴
    lon0: 经度
    lat0: 纬度
    length: 长度
    size: 控制粗细和距离的
    '''
    # style 3
    ax.hlines(y=lat0,  xmin = lon0, xmax = lon0+length/111, colors="black", ls="-", lw=1, label='%dkm' % (length))
    ax.vlines(x = lon0, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=1)
    ax.vlines(x = lon0+length/2/111, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=1)
    ax.vlines(x = lon0+length/111, ymin = lat0-size, ymax = lat0+size, colors="black", ls="-", lw=1)
    ax.text(lon0+length/111,lat0+size+0.01,'%d' % (length),horizontalalignment = 'center')
    ax.text(lon0+length/2/111,lat0+size+0.01,'%d' % (length/2),horizontalalignment = 'center')
    ax.text(lon0,lat0+size+0.01,'0',horizontalalignment = 'center')
    ax.text(lon0+length/111/2*3-0.04,lat0+size+0.01,'(km)',horizontalalignment = 'center')

sf=shapefile.Reader('D:\\jupyter_data\\cnmap\\lean.shp')

for shape_rec in sf.shapeRecords():
    if shape_rec.record[1] == '乐安县':#Hunan Sheng
        print(shape_rec.record)
        #print(shape_rec.shape.points)
        
lons = [115.7008,115.7097,116.0033,115.9333,115.9369,115.73,115.8061,115.6767,115.9111,115.7867,115.9186,115.8217
       ,115.7853,115.8475,115.71,115.8689,115.8333]
lats = [27.5236,27.3083,27.3831,27.1139,27.0381,27.3992,27.6356,27.1633,27.6564,27.1933,27.3144,27.5369,27.6011,27.3492
        ,27.22,27.1733,27.4333]
#t_ave = [27.7,28.2,26.1,26.2,26.3,27.7,28.2,28.1,28.1,28.1,27.8,27.9,28.1,27.8,28.3,27.5,27.9]
#t_max = [33.0 ,33.6 ,32.6 ,32.2 ,32.7 ,33.8 ,33.3 ,33.3 ,33.4 ,33.7 ,33.5 ,34.0 ,33.6 ,33.8 ,34.0 ,33.7 ,33.2 ]
#t_min = [23.8,24.3,22.3,22.5,22.4,23.4,24.4,24.5,24.2,24.0 ,23.8,23.8,24.2,23.7,24.2,23.3,24.8]
#rain = [6.6, 6.0, 7.3, 8.8, 6.5, 7.0, 6.7, 6.4, 6.4, 7.2, 7.0, 7.0, 6.4, 6.9, 7.0 ,6.5, 7.1]
#wind_ave = [1.25,1.3,0.91,1.08,0.76,1.07,1.86,1.37,1.34,0.98,1.26,1.24,1.81,1.54,1.30 ,1.11,1.52]
wind_max = [4.23,4.14,3.15,3.62,2.54,3.43,4.63,4,4,2.92,3.89,3.56,4.76,4.92,4.05,3.58,4.69]
station = ['戴坊','牛田','谷岗','金竹','金竹瀑布','大马头','山砀','罗陂','公溪镇','湖坪乡','南村乡','湖溪乡',
           '龚坊镇','增田镇','万崇镇','招携镇','乐安']

grid_lon = np.linspace(115.4, 116.4,1300)
grid_lat = np.linspace(26.8, 27.8,1300)

OK = OrdinaryKriging(lons, lats, wind_max, variogram_model='gaussian',nlags=6)
z1, ss1 = OK.execute('grid', grid_lon, grid_lat)
z1.shape

xgrid, ygrid = np.meshgrid(grid_lon, grid_lat)

df_grid = pd.DataFrame(dict(long=xgrid.flatten(),lat=ygrid.flatten()))

df_grid["Krig_gaussian"] = z1.flatten()

fig = plt.figure(figsize=[7,9]) 
ax = fig.add_subplot(111)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams.update({'font.size':13})

for shape_rec in sf.shapeRecords():
    if shape_rec.record[1] == '乐安县':#Hunan Sheng
        vertices = []
        codes = []
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i+1]):
                vertices.append((pts[j][0], pts[j][1]))
            codes += [Path.MOVETO]
            codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
            codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData)
def makedegreelabel(degreelist):
    labels=[str(x)+u'°E' for x in degreelist]
    return labels

m = Basemap(llcrnrlon=115.4,
    llcrnrlat=26.82,
    urcrnrlon=116.2,
    urcrnrlat=27.84,
    resolution = None, 
    projection = 'cyl')

cbar_kwargs = {
    'orientation': 'horizontal',
    'label': 'Temperature (℃)',
    'shrink': 0.05,
    'ticks': np.arange(0, 10 + 1, 1),
    'pad': -0.01,
    'shrink': 1.2
}

levels = np.linspace(3.0, 4.3, 200)  

#cs = ax.contourf(xgrid,ygrid,z1,levels=levels,cbar_kwargs=cbar_kwargs,cmap='RdYlBu_r')
cs = ax.contourf(xgrid,ygrid,z1,levels=levels,cmap='cool')

m.readshapefile('D:\\jupyter_data\\cnmap\\lean',"Le'an map",color='k',linewidth=1.2)

for contour in cs.collections:
        contour.set_clip_path(clip)
        
parallels = np.arange(26.8,28,0.2)
m.drawparallels(parallels,labels=[True,True,True,True],color='dimgrey',dashes=[1, 3])
meridians = np.arange(115.4,116.4,0.2)
m.drawmeridians(meridians,labels=[True,True,True,True],color='dimgrey',dashes=[1, 3])

plt.scatter(lons, lats,marker='.',s=100 ,color ="black")
for sta in np.arange(0,17,1):
    plt.text(lons[sta]-0.04, lats[sta]+0.01, station[sta], fontsize= 10 )
#plt.text(bill2-0.4, tip2+0.2, u"长沙市", fontsize= 20 ,fontproperties=ZHfont)

add_north(ax) 
add_scalebar(ax,115.5,26.85,20,size=0.02) # 添加比例尺

plt.ylabel('')    #Remove the defult  lat / lon label  
plt.xlabel('')
cax = fig.add_axes([ax.get_position().x0+0.03,ax.get_position().y0+0.08,0.03,ax.get_position().height-0.1])#(左，下，宽，高)
cb=plt.colorbar(cs,cax=cax,orientation='vertical',ticks=np.arange(3.0,4.3,0.2))#方向
cb.set_label('(m/s)')

#plt.savefig('D:\\jupyter_data\\lean\\最大风速空间分布.png')
