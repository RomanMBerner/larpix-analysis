import matplotlib.pyplot as plt
import numpy as np
plt.ion()


def vol3d_python3(x,y,z,q,geom_limits,name=None,fig=None,points=False):
    #print('geom_limits: ', geom_limits)
    xyz = np.array(list(zip(x,y,z)))
    #print(' xyz: ', xyz)
    q = q+1e-9
    if not points:
        vox_q, edges = np.histogramdd(xyz, weights=q,
            bins=(
                np.linspace(geom_limits[0],geom_limits[1],
                    int((geom_limits[1]-geom_limits[0])/geom_limits[-2])+1),
                np.linspace(geom_limits[2],geom_limits[3],
                    int((geom_limits[3]-geom_limits[2])/geom_limits[-2])+1),
                np.linspace(geom_limits[4],geom_limits[5],
                    int((geom_limits[5]-geom_limits[4])/geom_limits[-1])+1),
            ))
    if ((np.max(x) - max(np.min(x),0.001))) != 0.:
        #print(' denominator is: ', (np.max(x) - max(np.min(x),0.001)),0,1)
        norm = lambda x: np.clip((x - max(np.min(x),0.001)) / (np.max(x) - max(np.min(x),0.001)),0,1)
        #print(' norm: ', norm)
    else:
        print('WARNING: norm == 0, continue ...')
        return
    cmap = plt.cm.get_cmap('plasma')
    if not points:
        vox_color = cmap(norm(vox_q))
        vox_color[..., 3] = norm(vox_q)
    else:
        vox_color = cmap(norm(q))
        vox_color[..., 3] = norm(q)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    if not points:
        ax.voxels(np.meshgrid(*edges, indexing='ij'), vox_q, facecolors=vox_color)
    else:
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=vox_color,alpha=0.5)
    ax.text2D(0.5, 0.95, name, transform=ax.transAxes)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('t [0.1us]')
    plt.xlim(geom_limits[0],geom_limits[1])
    plt.ylim(geom_limits[2],geom_limits[3])
    plt.tight_layout()
    return fig


def vol3d_python2(x,y,z,q,name,fig):
    from mpl_toolkits.mplot3d import Axes3D
    geom_limits = [-155.15, 155.15, -155.15, 155.15, 0, 1818, 4.43, 18]
    points = True # <<<-------------- initially: False
    #print('geom_limits: ', geom_limits)
    xyz = np.array(list(zip(x,y,z)))
    #print(' xyz: ', xyz)
    q = q+1e-9
    if not points:
        vox_q, edges = np.histogramdd(xyz, weights=q,
            bins=(
                np.linspace(geom_limits[0],geom_limits[1],
                    int((geom_limits[1]-geom_limits[0])/geom_limits[-2])+1),
                np.linspace(geom_limits[2],geom_limits[3],
                    int((geom_limits[3]-geom_limits[2])/geom_limits[-2])+1),
                np.linspace(geom_limits[4],geom_limits[5],
                    int((geom_limits[5]-geom_limits[4])/geom_limits[-1])+1),
            ))
    if ((np.max(x) - max(np.min(x),0.001))) != 0.:
        #print(' denominator is: ', (np.max(x) - max(np.min(x),0.001)),0,1)
        norm = lambda x: np.clip((x - max(np.min(x),0.001)) / (np.max(x) - max(np.min(x),0.001)),0,1)
        #print(' norm: ', norm)
    else:
        print('WARNING: norm == 0, continue ...')
        return
    cmap = plt.cm.get_cmap('plasma')
    if not points:
        vox_color = cmap(norm(vox_q))
        vox_color[..., 3] = norm(vox_q)
    else:
        vox_color = (np.array(q) - np.min(q)) / (np.max(q)-np.min(q)) # cmap(norm(q))
        #vox_color[..., 3] = norm(q)
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111, projection='3d') # <<<-------------- 111 instead of 1,1,1 in python3
    if not points:
        ax.voxels(np.meshgrid(*edges, indexing='ij'), vox_q, facecolors=vox_color)
    else:
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=vox_color,alpha=0.5) #vox_color,alpha=0.5)
    ax.text2D(0.5, 0.95, name, transform=ax.transAxes)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('t [0.1us]')
    plt.xlim(geom_limits[0],geom_limits[1])
    plt.ylim(geom_limits[2],geom_limits[3])
    #plt.zlim(geom_limits[4],geom_limits[5])
    ax.set_zlim(geom_limits[4],geom_limits[5])
    plt.tight_layout()
    return fig


def draw_track(f,track,folder_name):
    hits   = f['hits']
    hit_ref = track['hit_ref']
    '''
    print(' Track ID:              ', track['track_id'])
    print(' Track event ref:       ', track['event_ref'])
    print(' Track hit ref:         ', track['hit_ref'])
    print(' Track theta:           ', track['theta'])
    print(' Track phi:             ', track['phi'])
    print(' Track xp:              ', track['xp'])
    print(' Track yp:              ', track['yp'])
    print(' Track nhit:            ', track['nhit'])
    print(' Track Q:               ', track['q'])
    print(' Track ts_start:        ', track['ts_start'])
    print(' Track ts_end:          ', track['ts_end'])
    print(' Track residual:        ', track['residual'])
    print(' Track lenght:          ', track['length'])
    print(' Track start:           ', track['start'])
    print(' Track end:             ', track['end'])
    print(' eventID:               ', f['events'][track['event_ref']]['evid'][0])
    print(' Hit ID:                ', hits[hit_ref]['hid'])
    print(' Hit x-coord. of track: ', hits[hit_ref]['px'])
    print(' Hit y-coord. of track: ', hits[hit_ref]['py'])
    print(' Hit t-coord. of track: ', hits[hit_ref]['ts'])
    print(' Hit q of event:        ', hits[hit_ref]['q'])
    print(' Hit iochain:           ', hits[hit_ref]['iochain'])
    print(' Hit chip ID:           ', hits[hit_ref]['chipid'])
    print(' Hit channel ID:        ', hits[hit_ref]['channelid'])
    print(' Hit geom:              ', hits[hit_ref]['geom'])
    print(' Hit event_ref:         ', hits[hit_ref]['event_ref'])
    print(' Hit track_ref:         ', hits[hit_ref]['track_ref'])
    '''
    x = hits[hit_ref]['px']
    y = hits[hit_ref]['py']
    z = hits[hit_ref]['ts'] - f['events'][track['event_ref']]['ts_start']
    q = hits[hit_ref]['q']*0.25 # Factor 0.25 from conversion dQ/dx = 0.25ke/mV * 3.9mV/ADC * (Q[ADCs]-78 (pedestal) / deltaX[cm])
    name='trackDisplay_evID_{}_trackID_{}'.format(f['events'][track['event_ref']]['evid'][0],track['track_id'])
    #fig = vol3d_python3(x,y,z,q,*geom_limits,name,fig)
    fig = vol3d_python2(x,y,z,q,name=name,fig=None)#,*geom_limits)
    plt.show()
    plt.savefig(str(folder_name)+'/'+str(name)+'.png')
    plt.close()


def plot_event_ts(events,folder_name):
    # Note: the events['ts_start'] timestamp is a 32 bit timestamp (2^32-1 = 2.147483647e9)
    plt.figure('event ts start')
    t_min = np.min(events['ts_start'])
    t_max = np.max(events['ts_end'])
    #print ' Event ts_min: ', t_min
    #print ' Event ts_max: ', t_max
    n_bins = 50
    plt.hist(events['ts_start'], label='ts_start', bins=np.linspace(t_min,t_max,n_bins), histtype='step')
    plt.hist(events['ts_end'],   label='ts_end',   bins=np.linspace(t_min,t_max,n_bins), histtype='step')
    plt.xlabel('Event Timestamp [s]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.75,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/timestamps.png')
    plt.close()


def plot_event_durations(events,folder_name):
    plt.figure('event durations')
    t_min = np.min((events['ts_end']-events['ts_start']))
    t_max = np.max((events['ts_end']-events['ts_start']))
    n_bins = 50
    plt.hist((events['ts_end']-events['ts_start']), label='ts_end - ts_start', bins=np.linspace(t_min,t_max,n_bins), histtype='step')
    plt.xlabel('Event Duration [0.1 us]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/durations.png')
    plt.close()


def plot_event_unix_ts(events,folder_name):
    plt.figure('event unix ts')
    t_min = np.min(events['unix_ts'])
    t_max = np.max(events['unix_ts'])
    #print ' Event ts_min: ', t_min
    #print ' Event ts_max: ', t_max
    n_bins = 50
    plt.hist(events['unix_ts'], label='unix_ts', bins=np.linspace(t_min,t_max,n_bins), histtype='step')
    plt.xlabel('Event Unix Timestamp')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.75,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/unix_timestamps.png')
    plt.close()


def plot_track_nhits(all_tracks,masked_tracks,folder_name):
    plt.figure('track nhits')
    x_min = np.min(all_tracks['nhit'])
    x_max = np.max(all_tracks['nhit'])
    n_bins = 50
    plt.hist(all_tracks['nhit']   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['nhit'],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Number of Hits [-]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_nhit.png')
    plt.close()


def plot_track_lengths(all_tracks,masked_tracks,folder_name):
    plt.figure('track lengths')
    x_min = np.min(all_tracks['length'])
    x_max = np.max(all_tracks['length'])
    n_bins = 50
    plt.hist(all_tracks['length']   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['length'],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track Length [mm]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_lengths.png')
    plt.close()


def plot_track_charge(all_tracks,masked_tracks,folder_name):
    plt.figure('track charge')
    x_min = np.min(all_tracks['q'])
    x_max = np.max(all_tracks['q'])
    n_bins = 50
    plt.hist(all_tracks['q']   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['q'],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track charge [ke]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_charge.png')
    plt.close()


def plot_track_charge_raw(all_tracks,masked_tracks,folder_name):
    plt.figure('track charge raw')
    x_min = np.min(all_tracks['q_raw'])
    x_max = np.max(all_tracks['q_raw'])
    n_bins = 50
    plt.hist(all_tracks['q_raw']   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['q_raw'],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track charge raw [ke]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_charge_raw.png')
    plt.close()


def plot_track_residuals(all_tracks,masked_tracks,folder_name):
    plt.figure('track residual_x')
    x_min = np.min(all_tracks['residual'][:,0])
    x_max = np.max(all_tracks['residual'][:,0])
    n_bins = 50
    plt.hist(all_tracks['residual'][:,0]   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['residual'][:,0],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track residual x [mm]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_residual_x.png')
    plt.close()

    plt.figure('track residual_y')
    y_min = np.min(all_tracks['residual'][:,1])
    y_max = np.max(all_tracks['residual'][:,1])
    n_bins = 50
    plt.hist(all_tracks['residual'][:,1]   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['residual'][:,1],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track residual y [mm]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_residual_y.png')
    plt.close()

    plt.figure('track residual_z')
    z_min = np.min(all_tracks['residual'][:,2])
    z_max = np.max(all_tracks['residual'][:,2])
    n_bins = 50
    plt.hist(all_tracks['residual'][:,2]   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['residual'][:,2],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track residual z [mm]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_residual_z.png')
    plt.close()


def plot_track_theta(all_tracks,masked_tracks,folder_name):
    plt.figure('track theta')
    x_min = np.min(all_tracks['theta'])
    x_max = np.max(all_tracks['theta'])
    n_bins = 50
    plt.hist(all_tracks['theta']   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['theta'],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track theta [rad]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_theta.png')
    plt.close()


def plot_track_phi(all_tracks,masked_tracks,folder_name):
    plt.figure('track phi')
    z_min = np.min(all_tracks['phi'])
    z_max = np.max(all_tracks['phi'])
    n_bins = 50
    plt.hist(all_tracks['phi']   ,label='all tracks'     ,bins=np.linspace(z_min,z_max,n_bins),histtype='step')
    plt.hist(masked_tracks['phi'],label='selected tracks',bins=np.linspace(z_min,z_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track phi [rad]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_phi.png')
    plt.close()


def plot_track_start_xy(all_tracks,masked_tracks,folder_name):
    plt.figure('track start xy (all)')
    x_min = -150. # np.min(all_tracks['start'][:,0])
    x_max =  150. # np.max(all_tracks['start'][:,0])
    y_min = -150. # np.min(all_tracks['start'][:,1])
    y_max =  150. # np.max(all_tracks['start'][:,1])
    n_bins_x = 50
    n_bins_y = 50
    plt.hist2d(all_tracks['start'][:,0], all_tracks['start'][:,1], (np.linspace(x_min,x_max,n_bins_x),np.linspace(y_min,y_max,n_bins_y)), cmap=plt.cm.jet)
    plt.xlabel('Track start x [mm]')
    plt.ylabel('Track start y [mm]')
    #plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.colorbar()
    plt.savefig(str(folder_name)+'/track_start_xy_all.png')
    plt.close()

    plt.figure('track start xy (selected)')
    x_min = -150. # np.min(all_tracks['start'][:,0])
    x_max =  150. # np.max(all_tracks['start'][:,0])
    y_min = -150. # np.min(all_tracks['start'][:,1])
    y_max =  150. # np.max(all_tracks['start'][:,1])
    n_bins_x = 50
    n_bins_y = 50
    plt.hist2d(masked_tracks['start'][:,0], masked_tracks['start'][:,1], (np.linspace(x_min,x_max,n_bins_x),np.linspace(y_min,y_max,n_bins_y)), cmap=plt.cm.jet)
    plt.xlabel('Track start x [mm]')
    plt.ylabel('Track start y [mm]')
    #plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.colorbar()
    plt.savefig(str(folder_name)+'/track_start_xy_selected.png')
    plt.close()


def plot_track_end_xy(all_tracks,masked_tracks,folder_name):
    plt.figure('track end xy')
    x_min = -150. # np.min(all_tracks['start'][:,0])
    x_max =  150. # np.max(all_tracks['start'][:,0])
    y_min = -150. # np.min(all_tracks['start'][:,1])
    y_max =  150. # np.max(all_tracks['start'][:,1])
    n_bins_x = 50
    n_bins_y = 50
    plt.hist2d(all_tracks['end'][:,0], all_tracks['end'][:,1], (np.linspace(x_min,x_max,n_bins_x),np.linspace(y_min,y_max,n_bins_y)), cmap=plt.cm.jet)
    plt.xlabel('Track end x [mm]')
    plt.ylabel('Track end y [mm]')
    #plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.colorbar()
    plt.savefig(str(folder_name)+'/track_end_xy_all.png')
    plt.close()

    plt.figure('track end xy (selected)')
    x_min = -150. # np.min(all_tracks['start'][:,0])
    x_max =  150. # np.max(all_tracks['start'][:,0])
    y_min = -150. # np.min(all_tracks['start'][:,1])
    y_max =  150. # np.max(all_tracks['start'][:,1])
    n_bins_x = 50
    n_bins_y = 50
    plt.hist2d(masked_tracks['end'][:,0], masked_tracks['end'][:,1], (np.linspace(x_min,x_max,n_bins_x),np.linspace(y_min,y_max,n_bins_y)), cmap=plt.cm.jet)
    plt.xlabel('Track end x [mm]')
    plt.ylabel('Track end y [mm]')
    #plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.colorbar()
    plt.savefig(str(folder_name)+'/track_end_xy_selected.png')
    plt.close()


def plot_track_start_z(all_tracks,masked_tracks,folder_name):
    plt.figure('track start z')
    x_min = np.min(all_tracks['start'][:,2])
    x_max = np.max(all_tracks['start'][:,2])
    n_bins = 50
    plt.hist(all_tracks['start'][:,2]   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['start'][:,2],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track start z [mm]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_start_z.png')
    plt.close()


def plot_track_start_t(all_tracks,masked_tracks,folder_name):
    plt.figure('track start t')
    t_min = np.min(all_tracks['start'][:,3])
    t_max = np.max(all_tracks['start'][:,3])
    n_bins = 50
    plt.hist(all_tracks['start'][:,3]   ,label='all tracks'     ,bins=np.linspace(t_min,t_max,n_bins),histtype='step')
    plt.hist(masked_tracks['start'][:,3],label='selected tracks',bins=np.linspace(t_min,t_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track start t [0.1 us]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_start_t.png')
    plt.close()


def plot_track_end_z(all_tracks,masked_tracks,folder_name):
    plt.figure('track end z')
    z_min = np.min(all_tracks['end'][:,2])
    z_max = np.max(all_tracks['end'][:,2])
    n_bins = 50
    plt.hist(all_tracks['end'][:,2]   ,label='all tracks'     ,bins=np.linspace(z_min,z_max,n_bins),histtype='step')
    plt.hist(masked_tracks['end'][:,2],label='selected tracks',bins=np.linspace(z_min,z_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track end z [mm]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_end_z.png')
    plt.close()


def plot_track_end_t(all_tracks,masked_tracks,folder_name):
    plt.figure('track end t')
    t_min = np.min(all_tracks['end'][:,3])
    t_max = np.max(all_tracks['end'][:,3])
    n_bins = 50
    plt.hist(all_tracks['end'][:,3]   ,label='all tracks'     ,bins=np.linspace(t_min,t_max,n_bins),histtype='step')
    plt.hist(masked_tracks['end'][:,3],label='selected tracks',bins=np.linspace(t_min,t_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track end t [0.1 us]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_end_t.png')
    plt.close()


def plot_track_duration(all_tracks,masked_tracks,folder_name):
    plt.figure('track end t')
    d_min = np.min(all_tracks['end'][:,3]-all_tracks['start'][:,3])
    d_max = np.max(all_tracks['end'][:,3]-all_tracks['start'][:,3])
    n_bins = 50
    plt.hist(all_tracks['end'][:,3]-all_tracks['start'][:,3]   ,label='all tracks'     ,bins=np.linspace(d_min,d_max,n_bins),histtype='step')
    plt.hist(masked_tracks['end'][:,3]-masked_tracks['start'][:,3],label='selected tracks',bins=np.linspace(d_min,d_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track duration [0.1 us]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_duration.png')
    plt.close()


'''
def plot_track_hit_fraction(all_tracks,masked_tracks):
    plt.figure('track hit fraction')
    x_min = np.min(all_tracks['phi'])
    x_max = np.max(all_tracks['phi'])
    n_bins = 50
    print('x_max: ', x_max)
    plt.hist(all_tracks['phi']   ,label='all tracks'     ,bins=np.linspace(x_min,x_max,n_bins),histtype='step')
    plt.hist(masked_tracks['phi'],label='selected tracks',bins=np.linspace(x_min,x_max,n_bins),histtype='step')#,bins=np.linspace(0,100,100),)
    plt.xlabel('Track phi [rad]')
    plt.ylabel('Entries [-]')
    plt.legend(loc=[0.6,0.85], prop={'size': 10})
    plt.savefig(str(folder_name)+'/track_phi.png')
    plt.close()
'''
