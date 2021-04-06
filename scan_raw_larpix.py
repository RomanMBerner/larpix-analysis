# Quick-and-dirty script to scan raw LArPix data files
# dadwyer@lbl.gov, 1 Apr 2021

import h5py
import numpy as np

import argparse

import yaml

import larpix.format.rawhdf5format as rh5
import larpix.format.hdf5format as h5
from larpix.format.pacman_msg_format import parse
from larpix import PacketCollection

parser = argparse.ArgumentParser(description='Scan raw larpix file')
parser.add_argument('--infile',
                    help='input larpix raw event file')
parser.add_argument('--geomfile',
                    help='larpix geometry file')

class RawFileStreamer(object):
    """Stream and process chunks of raw data in memory"""
    def __init__(self, filename, chunksize=25600):
        """Constructor"""
        self._filename = filename  # Name of file in raw hdf5 format
        self._chunksize = chunksize  # Default size of blocks to read
        self._raw_file_length = 0 # Length of raw hdf5 file
        self._current_read_index = 0 # Index of current read
        self._is_open = False
        
    def open(self):
        """Open the raw file"""
        if self._is_open:
            print("Why are you trying to open the file again?")
            return True
        try:
            length = rh5.len_rawfile(self._filename)
            self._raw_file_length = length
            self._current_read_index = 0
            self._is_open = True
            # DD: I would usually keep an open file handle here, but
            # rh5 format module doesn't support such a read mode.
            return True
        except Exception as e:
            print("Couldn't open file: ",self._filename)
            return False
    
    def next_chunk(self):
        """Read and parse the next chunk of raw data, return formatted data"""
        # Check if the file has been 'opened'
        if not self._is_open:
            print("You should open the file before reading!")
            return None
        # Check if at end of file
        if self._current_read_index == self._raw_file_length:
            # End of file
            return None
        # Set the chunk start and end indices
        start = self._current_read_index
        end = start + self._chunksize
        if end > self._raw_file_length: end = self._raw_file_length
        # Try to read the chunk
        try:
            rd = rh5.from_rawfile(self._filename, start=start, end=end)
            self._current_read_index = end
        except Exception as e:
            print("Couldn't read file: ",self._filename)
            return None
        # Associate io group to packets, following Peter's example
        pkts = list()
        for io_group,msg in zip(rd['msg_headers']['io_groups'], rd['msgs']):
            pkts.extend(parse(msg, io_group=io_group))
            #pkts.extend(msg)
        # Return the packets
        return pkts


class PixelMapper(object):
    """Map channels to pixel locations in TPC"""
    def __init__(self, geom_filename):
        """Constructor"""
        # Name of geometry file in yaml format
        self._geom_filename = geom_filename
        self._geom_data = None
        self._load()
        self._iogc_to_tileid = None
        self._pixel_map = {}
        self._build_pixel_map()

    def _load(self):
        """Load the geometry data from file into memory"""
        with open(self._geom_filename, 'r') as stream:
            try:
                geom_data = yaml.safe_load(stream)
                self._geom_data = geom_data
            except yaml.YAMLError as exc:
                print(exc)

    def _build_pixel_map(self):
        """Build map of (iogroup, iochannel, chipid, channelid) 
        to pixel position and direction"""
        # Tile position and orientation/rotation
        self._tile_positions = self._geom_data['tile_positions']
        self._tile_orientations = self._geom_data['tile_orientations']
        # Pixel pitch
        pixel_pitch = self._geom_data['pixel_pitch']
        # Map chip/channel id to position
        for (tileid, chips_to_io) in self._geom_data['tile_chip_to_io'].items():
            tile_center_xyz = self._tile_positions[tileid]
            tile_orientation = self._tile_orientations[tileid]
            for (chipchanid, posidx) in self._geom_data['chip_channel_to_position'].items():
                channelid = int(chipchanid % 1000)
                chipid = int((chipchanid - channelid) / 1000)
                # Retrieve iogroup, iochannel
                if not chipid in chips_to_io:
                    # Skip chips missing from tile I/O map
                    continue
                iogc = chips_to_io[chipid]
                iochannel = int(iogc % 1000)
                iogroup = int((iogc - iochannel) / 1000)
                # Calculate pixel position and orientation
                #  Relative position (to tile center),
                #  in tile coordinate system
                pix_pos_xyz_rel = [(35-posidx[1])*pixel_pitch,
                                   (35-posidx[0])*pixel_pitch,
                                   0]
                #  Relative position (to tile center),
                #  now in detector coordinate system
                pix_pos_xyz_dabs = [pix_pos_xyz_rel[0]*tile_orientation[2],
                                    pix_pos_xyz_rel[1]*tile_orientation[1],
                                    0]
                #  Absolute position, in detector coordinate system
                pix_pos_xyz = [tile_center_xyz[1] + pix_pos_xyz_dabs[0],
                               tile_center_xyz[2] + pix_pos_xyz_dabs[1],
                               tile_center_xyz[0] + pix_pos_xyz_dabs[2]]
                pix_orient_xyz = [0,
                                  0,
                                  tile_orientation[0]]
                syschannelid = (iogroup, iochannel, chipid, channelid)
                self._pixel_map[syschannelid] = (
                    pix_pos_xyz,
                    pix_orient_xyz)
        print("N Pixels loaded: ",len(self._pixel_map))
        print(list(self._pixel_map.keys())[0:10])
        return

    def get_pixel_geom(self, iogroup, iochannel, chipid, channelid):
        """Return geometry information for pixel, given channel info"""
        try: 
            return self._pixel_map[(iogroup, iochannel, chipid, channelid)]
        except Exception:
            return None


class HitQueueManager(object):
    """Circular queue for managing hits and event assembly"""
    def __init__(self):
        """Constructor"""
        self._event_end_idx = 0
        self._write_idx = 0
        
    def update_queue(self, hit_queue, packets, pixmap):
        """Update queue with additional packets"""
        cur_size = self._write_idx - self._event_end_idx
        # First, shift unused hits to start of queue
        if cur_size > 0:
            for (key, val) in hit_queue.items():
                val[0:cur_size] = val[self._event_end_idx:self._write_idx]
        self._write_idx = cur_size
        self._event_end_idx = 0
        # Append new hits to queue
        #  Step 0: Filter for data packets
        data_packets = []
        last_gts_packet = None
        for packet in packets:
            if packet.packet_type == 4:
                # Timestamp packet
                last_gts_packet = packet
                continue
            elif packet.packet_type == 0:
                # Data packet, add to list
                data_packets.append(packet)
            else:
                # Other packet type
                pass
                #print("Skipping packet type:",packet.packet_type)
        packets = data_packets
        #  Step 1: check if there is enough space.
        n_packets = len(packets)
        n_space = len(hit_queue['hit_ts']) - cur_size
        remaining_packets = []
        if n_packets > n_space:
            n_packets = n_space
            remaining_packets = packets[n_packets:]
            packets = packets[:n_packets]
        #  Step 2: update values
        print("Loading ",len(packets)," packets to hit queue")
        write_start = self._write_idx
        write_end = self._write_idx + n_packets
        hit_queue['hit_ts'][write_start:write_end] = [packet.timestamp for packet in packets]
        hit_queue['io_group'][write_start:write_end] = [packet.io_group for packet in packets]
        hit_queue['io_channel'][write_start:write_end] = [packet.io_channel for packet in packets]
        hit_queue['chip_id'][write_start:write_end] = [packet.chip_id for packet in packets]
        hit_queue['channel_id'][write_start:write_end] = [packet.channel_id for packet in packets]
        hit_queue['hit_adc'][write_start:write_end] = [packet.dataword for packet in packets]
        #   Step 3: Load pixel location and drift direction geometry info
        for idx in range(write_start, write_end):
            pix_geom = pixmap.get_pixel_geom(hit_queue['io_group'][idx],
                                             hit_queue['io_channel'][idx],
                                             hit_queue['chip_id'][idx],
                                             hit_queue['channel_id'][idx])
            if pix_geom is None: continue
            hit_queue['hit_px'][idx] = pix_geom[0][0]
            hit_queue['hit_py'][idx] = pix_geom[0][1]
            hit_queue['hit_pz'][idx] = pix_geom[0][2]
            hit_queue['hit_dx'][idx] = pix_geom[1][0]
            hit_queue['hit_dy'][idx] = pix_geom[1][1]
            hit_queue['hit_dz'][idx] = pix_geom[1][2]
        self._write_idx = write_end
        return remaining_packets

    '''
    #FIXME: with a little more work, it should be possible to 
    # cluster hits by time gaps, similar to the event builder.
    def find_next_event(self, hit_queue, dt_gap, dt_rollover):
        """Find the start and end index of the next event in the queue"""
        # Advance to next event start
        event_start_idx = self._event_end_idx
        for idx in range(event_start_idx+1, self._write_idx):
            # Find gap dividing this event from the next event
            is_end = (hit_queue['hit_ts'][idx]
                      -hit_queue['hit_ts'][idx-1]) > dt_gap
            is_rollover = (hit_queue['hit_ts'][idx]
                           -hit_queue['hit_ts'][idx-1]) < dt_rollover
            if (is_end or is_rollover):
                # Found gap or rollover.  Define event
                self._event_end_idx = idx
                return (event_start_idx, self._event_end_idx)
        # Hit end of hit queue, send null event
        return (event_start_idx, event_start_idx)
    '''    

    def find_next_nhits(self, hit_queue, n_hits):
        """Return the next n pixel hits from the queue"""
        # Advance to next event start
        event_start_idx = self._event_end_idx
        if (self._write_idx - self._event_end_idx) > n_hits:
            self._event_end_idx += n_hits
            return (event_start_idx, self._event_end_idx)
        # Hit end of hit queue, send null event
        return (event_start_idx, event_start_idx)    
    
if "__main__" == __name__:

    args = parser.parse_args()
    
    # Test parsing of geometry layout yaml
    pixmap = PixelMapper(args.geomfile)
    
    # Test streaming read of a larpix raw hdf5 file
    #filename = "raw_2021_04_01_11_46_23_CEST.h5"
    rfs = RawFileStreamer(args.infile) 
    
    open_status = rfs.open()
    #print("status (open): ",open_status)

    max_packets = 100000
    # Event data queue
    #  Raw data
    hit_ts = np.zeros(max_packets, dtype='int')
    io_group = np.zeros(max_packets, dtype='int64')
    io_channel = np.zeros(max_packets, dtype='int')
    chip_id = np.zeros(max_packets, dtype='int')
    channel_id = np.zeros(max_packets, dtype='int')
    hit_adc = np.zeros(max_packets, dtype='int')
    #  Interpreted data (hit level)
    hit_px = np.zeros(max_packets, dtype='float')
    hit_py = np.zeros(max_packets, dtype='float')
    hit_pz = np.zeros(max_packets, dtype='float')
    hit_dx = np.zeros(max_packets, dtype='float')
    hit_dy = np.zeros(max_packets, dtype='float')
    hit_dz = np.zeros(max_packets, dtype='float')
    #  Interpreted data (event level)
    hit_x_est = np.zeros(max_packets, dtype='float')
    hit_y_est = np.zeros(max_packets, dtype='float')
    hit_z_est = np.zeros(max_packets, dtype='float')
    
    hit_queue = {'hit_ts':hit_ts,
                 'io_group':io_group,
                 'io_channel':io_channel,
                 'chip_id':chip_id,
                 'channel_id':channel_id,
                 'hit_adc':hit_adc,
                 'hit_px':hit_px,
                 'hit_py':hit_py,
                 'hit_pz':hit_pz,
                 'hit_dx':hit_dx,
                 'hit_dy':hit_dy,
                 'hit_dz':hit_dz,
                 }
    
    
    
    ev_dt_gap_ticks = 5000  # 500 us
    ev_dt_rollover_ticks = -1e8  # catch rollover
    max_events = 10
    drift_vel = 1.68/10. # mm / clock-tick

    hit_queue_mgr = HitQueueManager()

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[8,6])
    ax = fig.add_subplot(projection='3d')
    
    remaining_packets = []
    while True:
        # Process and display event
        #  Step 1: Check for next set of hits
        
        (start_idx, end_idx) = hit_queue_mgr.find_next_nhits(hit_queue,
                                                             1000)
        event_nhit = end_idx - start_idx                                   
        if event_nhit > 0:
            # Process event
            hit_x_est[0:event_nhit] = hit_px[start_idx:end_idx]*1e-3
            hit_y_est[0:event_nhit] = hit_py[start_idx:end_idx]*1e-3
            hit_z_est[0:event_nhit] = hit_dz[start_idx]*(hit_ts[start_idx:end_idx]-np.mean(hit_ts[start_idx:end_idx]))*drift_vel*1e-3
            ax.cla()
            ax.view_init(30, 30)
            ax.scatter(hit_y_est[0:event_nhit],
                           hit_z_est[0:event_nhit],
                           hit_x_est[0:event_nhit],
                           #norm=cm_norm,
                           edgecolors='face',
                           #vmin=0, vmax = 100,
                           #c='b', norm=1,
                           marker='o', alpha=1,
                           s=2, linewidth=0)
            ax.set_xlabel('y [m]')
            ax.set_ylabel('z(t) [m]')
            ax.set_zlabel('x [m]')
            ax.set_xlim(-0.4,0.4)
            ax.set_ylim(-50,50)
            ax.set_zlim(-0.8,0.8)
            ax.set_xticks([-0.4,-0.2,0,0.2,0.4])
            ax.set_yticks([-50,-25,0,25,50])
            ax.set_zticks([-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8])
            ax.set_box_aspect((1,1,2))
            filename = str(args.infile[str(args.infile).find('raw_'):])
            plt.title('Raw Data Event Display by Dan Dwyer\n'+filename)
            plt.pause(3)
        else:
            # Add data to queue
            new_packets = rfs.next_chunk()
            if new_packets is None:
                print("Reached end of data stream")
                break
            packets = remaining_packets + new_packets
            remaining_packets = hit_queue_mgr.update_queue(hit_queue,
                                                           packets,
                                                           pixmap)
        # end of loop
