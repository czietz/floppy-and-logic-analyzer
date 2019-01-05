from struct import pack, iter_unpack
import time
import os
import tempfile
import numpy as np
import saleae
import serial

class SCPWriter:

    # number of tracks in a SCP files
    __NTRACKS = 168

    def __init__(self, imagetype):
        self.__trackdata = dict()
        self.__trackduration = self.__NTRACKS*[0.0]
        self.__imagetype = imagetype

    def fileheader(self):
        # SCP header
        scp_magic = b"SCP"
        scp_vers = 0x17 # version 1.7
        scp_type = self.__imagetype
        scp_nrev = 1 # only one revolution
        scp_starttrack = 0 # always start at track 0  not at min(self.trackdata.keys())
        scp_endtrack = max(self.__trackdata.keys())
        scp_flags = 1 # flux data starts at index
        scp_width = 0 # 16 bits
        scp_heads = 0 # both heads
        scp_res = 0 # 25 ns
        scp_checksum = 0 # TODO

        scp_header = pack("<3sBBBBBBBBBL", scp_magic, scp_vers, scp_type, scp_nrev, scp_starttrack, scp_endtrack, scp_flags, scp_width, scp_heads, scp_res, scp_checksum)
        return scp_header

    def trackoffsettable(self):
        # SCP track table
        scp_tracklist = self.__NTRACKS*[0]
        offs = len(self.fileheader()) + self.__NTRACKS*4 # 1 long word for each track
        for k in range(len(scp_tracklist)):
            try:
                scp_tlen = len(self.trackdata(k)) + len(self.trackheader(k)) # will raise KeyError if track does not exist
                scp_tracklist[k] = offs
                offs = offs + scp_tlen 
            except:
                scp_tracklist[k] = 0  # skip track
        scp_trackoffsets = pack("<%dL" % len(scp_tracklist), *scp_tracklist)
        return scp_trackoffsets

    def trackheader(self, num): # will raise KeyError if track does not exist
        # SCP track header
        scp_tmagic = b"TRK"
        scp_trackno = num
        scp_tduration = round(self.__trackduration[num]/25e-9) 
        scp_tlen = len(self.__trackdata[num]) # in bitcells, not in bytes!
        scp_tstart = 16 # after this header
        scp_trkhead = pack("<3sBLLL", scp_tmagic, scp_trackno, int(scp_tduration), scp_tlen, scp_tstart)
        return scp_trkhead

    def trackdata(self, num): # will raise KeyError if track does not exist
        # round to 25 ns precision
        bitcells = np.round(self.__trackdata[num] / 25e-9)
        # pack into output data
        bitdata = pack(">%dH" % len(bitcells), *bitcells.astype('int'))
        return bitdata

    def loadtrack(self, num, filename):
        # load and process one track
        # tr = np.recfromcsv(filename)
        
        # faster binary file parser: assumes index in channel 1, read_data in channel 3
        temp = iter_unpack("=qQ", open(filename,"rb").read())
        sample_rate = 40e6
        tr = np.rec.array([(k[0]/sample_rate,(k[1]&2) >> 1,(k[1]&8) >> 3) for k in temp], dtype=[('times','f8'),('index','i4'),('read_data','i4')])

        indexpulse = np.where(np.diff(tr.index) == -1) # index transitions from high->low
        trkstart = indexpulse[0][0] + 1
        trkstop = indexpulse[0][-1] + 1
        tr = tr[trkstart:trkstop] # only time between index pulses
        fluxchg = np.where(np.diff(tr.read_data) == -1) # transitions from high->low
        fluxchg = np.add(*fluxchg,1) # +1 so we get the indices where read_data is low
        self.__trackdata[num] = np.diff(tr.times[fluxchg])
        self.__trackduration[num] = tr.times.max() - tr.times.min()

    def saveimage(self, filename):
        with open(filename, "wb") as f:
            f.write(self.fileheader())
            f.write(self.trackoffsettable())
            for k in range(max(self.__trackdata.keys())+1):
                try:
                    f.write(self.trackheader(k))
                    f.write(self.trackdata(k))
                except:
                    pass
            f.write(time.asctime().encode("latin1"))

class LogicAnalyzer:
    def __init__(self):
        self.s = saleae.Saleae()

    def setup(self):
        # not all setup options are exposed via the Socket API, unfortunately.
        # therefore load "canned" settings from file
        self.s.load_from_file(os.path.join(os.getcwd(),"setup.logicsettings"))

    def captureandsave(self, filename):
        self.s.close_all_tabs()
        self.s.capture_start_and_wait_until_finished()
        # faster binary export
        #self.s.export_data2(filename, format="csv", display_base="separate")
        self.s.export_data2(filename, format="binary", each_sample=False, word_size=64)
        while (not self.s.is_processing_complete()):
            time.sleep(.25)

class FloppyDrive:
    # connections between serial cable and floppy:
    # RTS = direction select = green, pin 18
    # DTR = side select = grey, 32
    # TX = step = orange, 20
    # CTS = track 0 = brown, 26
    # GND = GND = black, odd pin
    def __init__(self, portname):
        self.p = serial.Serial(port=portname, baudrate=115200)
        self.track = 0
        self.rezero()
        self.sideselect(0)

    def rezero(self):
        while (not self.p.cts): # track 0 sensor
            self.step("out", settle=False)
        self.track = 0

    # dirstr = "in" (i.e. towards track 79) or "out" (i.e. towards track 0)
    def step(self, dirstr, settle=True):
        if dirstr == "out":
            direction = False
        else:
            direction = True
        self.p.rts = direction
        time.sleep(1e-6) # setup time for direction signal
        # will output one low pulse for the start bit, duration 1/115200 = 8.7µs
        self.p.write(b"\xff")
        self.p.flush()
        if settle:
            time.sleep(20e-3) # head settling time
        else:
            time.sleep(3e-3) # minimum step rate
        if direction:
            if self.track > 0:
                self.track = self.track - 1
        else:
            self.track = self.track + 1

    def sideselect(self, side):
        if side == 1:
            self.p.dtr = True
        else:
            self.p.dtr = False

s = SCPWriter((1<<4) + 5)
l = LogicAnalyzer()
l.setup()
f = FloppyDrive("com3")
f.rezero()

with tempfile.TemporaryDirectory() as tmpdirname:
    fname = os.path.join(tmpdirname,"export.bin")
    for trackno in range(83):
        print("T%dS0" % trackno)
        f.sideselect(0)
        l.captureandsave(fname)
        s.loadtrack(2*trackno + 0, fname)
        print("T%dS1" % trackno)
        f.sideselect(1)
        l.captureandsave(fname)
        s.loadtrack(2*trackno + 1, fname)
        f.step("in")

s.saveimage("floppy2.scp")
