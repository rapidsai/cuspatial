def get_ts_struct(ts):
	y=ts&0x3f
	ts=ts>>6
	m=ts&0xf
	ts=ts>>4
	d=ts&0x1f
	ts=ts>>5
	hh=ts&0x1f
	ts=ts>>5
	mm=ts&0x3f
	ts=ts>>6
	ss=ts&0x3f
	ts=ts>>6
	wd=ts&0x8
	ts=ts>>3
	yd=ts&0x1ff
	ts=ts>>9
	ms=ts&0x3ff
	ts=ts>>10
	pid=ts&0x3ff

	return y,m,d,hh,mm,ss,wd,yd,ms,pid

