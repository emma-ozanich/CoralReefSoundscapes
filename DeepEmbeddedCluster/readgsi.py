def readgsi(*argv):#fn, ctstart, tlen, formatt):
    
    import numpy as np
    from datetime import date
    from datetime import datetime
    from datetime import timedelta
    import unicodedata
    
    def read_header(f):
        # # read header # #
        # > == big endian #
        head = dict()
        head['dlabel'] = f.read(10).decode('utf-8')
        head['contents'] = f.read(4).decode('utf-8')
        ordinals = [ord(item)>34 for item in head['contents']]
        head['nc'] = sum(ordinals) # how many larger than 34?

        fillx = np.fromfile(f, dtype=np.uint8, count=50) # filler stuff
        a = np.fromfile(f, '>f8',count=9) # read big-endian float64 encoded
        
        head['ctbc'] = np.int( a[0] + 3600*8) #arbitrary 8 hr scaling (time zone?...)
        head['ctec'] = np.int( a[1] + 3600*8)
        head['tdrift'] = a[2]
        head['Fs'] = a[3]
        head['UTMX'] = a[4]
        head['UTMY'] = a[5]
        head['depth'] = a[6]
        head['UTMZone'] = a[7]
        head['brefa'] = a[8]

        head['ts'] = f.read(1).decode('utf-8')

        head['tabs_start'] = datetime.fromtimestamp(head['ctbc'])
        head['tabs_end'] = datetime.fromtimestamp(head['ctec'])
        
        return head
    
    
    # # Start of function # #
    fn = argv[0]
    f = open(fn, 'rb')
    head = read_header(f)
    
    t = np.empty((1,))
    omi = np.empty((1,))
  
    if len(argv)==1:
        ctstart = 0
        tlen = 24*3600 # 24 hours
        formatt = 'seconds'
        return omi, t, head 
    else:
        ctstart = argv[1]
        tlen = argv[2]
        
        if len(argv)<4:
            formatt = 'seconds'
        else:
            formatt = argv[3]
     
    # some checks on the start time -- mostly not needed in this case
   # if head['ctbc']>ctstart:
   #     dt = ctstart - head['ctbc']
   #     tlen = np.abs( tlen + dt )
   #     ctstart = 0
  #  if ctstart>0:
  #      if formatt=='datenum':
  #          ctstart = 86400*(cstart - date.toordinal( date(1970,1,1) ) + 366)
     #   elif formatt=='seconds':
     #       ctstart = 0
      #  if head['ctbc']>ctstart:
      #      print('error')
      #      return omi, t, head 
        
      #  tlen = ctstart - head['ctbc']
            
    tlen = np.int( tlen )
    fs = head['Fs']*(1 + head['tdrift']/86400)
    lread = (head['nc']*np.floor(tlen*head['Fs'])).astype(int)
    f.seek(512+(head['nc']*np.floor(ctstart*fs)*2).astype(int))
    omi = np.fromfile(f, dtype='>u2', count=lread)
    omi = np.reshape(omi, (head['nc'], np.floor(tlen*head['Fs']).astype(int)),order='F')
    t = np.arange(omi.shape[1]) / head['Fs']
    f.close()
    
    return omi, t, head
    
        
        
    
        
    
    
    