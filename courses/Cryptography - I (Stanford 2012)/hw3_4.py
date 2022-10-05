#HW3
from Crypto.Hash import SHA256
import os

def getHash(blk):
  sha = SHA256.new()
  sha.update(blk)
  return sha.digest() # sha.digest gives the raw form

file_handle = open("1.mp4", "r")
file_handle.seek(0, os.SEEK_END)
loc = file_size = file_handle.tell()
print loc
blksz = file_size % 1024
if blksz == 0:
  blksz = 1024
hsh = ''
while loc > 0:
  file_handle.seek(-blksz, os.SEEK_CUR)
  loc = file_handle.tell()
  blk = file_handle.read(blksz)
  blk += hsh
  file_handle.seek(-blksz, os.SEEK_CUR)
  hsh = getHash(blk)
  blksz = 1024

print hsh.encode('hex')

#HW4
import urllib2
import sys

TARGET = 'http://crypto-class.appspot.com/po?er='
#--------------------------------------------------------------
# padding oracle
#--------------------------------------------------------------
class PaddingOracle(object):
    def query(self, q):
        target = TARGET + urllib2.quote(q)    # Create query URL
        req = urllib2.Request(target)         # Send HTTP request to server
        try:
            f = urllib2.urlopen(req)          # Wait for response
        except urllib2.HTTPError, e:          
            print "We got: %d" % e.code       # Print response code
            if e.code == 404:
                return True # good padding
            return False # bad padding

#if __name__ == "__main__":
c = ['f20bdba6ff29eed7b046d1df9fb70000',
     '58b1ffb4210a580f748b4ac714c001bd',
     '4a61044426fb515dad3f21f18aa577c0',
     'bdf302936266926ff37dbf7035d5eeb4']
     
po = PaddingOracle()

po.query(sys.argv[1])       # Issue HTTP query with the given argument
