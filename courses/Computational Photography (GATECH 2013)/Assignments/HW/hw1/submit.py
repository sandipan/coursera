

### The only things you'll have to edit (unless you're porting this script over to a different language) 
### are at the bottom of this file.

import urllib
import urllib2
import hashlib
import random
import email
import email.message
import email.encoders
import StringIO
import sys

""""""""""""""""""""
""""""""""""""""""""

class NullDevice:
  def write(self, s):
    pass

def submit():   
  print '==\n== [compphoto] Submitting Solutions \n=='
  
  (login, password) = loginPrompt()
  if not login:
    print '!! Submission Cancelled'
    return
  
  print '\n== Connecting to Coursera ... '

  # Part Identifier
  choice = partPrompt()

  partIdx = []
  if choice == 0:
    partIdxs = range(len(partIds))
  else:
    partIdxs = [choice-1]

  for partIdx in partIdxs:
    sid = partIds[partIdx]

    # Get Challenge
    (login, ch, state, ch_aux) = getChallenge(login, sid) #sid is the "part identifier"
    if((not login) or (not ch) or (not state)):
      # Some error occured, error string in first return element.
      print '\n!! Error: %s\n' % login
      return

    # Attempt Submission with Challenge
    ch_resp = challengeResponse(login, password, ch)
    (result, string) = submitSolution(login, ch_resp, sid, output(partIdx), \
                                    source(partIdx), state, ch_aux)

    print '== %s' % string.strip()


# =========================== LOGIN HELPERS - NO NEED TO CONFIGURE THIS =======================================

def loginPrompt():
  """Prompt the user for login credentials. Returns a tuple (login, password)."""
  (login, password) = basicPrompt()
  return login, password

def basicPrompt():
  """Prompt the user for login credentials. Returns a tuple (login, password)."""
  login = raw_input('Login (Email address): ')
  password = raw_input('One-time Password (from the assignment page. This is NOT your own account\'s password): ')
  return login, password

def partPrompt():
  print 'Hello! These are the assignment parts that you can submit:'
  counter = 0
  print '0) Submit all parts'
  for part in partFriendlyNames:
    counter += 1
    print str(counter) + ') ' + partFriendlyNames[counter - 1]
  choice = int(raw_input('Please enter which part you want to submit (0-' + str(counter) + '): '))
  return choice 

def getChallenge(email, sid):
  """Gets the challenge salt from the server. Returns (email,ch,state,ch_aux)."""
  url = challenge_url()
  values = {'email_address' : email, 'assignment_part_sid' : sid, 'response_encoding' : 'delim'}
  data = urllib.urlencode(values)
  req = urllib2.Request(url, data)
  response = urllib2.urlopen(req)
  text = response.read().strip()

  # text is of the form email|ch|signature
  splits = text.split('|')
  if(len(splits) != 9):
    print 'Badly formatted challenge response: %s' % text
    return None
  return (splits[2], splits[4], splits[6], splits[8])

def challengeResponse(email, passwd, challenge):
  sha1 = hashlib.sha1()
  sha1.update("".join([challenge, passwd])) # hash the first elements
  digest = sha1.hexdigest()
  strAnswer = ''
  for i in range(0, len(digest)):
    strAnswer = strAnswer + digest[i]
  return strAnswer 
  
def challenge_url():
  """Returns the challenge url."""
  return "https://class.coursera.org/" + URL + "/assignment/challenge"

def submit_url():
  """Returns the submission url."""
  return "https://class.coursera.org/" + URL + "/assignment/submit"

def submitSolution(email_address, ch_resp, sid, output, source, state, ch_aux):
  """Submits a solution to the server. Returns (result, string)."""
  source_64_msg = email.message.Message()
  source_64_msg.set_payload(source)
  email.encoders.encode_base64(source_64_msg)

  output_64_msg = email.message.Message()
  output_64_msg.set_payload(output)
  email.encoders.encode_base64(output_64_msg)
  values = { 'assignment_part_sid' : sid, \
             'email_address' : email_address, \
             #'submission' : output, \
             'submission' : output_64_msg.get_payload(), \
             #'submission_aux' : source, \
             'submission_aux' : source_64_msg.get_payload(), \
             'challenge_response' : ch_resp, \
             'state' : state \
           }
  url = submit_url()  
  data = urllib.urlencode(values)
  req = urllib2.Request(url, data)
  response = urllib2.urlopen(req)
  string = response.read().strip()
  result = 0
  return result, string

## This collects the source code (just for logging purposes) 
def source(partIdx):
  # open the file, get all lines
  f = open(sourceFiles[partIdx])
  src = f.read() 
  f.close()
  return src

############ BEGIN ASSIGNMENT SPECIFIC CODE - YOU'LL HAVE TO EDIT THIS ##############

import part0, part1, part2, part3, part4

# Make sure you change this string to the last segment of your class URL.
# For example, if your URL is https://class.coursera.org/pgm-2012-001-staging, set it to "pgm-2012-001-staging".
URL = 'compphoto-001'

# the "Identifier" you used when creating the part
partIds = ['convolve', 'gaussian', 'sharpen', 'median', 'thetamag']                        
# used to generate readable run-time information for students
partFriendlyNames = ['2d convolution function', 'gaussian kernel function', \
    'sharpen filter function', 'median filter function', 'theta and magnitude'] 
# source files to collect (just for our records)
sourceFiles = ['part0.py', 'part1.py', 'part2.py', 'part3.py', 'part4.py', 'run.py']                           
          
def output(partIdx):
  """Uses the student code to compute the output for test cases."""
  outputString = ''

  if partIdx == 0: # convolve 
    outputString = str(int(part0.test()))
  elif partIdx == 1: # gaussian 
    outputString = str(int(part1.test()))
  elif partIdx == 2: # sharpen 
    outputString = str(int(part2.test()))
  elif partIdx == 3: # median 
    outputString = str(int(part3.test()))
  elif partIdx == 4: # theta and mag 
    outputString = str(int(part4.test()))

  return outputString

submit()
