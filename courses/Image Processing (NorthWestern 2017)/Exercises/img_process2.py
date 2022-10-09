
# coding: utf-8

# #Some Image Processing, Information and Coding Theory

# The following problems appeared in the exercises in the **coursera course Image Processing (by NorthWestern University)**. The following descriptions of the problems are taken directly from the assignment's description.

# In[5]:

#ipython nbconvert pcaiso.ipynb
get_ipython().magic(u'matplotlib inline')

from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# ## Some Information and Coding Theory 
# 
# ### Computing the Entropy of an Image 
# 
# The next figure shows the problem statement. Although it was originally implemented in *MATLAB*, in this article a *python* implementation is going to be described.

# In[6]:

from IPython.display import Image
Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\im11.png', width=700)


# ### Histogram Equalization and Entropy
# 
# *Histogram equalization* is a well-known image transformation technique to imrpove the contrast of an image. The following figure shows the theory for the technique: each pixel is transformed by the *CDF* of the image and as can be shown, the output image is expected to follow an **uniform distribution** (and thereby with the **highest entropy**) over the pixel intensities (as proved), considering the continuous pixel density. But since the pixel values are discrete (integers), the result obtained is going to be near-uniform.

# In[4]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\im9.png')


# The following figures show a few images and the corresponding equalized images and how the *PMF* and *CDF* changes.

# In[7]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\beans.png')


# In[8]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\eq.png')


# ### Image Data Compression with Huffman and Lempel Ziv (LZ78) Coding
# 
# Now let's implement the following couple of *compression techniques*:
# 
# 1. Huffman Coding
# 2. Lempel-Ziv Coding (LZ78)
# 
# and compress a few images and their *histogram equalized* versions and compare the *entropies*. 
# 
# The following figure shows the theory and the algorithms to be implemented for these two source coding techniques:

# In[10]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\im12.png')


# Let's now implement the **Huffman Coding algorithm** to compress the data for a few *gray-scale* images. 
# 
# The following figures show how the *tree* is getting constructed with the *Huffman Coding* algorithm (the starting, final and a few intermediate steps) for the following low-contrast image **beans**. Here we have *alphabets* from the set *{0,1,...,255}*, only the pixels present in the image are used. It takes **44 steps** to construct the tree.

# In[24]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\beans.png')


# In[25]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\htree_1_4l.png')


# In[26]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree_beans\\htree_14.png')


# In[28]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree_beans\\htree_40.png')


# In[29]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree_beans\\htree_41.png')


# In[30]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree_beans\\htree_42.png')


# In[31]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree_beans\\htree_43.png')


# In[32]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree_beans\\htree_44.png')


# The following table shows the *Huffman Codes* for different pixel values for the abobe **low-contrast image beans**.

# In[10]:

import pandas as pd
df = pd.read_csv('C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree_beans\\codes.csv')
df.sort(['code'])


# Let's now repeat the **Huffman-tree** construction for the following **histogram-equalized image beans**. The goal is:
# 
# 1. First *construct* the tree for the equalized image.
# 2. Use the tree to **encode** the image data and then **compare** the **compression ratio** with the one obtained using the same algorithm with the low-contrast image. 
# 3. Find if the *histogram equalization* increases / reduces the **compression ratio** or equivalently the **entropy** of the image. 
# 
# The following figures show how the *tree* is getting constructed with the *Huffman Coding* algorithm (the starting, final and a few intermediate steps) for the image **beans**. Here we have *alphabets* from the set *{0,1,...,255}*, only the pixels present in the image are used. It takes **40 steps** to construct the tree, also as can be seen from the following figures the tree constructed is structurally different from the one constructed on the low-contract version of the same image.

# In[23]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\equalized_beans.png')


# In[12]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\htree_1_4.png')


# In[14]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree2_beans\\htree_14.png')


# In[15]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree2_beans\\htree_35.png')


# In[16]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree2_beans\\htree_36.png')


# In[17]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree2_beans\\htree_37.png')


# In[18]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree2_beans\\htree_38.png')


# In[19]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree2_beans\\htree_39.png')


# In[20]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree2_beans\\htree_40.png')


# The following table shows the *Huffman Codes* for different pixel values for the abobe **high-contrast image beans**.

# In[9]:

df = pd.read_csv('C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\Week8\\htree2_beans\\codes.csv')
df.sort(['code'])


# The following figure shows the compression ratio for different images using **Huffman** and **LZ78** codes, both on the **low-contrast** and **high contrast** images (obtained using *histogram equalization*). The following observations can be drawn from the *comparative results* shown in the following figures (here **H** represents Huffman and **L** represents **LZ78**):
# 1. The **entropy** of the image stays almost the same after **histogram equalization** 
# 2. The **compression ratio** with **Huffman** / **LZ78** also stays almost the same with the image with / without **histogram equalization**.
# 3. The **Huffman codes** achieves higher compression in some cases than **LZ78**.

# In[34]:

Image(filename='C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\hist.png')


# The following shows the first few symbol/code pairs for the **dictionary** obtained with **LZ78** algorithm, with the alphabet set as *{0,1,..,255}* for the **low-contrast beans image**:

# In[8]:

import pandas as pd
df = pd.read_csv('C:\\courses\\coursera\\Past\\Image Processing & CV\\NorthWestern - Image Processing\\codes_Z78l.csv')
#pd.options.display.float_format = '{:20,.0f}'.format
#df['Code'] =  df['Code'].astype('str')  #pd.to_numeric(df['Code'], errors='coerce') #df['Code'].astype('int64') 
df.sort(['Code']) #.head(10)

