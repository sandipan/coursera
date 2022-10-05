// Stanford Crypto Exercises
#include <string>
#include <map>
#include <set>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cctype>

int hexToInt(int c)
{
    return c >= '0' && c <= '9' ? c - '0' : c - 'a' + 10;
}

std::string hexToChar(const std::string& m)
{
   std::string out = "";
   int c;       
   char s[4];
   for (int i = 0; i < m.length() - 1; i += 2)
   {
        c = hexToInt(m[i]);        
        c *= 16;
        c += hexToInt(m[i + 1]); 
        sprintf(s, "%c", c);
        out += s;
   }
   return out;
}

std::string xor(const std::string& m1, const std::string& m2)
{
   std::string out = "";
   char c[4];
   int i;
   for (i = 0; i < m1.length() && i < m2.length(); ++i)
   {
       sprintf(c, "%c", m1[i] ^ m2[i]);
       out += c;
   }
   if (i < m1.length())
   {
       out += m1.substr(i);
   }
   if (i < m2.length())
   {
       out += m2.substr(i);
   }    
   return out;
}
   
void getKey(const std::string& m1, const std::string& m2, const std::string& m, std::map<int, std::set<char> >& out)
{
   int len = m1.length() < m2.length() ? m1.length() : m2.length();
   int c1, c2 = ' ', c = '0';
   for (int i = 0; i < len; ++i)
   {
       if ((m[i] >= 'A' && m[i] <= 'Z') || (m[i] >= 'a' && m[i] <= 'z'))
       {
            c1 = (m[i] >= 'A' && m[i] <= 'Z') ? tolower(m[i]) : toupper(m[i]);
			c = '0';
            if (m1[i] ^ c1 == m2[i] ^ c2)
            {
                 c =  m1[i] ^ c1;    
            } 
            else if (m2[i] ^ c1 == m1[i] ^ c2)
            {
                 c =  m2[i] ^ c1;    
            }
			if (c != '0')
			{
				std::map<int, std::set<char> >::iterator it;
				std::set<char> vc;
				if ((it = out.find(i)) != out.end())
				{
					vc = it->second;
				}
				vc.insert(c);
				out[i] = vc;
			}
       }
   }
 }
   
void verify(std::map<int, std::set<char> >& m, std::string ct[], int n)
{
	for (int i = 0; i < ct[n - 1].length(); ++i)
	{
		std::map<int, std::set<char> >::iterator it = m.find(i);
		std::set<char> vc = it->second;
		for (std::set<char>::iterator it = vc.begin(); it != vc.end(); )
		{
			bool found = true;
			for (int j = 0; j < n - 1; ++j)
			{
				int c = ct[j][i] ^ (*it);
				if (!isalpha(c) && c != ' ' && c != '.') // && !(c >= '0' && c <= '9'))
				{
					found = false;
					break;
				}
			}
			if (!found)
			{
				vc.erase(it++);
			}
			else
			{
				++it;
			}
		}
		if (!vc.size())
		{
			for (int k = 0; k < 127; ++k)
			{
				bool found = true;
				for (int j = 0; j < n - 1; ++j)
				{
					int c = ct[j][i] ^ k;
					if (!isalpha(c) && c != ' ' && c != '.' && c != ',') // && !(c >= '0' && c <= '9'))
					{
						found = false;
						break;
					}
				}
				if (found)
				{
					vc.insert(k);
				}
			}
		}
		m[i] = vc;
	}
}

void decrypt(std::map<int, std::set<char> >& m, std::string ct[], int n)
{
	for (int j = 0; j < n; ++j)
	{
		for (int i = 0; i < ct[n - 1].length(); ++i)
		{
			std::map<int, std::set<char> >::iterator it = m.find(i);
			std::set<char> vc = it->second;
			int c = vc.size() ? ct[j][i] ^ (*(vc.begin())) : ' ';
			printf("%c", c);
		}
		std::cout << std::endl;
	}
}

void rverify(std::map<int, std::set<char> >& m, std::string ct[], int n)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < ct[i].length(); ++j)
		{
			//std::string s = "(";
			int c = ' ';
			//char ss[4];
			std::map<int, std::set<char> >::iterator it = m.find(j);
			if (it != m.end())
			{
				std::set<char> vc = it->second;
				for (std::set<char>::const_iterator it = vc.begin(); it != vc.end(); ++it)
				{
					c = ct[i][j] ^ (*it);
					if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == ' ') // || c == ',' || c == '.')
					{
						break;
					}
					//sprintf(ss, "%c", c);
					//s += ss;
				}
			}
			if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == ' ')
			{
			}
			else
			{
				c = ' ';
			}
			//s += ")";
			//std::cout << s;
			printf("%c", c);
		}
		std::cout << std::endl;
	}
}

void ex1_1()
{
     std::string ct[] = 
     {
      //ciphertext #1:
      "315c4eeaa8b5f8aaf9174145bf43e1784b8fa00dc71d885a804e5ee9fa40b16349c146fb778"
      "cdf2d3aff021dfff5b403b510d0d0455468aeb98622b137dae857553ccd8883a7bc37520e06e"
      "515d22c954eba5025b8cc57ee59418ce7dc6bc41556bdb36bbca3e8774301fbcaa3b83b22080"
      "9560987815f65286764703de0f3d524400a19b159610b11ef3e",

      //ciphertext #2:
      "234c02ecbbfbafa3ed18510abd11fa724fcda2018a1a8342cf064bbde548b12b07df44ba719"
      "1d9606ef4081ffde5ad46a5069d9f7f543bedb9c861bf29c7e205132eda9382b0bc2c5c4b45f9"
      "19cf3a9f1cb74151f6d551f4480c82b2cb24cc5b028aa76eb7b4ab24171ab3cdadb8356f",

      //ciphertext #3:
      "32510ba9a7b2bba9b8005d43a304b5714cc0bb0c8a34884dd91304b8ad40b62b07df44ba6e9d8a"
      "2368e51d04e0e7b207b70b9b8261112bacb6c866a232dfe257527dc29398f5f3251a0d47e503c66"
      "e935de81230b59b7afb5f41afa8d661cb",

      //ciphertext #4:
      "32510ba9aab2a8a4fd06414fb517b5605cc0aa0dc91a8908c2064ba8ad5ea06a029056f47a8ad33"
      "06ef5021eafe1ac01a81197847a5c68a1b78769a37bc8f4575432c198ccb4ef63590256e305cd3a9"
      "544ee4160ead45aef520489e7da7d835402bca670bda8eb775200b8dabbba246b130f040d8ec6447e"
      "2c767f3d30ed81ea2e4c1404e1315a1010e7229be6636aaa",

      //ciphertext #5:
      "3f561ba9adb4b6ebec54424ba317b564418fac0dd35f8c08d31a1fe9e24fe56808c213f17c81d96"
      "07cee021dafe1e001b21ade877a5e68bea88d61b93ac5ee0d562e8e9582f5ef375f0a4ae20ed86e"
      "935de81230b59b73fb4302cd95d770c65b40aaa065f2a5e33a5a0bb5dcaba43722130f042f8ec85b7c2070",

      //ciphertext #6:
      "32510bfbacfbb9befd54415da243e1695ecabd58c519cd4bd2061bbde24eb76a19d84aba34d8de28"
      "7be84d07e7e9a30ee714979c7e1123a8bd9822a33ecaf512472e8e8f8db3f9635c1949e640c621854"
      "eba0d79eccf52ff111284b4cc61d11902aebc66f2b2e436434eacc0aba938220b084800c2ca4e69352"
      "2643573b2c4ce35050b0cf774201f0fe52ac9f26d71b6cf61a711cc229f77ace7aa88a2f19983122b11be87a59c355d25f8e4",

      //ciphertext #7:
      "32510bfbacfbb9befd54415da243e1695ecabd58c519cd4bd90f1fa6ea5ba47b01c909ba7696cf606ef4"
      "0c04afe1ac0aa8148dd066592ded9f8774b529c7ea125d298e8883f5e9305f4b44f915cb2bd05af51373f"
      "d9b4af511039fa2d96f83414aaaf261bda2e97b170fb5cce2a53e675c154c0d9681596934777e2275b381c"
      "e2e40582afe67650b13e72287ff2270abcf73bb028932836fbdecfecee0a3b894473c1bbeb6b4913a536ce4"
      "f9b13f1efff71ea313c8661dd9a4ce",

      //ciphertext #8:
      "315c4eeaa8b5f8bffd11155ea506b56041c6a00c8a08854dd21a4bbde54ce56801d943ba708b8a3574f40"
      "c00fff9e00fa1439fd0654327a3bfc860b92f89ee04132ecb9298f5fd2d5e4b45e40ecc3b9d59e9417df7"
      "c95bba410e9aa2ca24c5474da2f276baa3ac325918b2daada43d6712150441c2e04f6565517f317da9d3",

      //ciphertext #9:
      "271946f9bbb2aeadec111841a81abc300ecaa01bd8069d5cc91005e9fe4aad6e04d513e96d99de2569bc5e"
      "50eeeca709b50a8a987f4264edb6896fb537d0a716132ddc938fb0f836480e06ed0fcd6e9759f40462f9cf57"
      "f4564186a2c1778f1543efa270bda5e933421cbe88a4a52222190f471e9bd15f652b653b7071aec59a2705081"
      "ffe72651d08f822c9ed6d76e48b63ab15d0208573a7eef027",

      //ciphertext #10:
      "466d06ece998b7a2fb1d464fed2ced7641ddaa3cc31c9941cf110abbf409ed39598005b3399ccfafb61d0315"
      "fca0a314be138a9f32503bedac8067f03adbf3575c3b8edc9ba7f537530541ab0f9f3cd04ff50d66f1d559ba520e89a2cb2a83",

      //target ciphertext (decrypt this one): 
      "32510ba9babebbbefd001547a810e67149caee11d945cd7fc81a05e9f85aac650e9052ba6a8cd8257bf14d13e6f0a803b54fde9e"
      "77472dbff89d71b57bddef121336cb85ccb8f3315f4b52e301d16e9f52f904"
      };
      
      for (int i = 0; i < 11; ++i)
	  {
		ct[i] = hexToChar(ct[i]);
		//std::cout << ct[i].length() << std::endl;
	  }
	  
      std::string m1, m2, m;
	  std::map<int, std::set<char> > out;
	  for (int i = 0; i < 10; ++i)
      {
	    m1 = ct[i];
		for (int j = 0; j < 10; ++j)
		{
		  if (i == j)
		  {
			continue;
		  }
		  m2 = ct[j];
          m = xor(m1, m2);
          getKey(m1, m2, m, out);
		} 
      }
	  
	  int i = 0;
      for (std::map<int, std::set<char> >::iterator it = out.begin(); i < ct[10].length(); ++it, ++i)
	  {
			std::cout << it->first << ": ";
			std::copy(it->second.begin(), it->second.end(), std::ostream_iterator<char>(std::cout, " "));
			std::cout << std::endl;
	  }
	  
	  verify(out, ct, 11);
      
	  i = 0;
      for (std::map<int, std::set<char> >::iterator it = out.begin(); i < ct[10].length(); ++it, ++i)
	  {
			std::cout << it->first << ": ";
			std::copy(it->second.begin(), it->second.end(), std::ostream_iterator<char>(std::cout, " "));
			std::cout << std::endl;
	  }
	  
	  decrypt(out, ct, 11);
	  
	  /*std::string s = ct[0];
      for (int i = 1; i <= 10; ++i)
      {
          std::cout << s << std:: endl;
          //std::cout << hexToChar(s) << std:: endl;
          s = xor(s, ct[i]);
      }*/
}

int main()
{
    ex1_1();
    return 0;
}
