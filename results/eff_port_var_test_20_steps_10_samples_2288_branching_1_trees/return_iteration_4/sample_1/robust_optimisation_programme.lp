\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.17714 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.96557 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.852703 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.15855 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.807688 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.01307 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.29974 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.27908 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.728222 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.916236 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.33077 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.03657 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.04312 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.1547 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.95423 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.22896 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.700456 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.956504 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.98354 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.842474 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.723373 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.882066 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.967799 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.42339 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.08176 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.722002 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.45948 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.789513 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.840398 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.15982 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.884961 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.06885 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.06765 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.779429 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.14882 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.966155 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.03497 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.5761 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24556 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.896745 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.899146 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.18006 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.685961 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.0832 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.33238 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.51678 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.710997 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.820958 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.897867 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.934963 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.22093 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.856446 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.07765 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.18226 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.630177 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.00212 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.47296 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.896069 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.742711 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.35197 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.31495 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.80315 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.01237 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.884181 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.5517 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.698309 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.11994 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.09507 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.736012 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.04078 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.851327 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.701764 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.30483 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.0009 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.17298 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.84176 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31763 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.845962 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.83576 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.25888 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.641496 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.14955 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.19216 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.71033 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.646994 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.01454 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.16592 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.819443 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.19093 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.85733 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.1077 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.23777 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.679974 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.907748 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.79607 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.851844 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.678596 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.48055 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.22419 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.7676 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.33686 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.842082 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.61751 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.785526 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.10235 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.940892 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.710358 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.824175 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.12514 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.627257 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.27326 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.13816 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.813291 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.88159 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20346 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.907598 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.918046 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.11782 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.777463 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0905 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.2701 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.49523 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.737538 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.938848 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.16548 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.908856 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.08633 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.995722 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.058 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.26783 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.808138 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.804186 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.42195 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.944269 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.698348 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.161 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.06901 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.31635 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.10504 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.76468 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.43407 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.852653 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.911989 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.07167 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.848407 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.04684 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.04503 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.710978 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.23172 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.906287 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.858177 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.43022 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15207 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.96572 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.762659 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.25187 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.780971 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.09544 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.899316 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.14405 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.946983 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.846414 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.72259 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.12031 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.446 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.7279 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.15169 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.795584 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.599346 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.67095 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.21666 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.839288 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.02963 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.86712 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.18962 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.59814 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.913455 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.10369 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.918906 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.58362 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.64893 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.737648 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.65753 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.627311 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.781717 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.966104 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.901441 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.47548 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.993054 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.72735 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.084 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.898857 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.894383 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.17027 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.872053 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.04912 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.35006 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.41343 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.946698 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.02799 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.16446 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.04802 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.36819 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.0763 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.00573 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.00518 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.906332 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.881125 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.23604 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.910674 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.752997 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.22819 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.01255 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.2796 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.06024 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.763708 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.41037 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.868389 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.945441 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.987571 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.842423 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.878401 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.08977 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.867373 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.23762 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.01675 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.868489 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.3457 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.16907 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.927331 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.844085 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.2277 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.750675 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.05483 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.0271 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.34847 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.837391 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.938924 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.970155 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.9446 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.30119 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.771193 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.11331 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.852651 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.637456 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.12578 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.42867 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.811795 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.903627 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.57895 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.28565 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.44169 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.05896 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.921934 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.25752 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.827063 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.2216 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.87522 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.700089 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.770429 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.00989 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.884549 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.08723 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.23147 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.962385 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.62612 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.21461 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.902437 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.891638 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.16519 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.802322 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.28304 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.821658 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.01574 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.731451 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.806836 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.675308 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.03991 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.15855 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.662042 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.42794 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.871712 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.694678 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.83025 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.74063 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.04392 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.008 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 2.322 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.35357 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.70322 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.668093 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.0232 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.14157 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.720293 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.49439 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.819361 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.443266 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.627323 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.540522 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.774816 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.05172 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.34244 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.96994 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.85754 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13669 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.897768 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.910242 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.16825 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.780008 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.10836 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.20264 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.36673 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.886917 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.02831 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.21941 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.946416 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.13066 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.05891 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.999421 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.14752 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.804223 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.757718 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.40922 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.859602 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.741108 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.22095 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.0427 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.36008 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.14576 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.909873 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.41941 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.91906 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.01452 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.12702 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.843749 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.894252 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.10983 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.808678 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.17995 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.963453 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.79964 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.40796 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09843 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.949349 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.903435 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.11313 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.839577 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.06069 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.17889 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.29922 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.884453 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.08786 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.01163 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.08087 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.20492 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.961449 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.08323 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.09278 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.843344 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.05321 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.18089 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.00284 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.813917 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.25698 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.11416 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.21249 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.02638 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.833868 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.22982 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.839622 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.08006 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.923641 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.78134 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.97366 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.916027 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.853463 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.17558 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.02776 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.953612 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.32091 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.213999180728954 z_k1_s0000
                                 - 0.153999865610884 z_k1_s0001
                                 - 0.0969999052945311 z_k1_s0002
                                 - 0.428999982122701 z_k1_s0003
                                 - 0.353998902102693 z_k1_s0004
                                 - 0.173999752050088 z_k1_s0005
                                 - 0.15799980847877 z_k1_s0006
                                 - 0.140999793140596 z_k1_s0007
                                 - 0.432000047862193 z_k1_s0010
                                 - 0.191000411482748 z_k1_s0011
                                 - 0.259001565801768 z_k1_s0012
                                 - 0.458001674203702 z_k1_s0013
                                 - 0.231000388281055 z_k1_s0014
                                 - 0.372001077466314 z_k1_s0015
                                 - 0.216000272631492 z_k1_s0016
                                 - 0.328001003508018 z_k1_s0017
                                 - 0.42000000664298 z_k1_s0020
                                 - 0.864999655181335 z_k1_s0021
                                 - 0.773001374526405 z_k1_s0022
                                 - 0.60900051153238 z_k1_s0023
                                 - 0.679999293755219 z_k1_s0024
                                 - 0.389000149552682 z_k1_s0025
                                 - 0.630000009964471 z_k1_s0026
                                 - 0.653002089628494 z_k1_s0027
                                 - 0.208999152672729 z_k1_s0030
                                 - 0.332998995597849 z_k1_s0031
                                 - 0.260000544519588 z_k1_s0032
                                 - 0.146999773009657 z_k1_s0033
                                 - 0.339999044988728 z_k1_s0034
                                 - 0.22599901462225 z_k1_s0035
                                 - 0.115999812278377 z_k1_s0036
                                 - 0.174999789973344 z_k1_s0037
                                 - 0.0759998854896651 z_k1_s0040
                                 - 0.154000367128144 z_k1_s0041
                                 - 0.302999243884253 z_k1_s0042
                                 - 0.0819999225387789 z_k1_s0043
                                 - 0.241999325917564 z_k1_s0044
                                 - 0.0859998638383153 z_k1_s0045
                                 - 0.16000015397764 z_k1_s0046
                                 - 0.147999329280557 z_k1_s0047
                                 - 0.171000053182688 z_k1_s0050
                                 - 0.237000020409299 z_k1_s0051
                                 - 0.2620012650188 z_k1_s0052
                                 - 0.475000277563118 z_k1_s0053
                                 - 0.392999560526461 z_k1_s0054
                                 - 0.334999851354785 z_k1_s0055
                                 - 0.268000348484176 z_k1_s0056
                                 - 0.196000129291319 z_k1_s0057
                                 - 0.452998810688058 z_k1_s0060
                                 - 0.463998352419128 z_k1_s0061
                                 - 0.404999770228448 z_k1_s0062
                                 - 0.285999145573357 z_k1_s0063
                                 - 0.74799934961769 z_k1_s0064
                                 - 0.377998641350117 z_k1_s0065
                                 - 0.216999142341976 z_k1_s0066
                                 - 0.86299871933864 z_k1_s0067
                                 - 0.114999896408734 z_k1_s0070
                                 - 0.327000225029341 z_k1_s0071
                                 - 0.300999300569532 z_k1_s0072
                                 - 0.21500016212019 z_k1_s0073
                                 - 0.0959999215794537 z_k1_s0074
                                 - 0.101999897378196 z_k1_s0075
                                 - 0.275998886742143 z_k1_s0076
                                 - 0.111999908509363 z_k1_s0077
                                 - 0.308999612840825 z_k1_s0100
                                 - 0.411999006721722 z_k1_s0101
                                 - 0.650998361048248 z_k1_s0102
                                 - 0.132999811358054 z_k1_s0103
                                 - 0.298999473681772 z_k1_s0104
                                 - 0.524000994662546 z_k1_s0105
                                 - 0.639999603391471 z_k1_s0106
                                 - 0.609998112515714 z_k1_s0107
                                 - 0.411998132379018 z_k1_s0110
                                 - 0.0879996927631323 z_k1_s0111
                                 - 0.250999491230951 z_k1_s0112
                                 - 0.286998625868671 z_k1_s0113
                                 - 0.197999258492238 z_k1_s0114
                                 - 0.139999447291589 z_k1_s0115
                                 - 0.335997950262553 z_k1_s0116
                                 - 0.296999389986454 z_k1_s0117
                                 - 0.12800007749559 z_k1_s0120
                                 - 0.130000091006463 z_k1_s0121
                                 - 0.117000003185789 z_k1_s0122
                                 - 0.0959999925216691 z_k1_s0123
                                 - 0.225000207972742 z_k1_s0124
                                 - 0.230000241749924 z_k1_s0125
                                 - 0.264000733834858 z_k1_s0126
                                 - 0.122000036962971 z_k1_s0127
                                 - 0.162999817804578 z_k1_s0130
                                 - 0.118000207674933 z_k1_s0131
                                 - 0.0840001332431535 z_k1_s0132
                                 - 0.265000164100082 z_k1_s0133
                                 - 0.142999991080292 z_k1_s0134
                                 - 0.227000493323938 z_k1_s0135
                                 - 0.0900000812604394 z_k1_s0136
                                 - 0.140000017071649 z_k1_s0137
                                 - 0.223000411317698 z_k1_s0140
                                 - 0.342000264127894 z_k1_s0141
                                 - 0.433001581573302 z_k1_s0142
                                 - 0.356000636812078 z_k1_s0143
                                 - 0.148000182798147 z_k1_s0144
                                 - 0.161000307861674 z_k1_s0145
                                 - 0.344000317368492 z_k1_s0146
                                 - 0.203000320912437 z_k1_s0147
                                 - 0.396999379806696 z_k1_s0150
                                 - 0.686000137704442 z_k1_s0151
                                 - 0.529000482246732 z_k1_s0152
                                 - 0.530000137727874 z_k1_s0153
                                 - 0.392001102400983 z_k1_s0154
                                 - 0.17299987594728 z_k1_s0155
                                 - 0.315999696776073 z_k1_s0156
                                 - 0.560000551145817 z_k1_s0157
                                 - 0.242000906070345 z_k1_s0160
                                 - 0.507002331742448 z_k1_s0161
                                 - 0.662003268964169 z_k1_s0162
                                 - 0.233000647795583 z_k1_s0163
                                 - 0.515002842187503 z_k1_s0164
                                 - 0.452999939491407 z_k1_s0165
                                 - 0.457002301220108 z_k1_s0166
                                 - 1.1440013724331 z_k1_s0167
                                 - 0.2080005648905 z_k1_s0170
                                 - 0.2080005648905 z_k1_s0171
                                 - 0.0990001465553862 z_k1_s0172
                                 - 0.165000049758678 z_k1_s0173
                                 - 0.0670001404415875 z_k1_s0174
                                 - 0.137000175696806 z_k1_s0175
                                 - 0.122999655164973 z_k1_s0176
                                 - 0.160000614069891 z_k1_s0177
                                 - 0.146999341952408 z_k1_s1000
                                 - 0.285998508125946 z_k1_s1001
                                 - 0.0789996298248303 z_k1_s1002
                                 - 0.375998865644766 z_k1_s1003
                                 - 0.2479983211737 z_k1_s1004
                                 - 0.272999680479409 z_k1_s1005
                                 - 0.118999536723099 z_k1_s1006
                                 - 0.0919995914668001 z_k1_s1007
                                 - 0.0810002128579456 z_k1_s1010
                                 - 0.0529999728515695 z_k1_s1011
                                 - 0.084999844073322 z_k1_s1012
                                 - 0.0659998715510533 z_k1_s1013
                                 - 0.0439999561673502 z_k1_s1014
                                 - 0.0680000006586017 z_k1_s1015
                                 - 0.0869999731808704 z_k1_s1016
                                 - 0.142999826193902 z_k1_s1017
                                 - 0.148999729057121 z_k1_s1020
                                 - 0.100999895282872 z_k1_s1021
                                 - 0.188999725735417 z_k1_s1022
                                 - 0.249999218740729 z_k1_s1023
                                 - 0.153999753991862 z_k1_s1024
                                 - 0.467998683498379 z_k1_s1025
                                 - 0.391999521288109 z_k1_s1026
                                 - 0.324998578763682 z_k1_s1027
                                 - 0.139000233968039 z_k1_s1030
                                 - 0.257000358959435 z_k1_s1031
                                 - 0.199000478872576 z_k1_s1032
                                 - 0.637002220155579 z_k1_s1033
                                 - 0.245000496058976 z_k1_s1034
                                 - 0.498000125383058 z_k1_s1035
                                 - 0.175000520471097 z_k1_s1036
                                 - 0.176000470279376 z_k1_s1037
                                 - 0.0310000106692675 z_k1_s1040
                                 - 0.0400000340635795 z_k1_s1041
                                 - 0.128999782384906 z_k1_s1042
                                 - 0.0380000352204017 z_k1_s1043
                                 - 0.0839999514134663 z_k1_s1044
                                 - 0.0849998078349949 z_k1_s1045
                                 - 0.0999999421588885 z_k1_s1046
                                 - 0.0649998194032172 z_k1_s1047
                                 - 0.229999477502117 z_k1_s1050
                                 - 0.227999438767408 z_k1_s1051
                                 - 0.117999678353374 z_k1_s1052
                                 - 0.46799816194509 z_k1_s1053
                                 - 0.583999934559207 z_k1_s1054
                                 - 0.176999517530061 z_k1_s1055
                                 - 0.348998345724615 z_k1_s1056
                                 - 0.21599944335865 z_k1_s1057
                                 - 0.414999497459883 z_k1_s1060
                                 - 0.0849999422941839 z_k1_s1061
                                 - 0.122999973536278 z_k1_s1062
                                 - 0.101999993313008 z_k1_s1063
                                 - 0.170999905989472 z_k1_s1064
                                 - 0.128999945542931 z_k1_s1065
                                 - 0.165999798983953 z_k1_s1066
                                 - 0.372999849813281 z_k1_s1067
                                 - 0.347999782783804 z_k1_s1070
                                 - 0.315999721171726 z_k1_s1071
                                 - 0.353999942210661 z_k1_s1072
                                 - 0.285998924037445 z_k1_s1073
                                 - 0.235999330542183 z_k1_s1074
                                 - 0.242999358807284 z_k1_s1075
                                 - 0.260999837087853 z_k1_s1076
                                 - 0.321999880598582 z_k1_s1077
                                 - 0.61999971690875 z_k1_s1100
                                 - 0.887999939598237 z_k1_s1101
                                 - 1.54000001570862 z_k1_s1102
                                 - 1.58099548966842 z_k1_s1103
                                 - 0.742999509784239 z_k1_s1104
                                 - 0.961000229758366 z_k1_s1105
                                 - 1.6229982453324 z_k1_s1106
                                 - 0.957996212640632 z_k1_s1107
                                 - 0.19900002341315 z_k1_s1110
                                 - 0.455998866503333 z_k1_s1111
                                 - 0.689000201951575 z_k1_s1112
                                 - 0.365999765969875 z_k1_s1113
                                 - 0.370000206526984 z_k1_s1114
                                 - 0.143000072514482 z_k1_s1115
                                 - 0.21000015374505 z_k1_s1116
                                 - 0.270000004600752 z_k1_s1117
                                 - 1.05399900070665 z_k1_s1120
                                 - 0.35299985011806 z_k1_s1121
                                 - 0.308000070789964 z_k1_s1122
                                 - 0.262999893961897 z_k1_s1123
                                 - 0.301999808712905 z_k1_s1124
                                 - 0.433998419408705 z_k1_s1125
                                 - 0.653001028971843 z_k1_s1126
                                 - 0.608000057143832 z_k1_s1127
                                 - 0.490000665034271 z_k1_s1130
                                 - 0.451000013658114 z_k1_s1131
                                 - 0.850998833747431 z_k1_s1132
                                 - 0.440000034998159 z_k1_s1133
                                 - 0.422999785241883 z_k1_s1134
                                 - 0.454998713389094 z_k1_s1135
                                 - 0.418000077678248 z_k1_s1136
                                 - 0.915000244441614 z_k1_s1137
                                 - 3.32599927170546 z_k1_s1140
                                 - 2.09699390440738 z_k1_s1141
                                 - 0.742999831812978 z_k1_s1142
                                 - 0.949999102702163 z_k1_s1143
                                 - 1.80700220627021 z_k1_s1144
                                 - 0.706999760058407 z_k1_s1145
                                 - 1.03599990839342 z_k1_s1146
                                 - 0.750999720902927 z_k1_s1147
                                 - 2.23100322538663 z_k1_s1150
                                 - 0.58300048718717 z_k1_s1151
                                 - 0.56900051178237 z_k1_s1152
                                 - 1.50799817514644 z_k1_s1153
                                 - 0.921001404790342 z_k1_s1154
                                 - 1.24700385487669 z_k1_s1155
                                 - 2.02200386735812 z_k1_s1156
                                 - 0.537000430599827 z_k1_s1157
                                 - 0.251000022721519 z_k1_s1160
                                 - 0.539000845890973 z_k1_s1161
                                 - 0.37900038857461 z_k1_s1162
                                 - 0.416000104622894 z_k1_s1163
                                 - 0.223999826943171 z_k1_s1164
                                 - 0.361999713066198 z_k1_s1165
                                 - 0.193999910633798 z_k1_s1166
                                 - 0.34599939623465 z_k1_s1167
                                 - 0.841000070304087 z_k1_s1170
                                 - 0.229999986678306 z_k1_s1171
                                 - 0.37600079776929 z_k1_s1172
                                 - 0.594999069157429 z_k1_s1173
                                 - 0.564001196653935 z_k1_s1174
                                 - 0.291999889682946 z_k1_s1175
                                 - 0.194000042998193 z_k1_s1176
                                 - 0.373000109463005 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.934113 w_k1_ADM_s000
                                 + 0.728583 w_k1_AXP_s000
                                 + 0.791307 w_k1_DIS_s000
                                 + 0.841796 w_k1_FDX_s000
                                 + 0.696648 w_k1_IBM_s000
                                 + 0.821976 w_k1_INTC_s000
                                 + 0.893181 w_k1_KO_s000
                                 + 0.871884 w_k1_MAT_s000
                                 + 0.913201 w_k1_MSFT_s000
                                 + 0.905894 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.963408 w_k1_ADM_s000
                                 + 1.04954 w_k1_AXP_s000
                                 + 1.18495 w_k1_DIS_s000
                                 + 1.02393 w_k1_FDX_s000
                                 + 1.50373 w_k1_IBM_s000
                                 + 1.04433 w_k1_INTC_s000
                                 + 1.24145 w_k1_KO_s000 + 1.44913 w_k1_MAT_s000
                                 + 1.20361 w_k1_MSFT_s000
                                 + 1.05181 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.778839 w_k1_ADM_s000
                                 + 0.673625 w_k1_AXP_s000
                                 + 1.02392 w_k1_DIS_s000
                                 + 0.848373 w_k1_FDX_s000
                                 + 1.11718 w_k1_IBM_s000
                                 + 0.739771 w_k1_INTC_s000
                                 + 1.14635 w_k1_KO_s000 + 1.0918 w_k1_MAT_s000
                                 + 0.873677 w_k1_MSFT_s000
                                 + 0.83985 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.08612 w_k1_ADM_s000
                                 + 0.920621 w_k1_AXP_s000
                                 + 0.947747 w_k1_DIS_s000
                                 + 0.980044 w_k1_FDX_s000
                                 + 0.856638 w_k1_IBM_s000
                                 + 0.901114 w_k1_INTC_s000
                                 + 0.945492 w_k1_KO_s000
                                 + 0.92613 w_k1_MAT_s000
                                 + 1.01294 w_k1_MSFT_s000
                                 + 1.01881 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.2587 w_k1_ADM_s000
                                 + 1.1008 w_k1_AXP_s000 + 1.52219 w_k1_DIS_s000
                                 + 1.28007 w_k1_FDX_s000
                                 + 1.04612 w_k1_IBM_s000
                                 + 1.24704 w_k1_INTC_s000
                                 + 1.05133 w_k1_KO_s000
                                 + 0.976862 w_k1_MAT_s000
                                 + 1.30613 w_k1_MSFT_s000
                                 + 1.09836 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.894429 w_k1_ADM_s000
                                 + 1.10782 w_k1_AXP_s000
                                 + 0.925774 w_k1_DIS_s000
                                 + 0.91104 w_k1_FDX_s000
                                 + 1.34215 w_k1_IBM_s000
                                 + 0.874809 w_k1_INTC_s000
                                 + 1.19065 w_k1_KO_s000 + 1.45841 w_k1_MAT_s000
                                 + 0.846886 w_k1_MSFT_s000
                                 + 1.05783 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.735721 w_k1_ADM_s000
                                 + 0.642832 w_k1_AXP_s000
                                 + 0.723565 w_k1_DIS_s000
                                 + 0.722087 w_k1_FDX_s000
                                 + 0.949281 w_k1_IBM_s000
                                 + 0.665339 w_k1_INTC_s000
                                 + 0.885073 w_k1_KO_s000
                                 + 1.16304 w_k1_MAT_s000
                                 + 0.764419 w_k1_MSFT_s000
                                 + 0.852306 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.15481 w_k1_ADM_s000
                                 + 1.63357 w_k1_AXP_s000
                                 + 1.13751 w_k1_DIS_s000
                                 + 1.12846 w_k1_FDX_s000 + 1.3325 w_k1_IBM_s000
                                 + 1.17502 w_k1_INTC_s000
                                 + 1.27423 w_k1_KO_s000 + 1.59528 w_k1_MAT_s000
                                 + 1.07444 w_k1_MSFT_s000
                                 + 0.990191 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.970598 w_k1_ADM_s001
                                 + 0.630306 w_k1_AXP_s001
                                 + 0.824052 w_k1_DIS_s001
                                 + 0.742846 w_k1_FDX_s001
                                 + 0.902171 w_k1_IBM_s001
                                 + 0.908288 w_k1_INTC_s001
                                 + 0.977256 w_k1_KO_s001
                                 + 0.985092 w_k1_MAT_s001
                                 + 0.869469 w_k1_MSFT_s001
                                 + 0.788593 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.47702 w_k1_ADM_s001
                                 + 1.28661 w_k1_AXP_s001 + 1.1854 w_k1_DIS_s001
                                 + 1.04156 w_k1_FDX_s001
                                 + 1.43001 w_k1_IBM_s001
                                 + 1.2264 w_k1_INTC_s001 + 1.29576 w_k1_KO_s001
                                 + 1.43416 w_k1_MAT_s001
                                 + 1.16589 w_k1_MSFT_s001
                                 + 1.16699 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.4066 w_k1_ADM_s001
                                 + 1.07053 w_k1_AXP_s001
                                 + 0.979441 w_k1_DIS_s001
                                 + 1.07772 w_k1_FDX_s001
                                 + 0.752909 w_k1_IBM_s001
                                 + 0.967922 w_k1_INTC_s001
                                 + 0.976167 w_k1_KO_s001
                                 + 0.722039 w_k1_MAT_s001
                                 + 1.06538 w_k1_MSFT_s001
                                 + 1.01975 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.47036 w_k1_ADM_s001
                                 + 1.20886 w_k1_AXP_s001
                                 + 1.30608 w_k1_DIS_s001
                                 + 1.28998 w_k1_FDX_s001
                                 + 0.920848 w_k1_IBM_s001
                                 + 1.10582 w_k1_INTC_s001
                                 + 1.05173 w_k1_KO_s001
                                 + 0.801167 w_k1_MAT_s001
                                 + 1.1325 w_k1_MSFT_s001
                                 + 0.985653 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.13045 w_k1_ADM_s001
                                 + 0.765003 w_k1_AXP_s001
                                 + 0.902132 w_k1_DIS_s001
                                 + 0.95917 w_k1_FDX_s001
                                 + 0.594457 w_k1_IBM_s001
                                 + 1.01561 w_k1_INTC_s001
                                 + 0.780019 w_k1_KO_s001
                                 + 0.47844 w_k1_MAT_s001
                                 + 1.01499 w_k1_MSFT_s001
                                 + 0.829159 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.859371 w_k1_ADM_s001
                                 + 0.912987 w_k1_AXP_s001
                                 + 0.744263 w_k1_DIS_s001
                                 + 0.865835 w_k1_FDX_s001
                                 + 0.886786 w_k1_IBM_s001
                                 + 0.864268 w_k1_INTC_s001
                                 + 0.843706 w_k1_KO_s001
                                 + 1.01926 w_k1_MAT_s001
                                 + 0.832917 w_k1_MSFT_s001
                                 + 0.936463 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.61024 w_k1_ADM_s001
                                 + 1.25418 w_k1_AXP_s001 + 1.6653 w_k1_DIS_s001
                                 + 1.24187 w_k1_FDX_s001
                                 + 1.73514 w_k1_IBM_s001
                                 + 1.18789 w_k1_INTC_s001 + 1.4579 w_k1_KO_s001
                                 + 1.34642 w_k1_MAT_s001
                                 + 1.41315 w_k1_MSFT_s001
                                 + 1.23131 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.939585 w_k1_ADM_s001
                                 + 1.08752 w_k1_AXP_s001
                                 + 1.06217 w_k1_DIS_s001
                                 + 0.943234 w_k1_FDX_s001
                                 + 1.38844 w_k1_IBM_s001
                                 + 1.0088 w_k1_INTC_s001 + 1.2237 w_k1_KO_s001
                                 + 1.65409 w_k1_MAT_s001
                                 + 1.07286 w_k1_MSFT_s001
                                 + 1.05266 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.386 w_k1_ADM_s002
                                 + 1.09447 w_k1_AXP_s002
                                 + 0.936638 w_k1_DIS_s002
                                 + 1.08426 w_k1_FDX_s002
                                 + 0.718015 w_k1_IBM_s002
                                 + 0.939796 w_k1_INTC_s002
                                 + 0.697646 w_k1_KO_s002
                                 + 0.849963 w_k1_MAT_s002
                                 + 0.903427 w_k1_MSFT_s002
                                 + 0.839762 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.43049 w_k1_ADM_s002
                                 + 1.16508 w_k1_AXP_s002
                                 + 1.35677 w_k1_DIS_s002
                                 + 1.12031 w_k1_FDX_s002
                                 + 1.67057 w_k1_IBM_s002
                                 + 0.77531 w_k1_INTC_s002
                                 + 1.31256 w_k1_KO_s002 + 1.9815 w_k1_MAT_s002
                                 + 0.952163 w_k1_MSFT_s002
                                 + 1.23883 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.15505 w_k1_ADM_s002
                                 + 1.17209 w_k1_AXP_s002
                                 + 1.10417 w_k1_DIS_s002
                                 + 1.12649 w_k1_FDX_s002 + 1.0385 w_k1_IBM_s002
                                 + 1.07089 w_k1_INTC_s002
                                 + 1.01688 w_k1_KO_s002
                                 + 0.881298 w_k1_MAT_s002
                                 + 1.039 w_k1_MSFT_s002 + 1.04681 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.11681 w_k1_ADM_s002
                                 + 0.798073 w_k1_AXP_s002
                                 + 1.12712 w_k1_DIS_s002
                                 + 0.965182 w_k1_FDX_s002
                                 + 1.13365 w_k1_IBM_s002
                                 + 0.83259 w_k1_INTC_s002
                                 + 1.08462 w_k1_KO_s002 + 1.07973 w_k1_MAT_s002
                                 + 0.978889 w_k1_MSFT_s002
                                 + 0.983045 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.431 w_k1_ADM_s002
                                 + 0.886365 w_k1_AXP_s002
                                 + 1.42507 w_k1_DIS_s002
                                 + 1.22283 w_k1_FDX_s002
                                 + 0.780703 w_k1_IBM_s002
                                 + 1.08914 w_k1_INTC_s002
                                 + 0.974578 w_k1_KO_s002
                                 + 0.57549 w_k1_MAT_s002
                                 + 1.19425 w_k1_MSFT_s002
                                 + 1.04704 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.19577 w_k1_ADM_s002
                                 + 1.02788 w_k1_AXP_s002
                                 + 0.690747 w_k1_DIS_s002
                                 + 0.907735 w_k1_FDX_s002
                                 + 1.12894 w_k1_IBM_s002
                                 + 0.727397 w_k1_INTC_s002
                                 + 0.783971 w_k1_KO_s002 + 1.2545 w_k1_MAT_s002
                                 + 0.747689 w_k1_MSFT_s002
                                 + 0.971264 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.934902 w_k1_ADM_s002
                                 + 0.687757 w_k1_AXP_s002
                                 + 0.81071 w_k1_DIS_s002
                                 + 0.766588 w_k1_FDX_s002
                                 + 0.971575 w_k1_IBM_s002
                                 + 0.905194 w_k1_INTC_s002
                                 + 0.894161 w_k1_KO_s002
                                 + 1.02221 w_k1_MAT_s002
                                 + 0.82683 w_k1_MSFT_s002
                                 + 0.955741 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.774874 w_k1_ADM_s002
                                 + 0.688271 w_k1_AXP_s002
                                 + 0.626819 w_k1_DIS_s002
                                 + 0.782201 w_k1_FDX_s002
                                 + 0.627866 w_k1_IBM_s002
                                 + 0.805806 w_k1_INTC_s002
                                 + 0.646266 w_k1_KO_s002
                                 + 0.367215 w_k1_MAT_s002
                                 + 0.677031 w_k1_MSFT_s002
                                 + 0.801413 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.15693 w_k1_ADM_s003
                                 + 1.13458 w_k1_AXP_s003
                                 + 0.699242 w_k1_DIS_s003
                                 + 0.95984 w_k1_FDX_s003
                                 + 0.801459 w_k1_IBM_s003
                                 + 0.76165 w_k1_INTC_s003
                                 + 0.767982 w_k1_KO_s003
                                 + 0.796773 w_k1_MAT_s003
                                 + 0.834389 w_k1_MSFT_s003
                                 + 0.810426 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.976172 w_k1_ADM_s003
                                 + 0.847278 w_k1_AXP_s003
                                 + 0.704738 w_k1_DIS_s003
                                 + 0.827407 w_k1_FDX_s003 + 1.017 w_k1_IBM_s003
                                 + 1.01026 w_k1_INTC_s003
                                 + 0.804973 w_k1_KO_s003
                                 + 0.67323 w_k1_MAT_s003
                                 + 0.84324 w_k1_MSFT_s003
                                 + 0.790703 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.059 w_k1_ADM_s003
                                 + 1.17104 w_k1_AXP_s003
                                 + 1.52729 w_k1_DIS_s003
                                 + 1.17405 w_k1_FDX_s003
                                 + 1.35548 w_k1_IBM_s003
                                 + 1.27094 w_k1_INTC_s003
                                 + 1.52349 w_k1_KO_s003 + 1.10356 w_k1_MAT_s003
                                 + 1.30503 w_k1_MSFT_s003
                                 + 1.23231 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.03152 w_k1_ADM_s003
                                 + 1.5539 w_k1_AXP_s003 + 1.11194 w_k1_DIS_s003
                                 + 1.12021 w_k1_FDX_s003
                                 + 1.25663 w_k1_IBM_s003
                                 + 1.07689 w_k1_INTC_s003
                                 + 1.02624 w_k1_KO_s003 + 1.88468 w_k1_MAT_s003
                                 + 0.948209 w_k1_MSFT_s003
                                 + 0.965601 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.22038 w_k1_ADM_s003
                                 + 0.913059 w_k1_AXP_s003
                                 + 0.978567 w_k1_DIS_s003
                                 + 1.04187 w_k1_FDX_s003
                                 + 0.828292 w_k1_IBM_s003
                                 + 1.18298 w_k1_INTC_s003
                                 + 1.05059 w_k1_KO_s003
                                 + 0.686098 w_k1_MAT_s003
                                 + 1.09917 w_k1_MSFT_s003
                                 + 1.01504 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.44249 w_k1_ADM_s003
                                 + 1.50683 w_k1_AXP_s003
                                 + 1.25275 w_k1_DIS_s003
                                 + 1.25227 w_k1_FDX_s003
                                 + 0.972553 w_k1_IBM_s003
                                 + 1.30001 w_k1_INTC_s003
                                 + 1.12597 w_k1_KO_s003
                                 + 0.951771 w_k1_MAT_s003
                                 + 1.24835 w_k1_MSFT_s003
                                 + 1.07988 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.923142 w_k1_ADM_s003
                                 + 0.945672 w_k1_AXP_s003
                                 + 1.13818 w_k1_DIS_s003
                                 + 0.956596 w_k1_FDX_s003
                                 + 1.23405 w_k1_IBM_s003
                                 + 1.03178 w_k1_INTC_s003
                                 + 1.13696 w_k1_KO_s003 + 1.04181 w_k1_MAT_s003
                                 + 1.09625 w_k1_MSFT_s003
                                 + 0.964955 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.847528 w_k1_ADM_s003
                                 + 0.679725 w_k1_AXP_s003
                                 + 0.591265 w_k1_DIS_s003
                                 + 0.828834 w_k1_FDX_s003
                                 + 0.556067 w_k1_IBM_s003
                                 + 0.83773 w_k1_INTC_s003
                                 + 0.675443 w_k1_KO_s003
                                 + 0.594445 w_k1_MAT_s003
                                 + 0.815122 w_k1_MSFT_s003
                                 + 0.851896 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.996793 w_k1_ADM_s004
                                 + 1.15112 w_k1_AXP_s004
                                 + 1.13641 w_k1_DIS_s004
                                 + 1.14066 w_k1_FDX_s004
                                 + 1.23006 w_k1_IBM_s004
                                 + 1.13879 w_k1_INTC_s004
                                 + 1.03505 w_k1_KO_s004
                                 + 0.943159 w_k1_MAT_s004
                                 + 0.981792 w_k1_MSFT_s004
                                 + 1.08253 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.939103 w_k1_ADM_s004
                                 + 1.47246 w_k1_AXP_s004
                                 + 1.42239 w_k1_DIS_s004 + 1.317 w_k1_FDX_s004
                                 + 0.997243 w_k1_IBM_s004
                                 + 1.19028 w_k1_INTC_s004 + 1.0719 w_k1_KO_s004
                                 + 1.13024 w_k1_MAT_s004
                                 + 1.24657 w_k1_MSFT_s004
                                 + 1.32441 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.705701 w_k1_ADM_s004
                                 + 0.788581 w_k1_AXP_s004
                                 + 0.621891 w_k1_DIS_s004
                                 + 0.840702 w_k1_FDX_s004
                                 + 0.774213 w_k1_IBM_s004
                                 + 0.931431 w_k1_INTC_s004
                                 + 0.808796 w_k1_KO_s004
                                 + 0.880359 w_k1_MAT_s004
                                 + 0.79934 w_k1_MSFT_s004
                                 + 0.912028 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.09532 w_k1_ADM_s004
                                 + 1.45225 w_k1_AXP_s004
                                 + 1.57888 w_k1_DIS_s004
                                 + 1.26073 w_k1_FDX_s004
                                 + 1.66459 w_k1_IBM_s004
                                 + 1.37034 w_k1_INTC_s004 + 1.3455 w_k1_KO_s004
                                 + 1.58569 w_k1_MAT_s004
                                 + 1.26901 w_k1_MSFT_s004
                                 + 1.37194 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.01707 w_k1_ADM_s004
                                 + 1.11001 w_k1_AXP_s004
                                 + 1.00814 w_k1_DIS_s004
                                 + 1.14588 w_k1_FDX_s004
                                 + 0.786191 w_k1_IBM_s004
                                 + 1.36292 w_k1_INTC_s004
                                 + 0.864444 w_k1_KO_s004
                                 + 0.783724 w_k1_MAT_s004
                                 + 1.13763 w_k1_MSFT_s004
                                 + 1.02502 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.940715 w_k1_ADM_s004
                                 + 1.0739 w_k1_AXP_s004 + 1.00431 w_k1_DIS_s004
                                 + 0.976822 w_k1_FDX_s004
                                 + 1.67859 w_k1_IBM_s004
                                 + 1.22586 w_k1_INTC_s004
                                 + 1.21764 w_k1_KO_s004 + 1.74059 w_k1_MAT_s004
                                 + 1.02272 w_k1_MSFT_s004
                                 + 1.14884 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.842844 w_k1_ADM_s004
                                 + 1.02378 w_k1_AXP_s004
                                 + 0.907742 w_k1_DIS_s004
                                 + 0.908185 w_k1_FDX_s004
                                 + 1.23652 w_k1_IBM_s004
                                 + 1.05575 w_k1_INTC_s004
                                 + 1.05083 w_k1_KO_s004 + 1.06918 w_k1_MAT_s004
                                 + 0.920309 w_k1_MSFT_s004
                                 + 0.95225 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.881789 w_k1_ADM_s004
                                 + 0.986275 w_k1_AXP_s004
                                 + 1.09384 w_k1_DIS_s004
                                 + 0.932276 w_k1_FDX_s004
                                 + 1.07886 w_k1_IBM_s004
                                 + 1.00185 w_k1_INTC_s004
                                 + 0.993909 w_k1_KO_s004
                                 + 0.751931 w_k1_MAT_s004
                                 + 0.931194 w_k1_MSFT_s004
                                 + 1.07348 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.698946 w_k1_ADM_s005
                                 + 0.535329 w_k1_AXP_s005
                                 + 0.647878 w_k1_DIS_s005
                                 + 0.734818 w_k1_FDX_s005
                                 + 0.729761 w_k1_IBM_s005
                                 + 0.749849 w_k1_INTC_s005
                                 + 0.734036 w_k1_KO_s005
                                 + 0.621031 w_k1_MAT_s005
                                 + 0.784387 w_k1_MSFT_s005
                                 + 0.776323 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.862559 w_k1_ADM_s005
                                 + 1.03191 w_k1_AXP_s005
                                 + 1.43026 w_k1_DIS_s005
                                 + 1.07004 w_k1_FDX_s005 + 1.4341 w_k1_IBM_s005
                                 + 1.16414 w_k1_INTC_s005
                                 + 1.27314 w_k1_KO_s005 + 1.63716 w_k1_MAT_s005
                                 + 1.13976 w_k1_MSFT_s005
                                 + 1.00258 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.08262 w_k1_ADM_s005
                                 + 1.08407 w_k1_AXP_s005
                                 + 1.33125 w_k1_DIS_s005
                                 + 1.21042 w_k1_FDX_s005
                                 + 0.810388 w_k1_IBM_s005
                                 + 0.93857 w_k1_INTC_s005
                                 + 1.04523 w_k1_KO_s005
                                 + 0.713078 w_k1_MAT_s005
                                 + 1.06189 w_k1_MSFT_s005
                                 + 0.897439 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.12275 w_k1_ADM_s005
                                 + 0.854783 w_k1_AXP_s005
                                 + 1.10632 w_k1_DIS_s005
                                 + 0.98773 w_k1_FDX_s005
                                 + 0.961273 w_k1_IBM_s005
                                 + 0.86111 w_k1_INTC_s005
                                 + 1.00643 w_k1_KO_s005
                                 + 0.728917 w_k1_MAT_s005
                                 + 1.02171 w_k1_MSFT_s005
                                 + 0.981555 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.921289 w_k1_ADM_s005
                                 + 0.845683 w_k1_AXP_s005
                                 + 0.84301 w_k1_DIS_s005
                                 + 0.886234 w_k1_FDX_s005
                                 + 0.695126 w_k1_IBM_s005
                                 + 0.798299 w_k1_INTC_s005
                                 + 0.802167 w_k1_KO_s005
                                 + 0.610488 w_k1_MAT_s005
                                 + 0.970042 w_k1_MSFT_s005
                                 + 0.761432 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.10657 w_k1_ADM_s005
                                 + 1.28115 w_k1_AXP_s005
                                 + 1.21413 w_k1_DIS_s005
                                 + 1.08825 w_k1_FDX_s005
                                 + 1.14965 w_k1_IBM_s005
                                 + 1.10104 w_k1_INTC_s005
                                 + 1.11677 w_k1_KO_s005 + 1.41709 w_k1_MAT_s005
                                 + 1.04168 w_k1_MSFT_s005
                                 + 1.02952 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.893593 w_k1_ADM_s005
                                 + 0.888245 w_k1_AXP_s005
                                 + 0.981048 w_k1_DIS_s005
                                 + 0.873058 w_k1_FDX_s005
                                 + 1.21908 w_k1_IBM_s005
                                 + 0.98555 w_k1_INTC_s005
                                 + 1.01226 w_k1_KO_s005 + 1.04218 w_k1_MAT_s005
                                 + 0.900351 w_k1_MSFT_s005
                                 + 0.954738 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.22927 w_k1_ADM_s005
                                 + 1.44407 w_k1_AXP_s005
                                 + 1.61357 w_k1_DIS_s005
                                 + 1.41687 w_k1_FDX_s005
                                 + 1.12034 w_k1_IBM_s005
                                 + 1.37246 w_k1_INTC_s005
                                 + 1.45915 w_k1_KO_s005 + 1.44728 w_k1_MAT_s005
                                 + 1.38887 w_k1_MSFT_s005
                                 + 1.15418 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.859911 w_k1_ADM_s006
                                 + 0.737828 w_k1_AXP_s006
                                 + 0.976297 w_k1_DIS_s006
                                 + 0.898805 w_k1_FDX_s006
                                 + 0.886101 w_k1_IBM_s006
                                 + 0.923583 w_k1_INTC_s006
                                 + 1.04054 w_k1_KO_s006
                                 + 0.844128 w_k1_MAT_s006
                                 + 0.96093 w_k1_MSFT_s006
                                 + 0.896907 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.53687 w_k1_ADM_s006
                                 + 1.20098 w_k1_AXP_s006
                                 + 1.11016 w_k1_DIS_s006
                                 + 1.16054 w_k1_FDX_s006
                                 + 0.835804 w_k1_IBM_s006
                                 + 1.01638 w_k1_INTC_s006
                                 + 0.867671 w_k1_KO_s006
                                 + 0.844232 w_k1_MAT_s006
                                 + 1.12909 w_k1_MSFT_s006
                                 + 0.969365 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.50011 w_k1_ADM_s006
                                 + 1.54646 w_k1_AXP_s006
                                 + 1.67403 w_k1_DIS_s006
                                 + 1.38977 w_k1_FDX_s006
                                 + 1.41471 w_k1_IBM_s006
                                 + 1.08946 w_k1_INTC_s006
                                 + 1.24424 w_k1_KO_s006 + 1.78151 w_k1_MAT_s006
                                 + 1.13033 w_k1_MSFT_s006
                                 + 1.14057 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.06885 w_k1_ADM_s006
                                 + 1.01445 w_k1_AXP_s006
                                 + 0.979909 w_k1_DIS_s006
                                 + 1.02694 w_k1_FDX_s006
                                 + 0.705948 w_k1_IBM_s006
                                 + 0.874054 w_k1_INTC_s006
                                 + 0.810905 w_k1_KO_s006
                                 + 0.691257 w_k1_MAT_s006
                                 + 1.10682 w_k1_MSFT_s006
                                 + 0.936996 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.09562 w_k1_ADM_s006
                                 + 1.11557 w_k1_AXP_s006
                                 + 0.944333 w_k1_DIS_s006
                                 + 0.999722 w_k1_FDX_s006
                                 + 1.12561 w_k1_IBM_s006
                                 + 0.787496 w_k1_INTC_s006
                                 + 0.901604 w_k1_KO_s006
                                 + 1.49891 w_k1_MAT_s006
                                 + 0.956705 w_k1_MSFT_s006
                                 + 0.948044 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.03192 w_k1_ADM_s006
                                 + 0.698711 w_k1_AXP_s006
                                 + 0.714488 w_k1_DIS_s006
                                 + 0.833379 w_k1_FDX_s006
                                 + 0.67519 w_k1_IBM_s006
                                 + 0.803803 w_k1_INTC_s006
                                 + 0.713261 w_k1_KO_s006
                                 + 0.621684 w_k1_MAT_s006
                                 + 0.844112 w_k1_MSFT_s006
                                 + 0.796234 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.793631 w_k1_ADM_s006
                                 + 0.597173 w_k1_AXP_s006
                                 + 0.634056 w_k1_DIS_s006
                                 + 0.687501 w_k1_FDX_s006
                                 + 0.997941 w_k1_IBM_s006
                                 + 0.659605 w_k1_INTC_s006
                                 + 0.743849 w_k1_KO_s006
                                 + 0.906003 w_k1_MAT_s006
                                 + 0.683455 w_k1_MSFT_s006
                                 + 0.77664 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.12777 w_k1_ADM_s006
                                 + 1.06509 w_k1_AXP_s006
                                 + 1.18739 w_k1_DIS_s006
                                 + 0.933576 w_k1_FDX_s006
                                 + 1.38335 w_k1_IBM_s006
                                 + 0.782043 w_k1_INTC_s006
                                 + 1.05758 w_k1_KO_s006 + 1.35821 w_k1_MAT_s006
                                 + 0.985667 w_k1_MSFT_s006
                                 + 1.01212 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.08923 w_k1_ADM_s007
                                 + 1.03049 w_k1_AXP_s007
                                 + 0.776548 w_k1_DIS_s007
                                 + 1.01394 w_k1_FDX_s007
                                 + 0.671693 w_k1_IBM_s007
                                 + 0.753589 w_k1_INTC_s007
                                 + 0.800654 w_k1_KO_s007
                                 + 0.686128 w_k1_MAT_s007
                                 + 0.8879 w_k1_MSFT_s007
                                 + 0.934661 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.959251 w_k1_ADM_s007
                                 + 1.19985 w_k1_AXP_s007
                                 + 1.06024 w_k1_DIS_s007
                                 + 1.09683 w_k1_FDX_s007
                                 + 1.05471 w_k1_IBM_s007
                                 + 0.896886 w_k1_INTC_s007
                                 + 1.04777 w_k1_KO_s007
                                 + 0.669217 w_k1_MAT_s007
                                 + 0.83177 w_k1_MSFT_s007
                                 + 0.997678 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.914489 w_k1_ADM_s007
                                 + 0.928515 w_k1_AXP_s007
                                 + 1.07385 w_k1_DIS_s007
                                 + 0.922927 w_k1_FDX_s007
                                 + 1.33375 w_k1_IBM_s007
                                 + 0.925915 w_k1_INTC_s007
                                 + 1.21664 w_k1_KO_s007
                                 + 0.837493 w_k1_MAT_s007
                                 + 0.847752 w_k1_MSFT_s007
                                 + 0.954411 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.43213 w_k1_ADM_s007
                                 + 1.46669 w_k1_AXP_s007
                                 + 1.67571 w_k1_DIS_s007
                                 + 1.34633 w_k1_FDX_s007
                                 + 1.22343 w_k1_IBM_s007
                                 + 1.20071 w_k1_INTC_s007
                                 + 1.43032 w_k1_KO_s007 + 0.97967 w_k1_MAT_s007
                                 + 1.13699 w_k1_MSFT_s007
                                 + 1.23357 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.931093 w_k1_ADM_s007
                                 + 0.996322 w_k1_AXP_s007
                                 + 0.656742 w_k1_DIS_s007
                                 + 0.86406 w_k1_FDX_s007
                                 + 0.951025 w_k1_IBM_s007
                                 + 0.868531 w_k1_INTC_s007
                                 + 1.00812 w_k1_KO_s007
                                 + 0.994641 w_k1_MAT_s007
                                 + 0.829656 w_k1_MSFT_s007
                                 + 0.929101 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.813008 w_k1_ADM_s007
                                 + 0.663363 w_k1_AXP_s007
                                 + 0.609476 w_k1_DIS_s007
                                 + 0.807764 w_k1_FDX_s007
                                 + 0.598079 w_k1_IBM_s007
                                 + 0.91297 w_k1_INTC_s007
                                 + 0.760237 w_k1_KO_s007
                                 + 0.623867 w_k1_MAT_s007
                                 + 0.801426 w_k1_MSFT_s007
                                 + 0.810811 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.3212 w_k1_ADM_s007
                                 + 0.954755 w_k1_AXP_s007
                                 + 1.16302 w_k1_DIS_s007
                                 + 1.15991 w_k1_FDX_s007
                                 + 0.830484 w_k1_IBM_s007
                                 + 1.14403 w_k1_INTC_s007
                                 + 1.07762 w_k1_KO_s007
                                 + 0.706805 w_k1_MAT_s007
                                 + 0.983605 w_k1_MSFT_s007
                                 + 0.906123 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.609403 w_k1_ADM_s007
                                 + 0.701224 w_k1_AXP_s007
                                 + 0.704219 w_k1_DIS_s007
                                 + 0.725943 w_k1_FDX_s007
                                 + 0.766763 w_k1_IBM_s007
                                 + 0.753676 w_k1_INTC_s007
                                 + 0.761886 w_k1_KO_s007
                                 + 0.628365 w_k1_MAT_s007
                                 + 0.689293 w_k1_MSFT_s007
                                 + 0.714483 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.844454 w_k1_ADM_s010
                                 + 0.862159 w_k1_AXP_s010
                                 + 0.764802 w_k1_DIS_s010
                                 + 0.931415 w_k1_FDX_s010
                                 + 0.648181 w_k1_IBM_s010
                                 + 1.15773 w_k1_INTC_s010
                                 + 0.715345 w_k1_KO_s010
                                 + 0.533688 w_k1_MAT_s010
                                 + 0.931882 w_k1_MSFT_s010
                                 + 0.862906 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.904674 w_k1_ADM_s010
                                 + 0.745267 w_k1_AXP_s010
                                 + 0.93768 w_k1_DIS_s010
                                 + 0.844351 w_k1_FDX_s010
                                 + 0.94385 w_k1_IBM_s010
                                 + 1.02976 w_k1_INTC_s010
                                 + 0.894519 w_k1_KO_s010
                                 + 0.741388 w_k1_MAT_s010
                                 + 0.996383 w_k1_MSFT_s010
                                 + 0.861371 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.13029 w_k1_ADM_s010
                                 + 1.02737 w_k1_AXP_s010
                                 + 0.983752 w_k1_DIS_s010
                                 + 0.970482 w_k1_FDX_s010
                                 + 1.27205 w_k1_IBM_s010
                                 + 0.858512 w_k1_INTC_s010
                                 + 1.09005 w_k1_KO_s010 + 1.42616 w_k1_MAT_s010
                                 + 0.962909 w_k1_MSFT_s010
                                 + 0.988009 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.25302 w_k1_ADM_s010
                                 + 1.65634 w_k1_AXP_s010 + 1.7031 w_k1_DIS_s010
                                 + 1.3697 w_k1_FDX_s010 + 1.5154 w_k1_IBM_s010
                                 + 1.55657 w_k1_INTC_s010
                                 + 1.46903 w_k1_KO_s010 + 1.35694 w_k1_MAT_s010
                                 + 1.45173 w_k1_MSFT_s010
                                 + 1.18447 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.07315 w_k1_ADM_s010
                                 + 1.44066 w_k1_AXP_s010 + 1.0257 w_k1_DIS_s010
                                 + 1.1039 w_k1_FDX_s010
                                 + 0.876786 w_k1_IBM_s010
                                 + 1.0058 w_k1_INTC_s010
                                 + 0.820681 w_k1_KO_s010
                                 + 1.03908 w_k1_MAT_s010
                                 + 0.939539 w_k1_MSFT_s010
                                 + 1.03868 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.879716 w_k1_ADM_s010
                                 + 0.82604 w_k1_AXP_s010
                                 + 0.64068 w_k1_DIS_s010
                                 + 0.805677 w_k1_FDX_s010
                                 + 0.858591 w_k1_IBM_s010
                                 + 1.02583 w_k1_INTC_s010
                                 + 0.755368 w_k1_KO_s010
                                 + 0.653156 w_k1_MAT_s010
                                 + 0.769905 w_k1_MSFT_s010
                                 + 0.83323 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.17605 w_k1_ADM_s010
                                 + 1.05252 w_k1_AXP_s010
                                 + 1.04152 w_k1_DIS_s010
                                 + 1.12832 w_k1_FDX_s010
                                 + 0.862906 w_k1_IBM_s010
                                 + 1.18182 w_k1_INTC_s010
                                 + 1.10373 w_k1_KO_s010
                                 + 0.785893 w_k1_MAT_s010
                                 + 1.14353 w_k1_MSFT_s010
                                 + 1.08884 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.19993 w_k1_ADM_s010
                                 + 1.45972 w_k1_AXP_s010
                                 + 1.26543 w_k1_DIS_s010
                                 + 1.21816 w_k1_FDX_s010
                                 + 1.31758 w_k1_IBM_s010
                                 + 1.21414 w_k1_INTC_s010
                                 + 1.26131 w_k1_KO_s010 + 1.1659 w_k1_MAT_s010
                                 + 1.23365 w_k1_MSFT_s010
                                 + 1.11143 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.08656 w_k1_ADM_s011
                                 + 1.14348 w_k1_AXP_s011
                                 + 1.17323 w_k1_DIS_s011
                                 + 1.10888 w_k1_FDX_s011
                                 + 0.926122 w_k1_IBM_s011
                                 + 1.13273 w_k1_INTC_s011
                                 + 1.05079 w_k1_KO_s011
                                 + 0.791498 w_k1_MAT_s011
                                 + 0.983882 w_k1_MSFT_s011
                                 + 0.940711 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.914777 w_k1_ADM_s011
                                 + 1.26497 w_k1_AXP_s011
                                 + 0.948159 w_k1_DIS_s011
                                 + 1.0848 w_k1_FDX_s011 + 1.18222 w_k1_IBM_s011
                                 + 0.991318 w_k1_INTC_s011
                                 + 1.12205 w_k1_KO_s011 + 1.29569 w_k1_MAT_s011
                                 + 0.932204 w_k1_MSFT_s011
                                 + 0.97177 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.024 w_k1_ADM_s011
                                 + 1.34687 w_k1_AXP_s011
                                 + 1.65511 w_k1_DIS_s011
                                 + 1.32264 w_k1_FDX_s011
                                 + 1.07571 w_k1_IBM_s011
                                 + 1.55307 w_k1_INTC_s011
                                 + 1.65414 w_k1_KO_s011 + 1.16715 w_k1_MAT_s011
                                 + 1.458 w_k1_MSFT_s011 + 1.16068 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.88427 w_k1_ADM_s011
                                 + 0.873609 w_k1_AXP_s011
                                 + 0.933417 w_k1_DIS_s011
                                 + 0.921958 w_k1_FDX_s011
                                 + 0.754819 w_k1_IBM_s011
                                 + 1.13115 w_k1_INTC_s011
                                 + 1.00465 w_k1_KO_s011
                                 + 0.766512 w_k1_MAT_s011
                                 + 0.882489 w_k1_MSFT_s011
                                 + 0.876351 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.905486 w_k1_ADM_s011
                                 + 0.771106 w_k1_AXP_s011
                                 + 1.05917 w_k1_DIS_s011
                                 + 0.86851 w_k1_FDX_s011
                                 + 1.14457 w_k1_IBM_s011
                                 + 0.978094 w_k1_INTC_s011
                                 + 1.35509 w_k1_KO_s011
                                 + 0.973009 w_k1_MAT_s011
                                 + 1.04697 w_k1_MSFT_s011
                                 + 0.990637 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.913451 w_k1_ADM_s011
                                 + 1.33972 w_k1_AXP_s011
                                 + 1.18963 w_k1_DIS_s011
                                 + 0.979912 w_k1_FDX_s011
                                 + 1.46418 w_k1_IBM_s011
                                 + 1.08636 w_k1_INTC_s011
                                 + 1.36235 w_k1_KO_s011 + 2.02014 w_k1_MAT_s011
                                 + 1.03912 w_k1_MSFT_s011
                                 + 1.08905 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.800777 w_k1_ADM_s011
                                 + 0.952735 w_k1_AXP_s011
                                 + 0.719903 w_k1_DIS_s011
                                 + 0.896405 w_k1_FDX_s011
                                 + 0.924216 w_k1_IBM_s011
                                 + 1.08612 w_k1_INTC_s011
                                 + 0.974792 w_k1_KO_s011
                                 + 1.47206 w_k1_MAT_s011
                                 + 0.897367 w_k1_MSFT_s011
                                 + 1.05685 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.834533 w_k1_ADM_s011
                                 + 0.677184 w_k1_AXP_s011
                                 + 0.817609 w_k1_DIS_s011
                                 + 0.770467 w_k1_FDX_s011
                                 + 0.729074 w_k1_IBM_s011
                                 + 0.851101 w_k1_INTC_s011
                                 + 1.02738 w_k1_KO_s011
                                 + 0.711043 w_k1_MAT_s011
                                 + 0.839915 w_k1_MSFT_s011
                                 + 0.849677 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.46715 w_k1_ADM_s012
                                 + 1.39362 w_k1_AXP_s012
                                 + 1.28388 w_k1_DIS_s012
                                 + 1.10119 w_k1_FDX_s012
                                 + 1.45847 w_k1_IBM_s012
                                 + 0.926807 w_k1_INTC_s012
                                 + 1.35831 w_k1_KO_s012 + 1.21867 w_k1_MAT_s012
                                 + 0.997863 w_k1_MSFT_s012
                                 + 1.13845 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.987305 w_k1_ADM_s012
                                 + 1.10385 w_k1_AXP_s012
                                 + 1.22138 w_k1_DIS_s012
                                 + 1.14324 w_k1_FDX_s012
                                 + 0.895416 w_k1_IBM_s012
                                 + 1.12087 w_k1_INTC_s012
                                 + 0.928104 w_k1_KO_s012
                                 + 0.750477 w_k1_MAT_s012
                                 + 1.09808 w_k1_MSFT_s012
                                 + 0.936679 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.974903 w_k1_ADM_s012
                                 + 0.866933 w_k1_AXP_s012
                                 + 0.83065 w_k1_DIS_s012
                                 + 0.899261 w_k1_FDX_s012
                                 + 1.10683 w_k1_IBM_s012
                                 + 0.821384 w_k1_INTC_s012
                                 + 1.10943 w_k1_KO_s012 + 1.14555 w_k1_MAT_s012
                                 + 0.977795 w_k1_MSFT_s012
                                 + 0.984582 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.03865 w_k1_ADM_s012
                                 + 1.1001 w_k1_AXP_s012 + 1.09533 w_k1_DIS_s012
                                 + 1.0439 w_k1_FDX_s012 + 1.12777 w_k1_IBM_s012
                                 + 0.896464 w_k1_INTC_s012
                                 + 0.989066 w_k1_KO_s012
                                 + 0.91419 w_k1_MAT_s012
                                 + 1.03705 w_k1_MSFT_s012
                                 + 0.955676 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.01769 w_k1_ADM_s012
                                 + 0.85288 w_k1_AXP_s012
                                 + 0.918358 w_k1_DIS_s012
                                 + 0.977925 w_k1_FDX_s012
                                 + 0.794309 w_k1_IBM_s012
                                 + 0.816158 w_k1_INTC_s012
                                 + 0.893125 w_k1_KO_s012
                                 + 0.784838 w_k1_MAT_s012
                                 + 0.970563 w_k1_MSFT_s012
                                 + 0.889143 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.44424 w_k1_ADM_s012
                                 + 1.22298 w_k1_AXP_s012
                                 + 1.57376 w_k1_DIS_s012
                                 + 1.36452 w_k1_FDX_s012
                                 + 1.03726 w_k1_IBM_s012
                                 + 1.26456 w_k1_INTC_s012
                                 + 1.33639 w_k1_KO_s012
                                 + 0.950361 w_k1_MAT_s012
                                 + 1.41869 w_k1_MSFT_s012
                                 + 1.13833 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.860773 w_k1_ADM_s012
                                 + 0.762872 w_k1_AXP_s012
                                 + 0.640926 w_k1_DIS_s012
                                 + 0.72566 w_k1_FDX_s012
                                 + 0.834747 w_k1_IBM_s012
                                 + 0.779596 w_k1_INTC_s012
                                 + 0.924453 w_k1_KO_s012
                                 + 0.750163 w_k1_MAT_s012
                                 + 0.853966 w_k1_MSFT_s012
                                 + 0.777736 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.24629 w_k1_ADM_s012
                                 + 1.37845 w_k1_AXP_s012
                                 + 0.803027 w_k1_DIS_s012
                                 + 0.874769 w_k1_FDX_s012
                                 + 1.31099 w_k1_IBM_s012
                                 + 0.961231 w_k1_INTC_s012
                                 + 0.936666 w_k1_KO_s012
                                 + 0.926114 w_k1_MAT_s012
                                 + 0.83593 w_k1_MSFT_s012
                                 + 1.01229 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.700173 w_k1_ADM_s013
                                 + 1.21136 w_k1_AXP_s013
                                 + 0.983273 w_k1_DIS_s013
                                 + 1.0198 w_k1_FDX_s013 + 1.39204 w_k1_IBM_s013
                                 + 1.30108 w_k1_INTC_s013
                                 + 1.19124 w_k1_KO_s013 + 1.31856 w_k1_MAT_s013
                                 + 1.13115 w_k1_MSFT_s013
                                 + 0.998102 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.594103 w_k1_ADM_s013
                                 + 0.528606 w_k1_AXP_s013
                                 + 0.663277 w_k1_DIS_s013
                                 + 0.795818 w_k1_FDX_s013
                                 + 0.642295 w_k1_IBM_s013
                                 + 0.931538 w_k1_INTC_s013
                                 + 0.728103 w_k1_KO_s013
                                 + 0.594861 w_k1_MAT_s013
                                 + 0.961964 w_k1_MSFT_s013
                                 + 0.770834 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.37294 w_k1_ADM_s013
                                 + 1.75442 w_k1_AXP_s013 + 1.6881 w_k1_DIS_s013
                                 + 1.40521 w_k1_FDX_s013
                                 + 1.50144 w_k1_IBM_s013
                                 + 1.24385 w_k1_INTC_s013
                                 + 1.57068 w_k1_KO_s013 + 1.85517 w_k1_MAT_s013
                                 + 1.31503 w_k1_MSFT_s013
                                 + 1.39958 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.82221 w_k1_ADM_s013
                                 + 0.906867 w_k1_AXP_s013
                                 + 0.75905 w_k1_DIS_s013
                                 + 0.861281 w_k1_FDX_s013
                                 + 0.858516 w_k1_IBM_s013
                                 + 0.917617 w_k1_INTC_s013
                                 + 0.854178 w_k1_KO_s013
                                 + 0.76158 w_k1_MAT_s013
                                 + 1.02818 w_k1_MSFT_s013
                                 + 0.867974 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.26162 w_k1_ADM_s013
                                 + 1.2968 w_k1_AXP_s013 + 1.26362 w_k1_DIS_s013
                                 + 1.3002 w_k1_FDX_s013 + 1.17953 w_k1_IBM_s013
                                 + 1.15041 w_k1_INTC_s013
                                 + 1.37362 w_k1_KO_s013 + 1.30731 w_k1_MAT_s013
                                 + 1.35521 w_k1_MSFT_s013
                                 + 1.22993 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.20182 w_k1_ADM_s013
                                 + 0.989123 w_k1_AXP_s013
                                 + 1.11026 w_k1_DIS_s013
                                 + 1.12553 w_k1_FDX_s013
                                 + 0.846386 w_k1_IBM_s013
                                 + 0.977741 w_k1_INTC_s013
                                 + 0.914151 w_k1_KO_s013
                                 + 0.820867 w_k1_MAT_s013
                                 + 1.17299 w_k1_MSFT_s013
                                 + 0.960421 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.715035 w_k1_ADM_s013
                                 + 0.953305 w_k1_AXP_s013
                                 + 0.699282 w_k1_DIS_s013
                                 + 0.875245 w_k1_FDX_s013
                                 + 1.06479 w_k1_IBM_s013
                                 + 1.04362 w_k1_INTC_s013
                                 + 0.919287 w_k1_KO_s013
                                 + 1.00609 w_k1_MAT_s013
                                 + 1.07322 w_k1_MSFT_s013
                                 + 0.902962 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.04547 w_k1_ADM_s013
                                 + 1.20539 w_k1_AXP_s013
                                 + 1.13126 w_k1_DIS_s013
                                 + 1.24874 w_k1_FDX_s013
                                 + 1.03859 w_k1_IBM_s013
                                 + 1.02157 w_k1_INTC_s013
                                 + 1.14772 w_k1_KO_s013 + 1.04634 w_k1_MAT_s013
                                 + 1.15075 w_k1_MSFT_s013
                                 + 1.05266 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.50438 w_k1_ADM_s014
                                 + 0.886048 w_k1_AXP_s014
                                 + 1.19974 w_k1_DIS_s014
                                 + 1.11395 w_k1_FDX_s014
                                 + 0.816864 w_k1_IBM_s014
                                 + 1.16916 w_k1_INTC_s014
                                 + 1.07929 w_k1_KO_s014
                                 + 0.548939 w_k1_MAT_s014
                                 + 1.09587 w_k1_MSFT_s014
                                 + 1.04566 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.47517 w_k1_ADM_s014
                                 + 1.36522 w_k1_AXP_s014
                                 + 1.08567 w_k1_DIS_s014
                                 + 1.09313 w_k1_FDX_s014
                                 + 1.31363 w_k1_IBM_s014
                                 + 1.18726 w_k1_INTC_s014
                                 + 1.09555 w_k1_KO_s014 + 1.06249 w_k1_MAT_s014
                                 + 1.01158 w_k1_MSFT_s014
                                 + 1.16075 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.31096 w_k1_ADM_s014
                                 + 0.797923 w_k1_AXP_s014
                                 + 0.924096 w_k1_DIS_s014
                                 + 0.912859 w_k1_FDX_s014
                                 + 0.980223 w_k1_IBM_s014
                                 + 0.867699 w_k1_INTC_s014
                                 + 0.965332 w_k1_KO_s014
                                 + 0.765871 w_k1_MAT_s014
                                 + 0.877186 w_k1_MSFT_s014
                                 + 1.13301 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.920065 w_k1_ADM_s014
                                 + 0.877011 w_k1_AXP_s014
                                 + 0.704468 w_k1_DIS_s014
                                 + 0.843309 w_k1_FDX_s014
                                 + 0.701192 w_k1_IBM_s014
                                 + 0.774671 w_k1_INTC_s014
                                 + 0.662331 w_k1_KO_s014
                                 + 0.568054 w_k1_MAT_s014
                                 + 0.738051 w_k1_MSFT_s014
                                 + 0.950783 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.37668 w_k1_ADM_s014
                                 + 1.33287 w_k1_AXP_s014
                                 + 1.60951 w_k1_DIS_s014
                                 + 1.22042 w_k1_FDX_s014
                                 + 1.25734 w_k1_IBM_s014
                                 + 1.34556 w_k1_INTC_s014
                                 + 1.36378 w_k1_KO_s014
                                 + 0.962074 w_k1_MAT_s014
                                 + 1.13624 w_k1_MSFT_s014
                                 + 1.33779 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 2.52016 w_k1_ADM_s014
                                 + 1.76455 w_k1_AXP_s014
                                 + 1.90476 w_k1_DIS_s014 + 1.6045 w_k1_FDX_s014
                                 + 1.54905 w_k1_IBM_s014
                                 + 1.33145 w_k1_INTC_s014
                                 + 1.45339 w_k1_KO_s014 + 1.57458 w_k1_MAT_s014
                                 + 1.3644 w_k1_MSFT_s014 + 1.5055 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.23503 w_k1_ADM_s014
                                 + 0.821319 w_k1_AXP_s014
                                 + 0.713261 w_k1_DIS_s014
                                 + 0.761783 w_k1_FDX_s014
                                 + 0.987551 w_k1_IBM_s014
                                 + 0.933087 w_k1_INTC_s014
                                 + 0.914146 w_k1_KO_s014
                                 + 0.841375 w_k1_MAT_s014
                                 + 0.859514 w_k1_MSFT_s014
                                 + 0.913429 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.81295 w_k1_ADM_s014
                                 + 1.24728 w_k1_AXP_s014
                                 + 1.17301 w_k1_DIS_s014
                                 + 1.28172 w_k1_FDX_s014
                                 + 0.688439 w_k1_IBM_s014
                                 + 1.42359 w_k1_INTC_s014
                                 + 0.956461 w_k1_KO_s014
                                 + 0.67697 w_k1_MAT_s014
                                 + 1.32761 w_k1_MSFT_s014
                                 + 1.17231 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.07609 w_k1_ADM_s015
                                 + 1.4048 w_k1_AXP_s015 + 1.4162 w_k1_DIS_s015
                                 + 1.30174 w_k1_FDX_s015
                                 + 1.07288 w_k1_IBM_s015
                                 + 1.38633 w_k1_INTC_s015
                                 + 1.29613 w_k1_KO_s015
                                 + 0.967933 w_k1_MAT_s015
                                 + 1.2548 w_k1_MSFT_s015 + 1.09362 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.95835 w_k1_ADM_s015
                                 + 1.2133 w_k1_AXP_s015
                                 + 0.932315 w_k1_DIS_s015
                                 + 0.969551 w_k1_FDX_s015
                                 + 0.956569 w_k1_IBM_s015
                                 + 1.00569 w_k1_INTC_s015
                                 + 1.15054 w_k1_KO_s015 + 1.25418 w_k1_MAT_s015
                                 + 1.1071 w_k1_MSFT_s015
                                 + 0.898134 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.953201 w_k1_ADM_s015
                                 + 0.80663 w_k1_AXP_s015
                                 + 0.764963 w_k1_DIS_s015
                                 + 0.781237 w_k1_FDX_s015
                                 + 0.883699 w_k1_IBM_s015
                                 + 0.743879 w_k1_INTC_s015
                                 + 0.938198 w_k1_KO_s015
                                 + 0.757089 w_k1_MAT_s015
                                 + 0.794869 w_k1_MSFT_s015
                                 + 0.854833 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.21776 w_k1_ADM_s015
                                 + 1.15506 w_k1_AXP_s015
                                 + 1.18292 w_k1_DIS_s015
                                 + 1.06174 w_k1_FDX_s015 + 1.1206 w_k1_IBM_s015
                                 + 1.11015 w_k1_INTC_s015 + 1.3771 w_k1_KO_s015
                                 + 0.83001 w_k1_MAT_s015
                                 + 1.1735 w_k1_MSFT_s015
                                 + 0.998112 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.93676 w_k1_ADM_s015
                                 + 1.24883 w_k1_AXP_s015
                                 + 1.13836 w_k1_DIS_s015
                                 + 0.975369 w_k1_FDX_s015
                                 + 1.60169 w_k1_IBM_s015
                                 + 1.15548 w_k1_INTC_s015
                                 + 1.35239 w_k1_KO_s015 + 1.64027 w_k1_MAT_s015
                                 + 1.09413 w_k1_MSFT_s015
                                 + 0.937423 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.941701 w_k1_ADM_s015
                                 + 1.20691 w_k1_AXP_s015
                                 + 0.687113 w_k1_DIS_s015
                                 + 0.900532 w_k1_FDX_s015
                                 + 1.04086 w_k1_IBM_s015
                                 + 0.915213 w_k1_INTC_s015
                                 + 0.929516 w_k1_KO_s015
                                 + 0.93797 w_k1_MAT_s015
                                 + 0.887141 w_k1_MSFT_s015
                                 + 0.958033 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.36983 w_k1_ADM_s015
                                 + 0.910555 w_k1_AXP_s015
                                 + 1.38774 w_k1_DIS_s015 + 1.2199 w_k1_FDX_s015
                                 + 0.863055 w_k1_IBM_s015
                                 + 1.31853 w_k1_INTC_s015
                                 + 1.26312 w_k1_KO_s015
                                 + 0.533518 w_k1_MAT_s015
                                 + 1.37519 w_k1_MSFT_s015
                                 + 1.08337 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.01649 w_k1_ADM_s015
                                 + 0.86501 w_k1_AXP_s015
                                 + 0.711587 w_k1_DIS_s015
                                 + 0.855602 w_k1_FDX_s015
                                 + 0.638308 w_k1_IBM_s015
                                 + 1.0176 w_k1_INTC_s015
                                 + 0.775161 w_k1_KO_s015
                                 + 0.451773 w_k1_MAT_s015
                                 + 0.90997 w_k1_MSFT_s015
                                 + 0.750286 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.38271 w_k1_ADM_s016
                                 + 1.27043 w_k1_AXP_s016
                                 + 1.59041 w_k1_DIS_s016
                                 + 1.40529 w_k1_FDX_s016
                                 + 0.948596 w_k1_IBM_s016
                                 + 1.20199 w_k1_INTC_s016 + 1.0557 w_k1_KO_s016
                                 + 0.920589 w_k1_MAT_s016
                                 + 1.29192 w_k1_MSFT_s016
                                 + 1.10767 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.786372 w_k1_ADM_s016
                                 + 0.876033 w_k1_AXP_s016
                                 + 0.935101 w_k1_DIS_s016
                                 + 0.94684 w_k1_FDX_s016
                                 + 0.916723 w_k1_IBM_s016
                                 + 0.922157 w_k1_INTC_s016
                                 + 0.840508 w_k1_KO_s016
                                 + 0.905576 w_k1_MAT_s016
                                 + 0.991028 w_k1_MSFT_s016
                                 + 1.00176 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.728622 w_k1_ADM_s016
                                 + 0.625862 w_k1_AXP_s016
                                 + 0.574339 w_k1_DIS_s016
                                 + 0.762069 w_k1_FDX_s016
                                 + 0.65608 w_k1_IBM_s016
                                 + 0.779755 w_k1_INTC_s016
                                 + 0.711789 w_k1_KO_s016
                                 + 0.613021 w_k1_MAT_s016
                                 + 0.835957 w_k1_MSFT_s016
                                 + 0.898674 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.834179 w_k1_ADM_s016
                                 + 0.87259 w_k1_AXP_s016
                                 + 1.10908 w_k1_DIS_s016
                                 + 0.94664 w_k1_FDX_s016
                                 + 1.06438 w_k1_IBM_s016
                                 + 1.14119 w_k1_INTC_s016
                                 + 1.00039 w_k1_KO_s016
                                 + 0.786553 w_k1_MAT_s016
                                 + 1.00314 w_k1_MSFT_s016
                                 + 1.00039 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.27612 w_k1_ADM_s016
                                 + 1.0685 w_k1_AXP_s016 + 1.21068 w_k1_DIS_s016
                                 + 1.17889 w_k1_FDX_s016
                                 + 0.862283 w_k1_IBM_s016
                                 + 1.13123 w_k1_INTC_s016
                                 + 0.974629 w_k1_KO_s016
                                 + 1.06611 w_k1_MAT_s016
                                 + 1.08112 w_k1_MSFT_s016 + 1.1626 w_k1_PG_s016
                                 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.785712 w_k1_ADM_s016
                                 + 0.809519 w_k1_AXP_s016
                                 + 0.609233 w_k1_DIS_s016
                                 + 0.776391 w_k1_FDX_s016
                                 + 0.969449 w_k1_IBM_s016
                                 + 0.872219 w_k1_INTC_s016
                                 + 0.8298 w_k1_KO_s016 + 1.06894 w_k1_MAT_s016
                                 + 0.755727 w_k1_MSFT_s016
                                 + 0.840527 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.20932 w_k1_ADM_s016
                                 + 1.00543 w_k1_AXP_s016
                                 + 1.13471 w_k1_DIS_s016
                                 + 1.19839 w_k1_FDX_s016
                                 + 0.983568 w_k1_IBM_s016
                                 + 1.12298 w_k1_INTC_s016
                                 + 0.972852 w_k1_KO_s016
                                 + 0.742214 w_k1_MAT_s016
                                 + 1.0876 w_k1_MSFT_s016 + 1.03695 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.17816 w_k1_ADM_s016
                                 + 1.28242 w_k1_AXP_s016
                                 + 1.24286 w_k1_DIS_s016
                                 + 1.16126 w_k1_FDX_s016
                                 + 1.42521 w_k1_IBM_s016
                                 + 1.17085 w_k1_INTC_s016
                                 + 1.14872 w_k1_KO_s016 + 1.64417 w_k1_MAT_s016
                                 + 1.10267 w_k1_MSFT_s016
                                 + 1.22139 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.26432 w_k1_ADM_s017
                                 + 0.870383 w_k1_AXP_s017
                                 + 0.988369 w_k1_DIS_s017
                                 + 0.951223 w_k1_FDX_s017
                                 + 0.933102 w_k1_IBM_s017
                                 + 1.08899 w_k1_INTC_s017
                                 + 0.94439 w_k1_KO_s017
                                 + 0.729478 w_k1_MAT_s017
                                 + 1.04776 w_k1_MSFT_s017
                                 + 0.966877 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.932592 w_k1_ADM_s017
                                 + 0.848695 w_k1_AXP_s017
                                 + 0.635262 w_k1_DIS_s017
                                 + 0.786254 w_k1_FDX_s017
                                 + 0.732624 w_k1_IBM_s017
                                 + 0.953072 w_k1_INTC_s017
                                 + 0.650489 w_k1_KO_s017
                                 + 0.52065 w_k1_MAT_s017
                                 + 0.800399 w_k1_MSFT_s017
                                 + 0.897078 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.28515 w_k1_ADM_s017
                                 + 0.750293 w_k1_AXP_s017
                                 + 0.948402 w_k1_DIS_s017
                                 + 0.973719 w_k1_FDX_s017
                                 + 0.643935 w_k1_IBM_s017
                                 + 0.932992 w_k1_INTC_s017
                                 + 0.812383 w_k1_KO_s017
                                 + 0.50311 w_k1_MAT_s017
                                 + 1.01397 w_k1_MSFT_s017
                                 + 0.974646 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.14895 w_k1_ADM_s017
                                 + 1.24089 w_k1_AXP_s017
                                 + 0.875123 w_k1_DIS_s017
                                 + 0.906624 w_k1_FDX_s017
                                 + 1.09827 w_k1_IBM_s017
                                 + 0.843825 w_k1_INTC_s017
                                 + 0.827796 w_k1_KO_s017
                                 + 1.12178 w_k1_MAT_s017
                                 + 0.753023 w_k1_MSFT_s017
                                 + 0.949715 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.36065 w_k1_ADM_s017
                                 + 1.30789 w_k1_AXP_s017
                                 + 1.42253 w_k1_DIS_s017
                                 + 1.19811 w_k1_FDX_s017
                                 + 1.14291 w_k1_IBM_s017
                                 + 1.14586 w_k1_INTC_s017
                                 + 1.15604 w_k1_KO_s017 + 1.08647 w_k1_MAT_s017
                                 + 1.21891 w_k1_MSFT_s017 + 1.0176 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.59569 w_k1_ADM_s017
                                 + 1.76381 w_k1_AXP_s017
                                 + 1.77935 w_k1_DIS_s017 + 1.3956 w_k1_FDX_s017
                                 + 1.53484 w_k1_IBM_s017
                                 + 1.13104 w_k1_INTC_s017
                                 + 1.52928 w_k1_KO_s017 + 1.92076 w_k1_MAT_s017
                                 + 1.18131 w_k1_MSFT_s017 + 1.2415 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.36611 w_k1_ADM_s017
                                 + 1.35105 w_k1_AXP_s017
                                 + 1.32946 w_k1_DIS_s017 + 1.3942 w_k1_FDX_s017
                                 + 0.863533 w_k1_IBM_s017
                                 + 1.13632 w_k1_INTC_s017
                                 + 1.11136 w_k1_KO_s017
                                 + 0.930463 w_k1_MAT_s017
                                 + 1.21158 w_k1_MSFT_s017 + 1.1104 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.925491 w_k1_ADM_s017
                                 + 0.911358 w_k1_AXP_s017
                                 + 1.02771 w_k1_DIS_s017
                                 + 0.798128 w_k1_FDX_s017
                                 + 1.44084 w_k1_IBM_s017
                                 + 0.870869 w_k1_INTC_s017
                                 + 1.22761 w_k1_KO_s017 + 1.3312 w_k1_MAT_s017
                                 + 0.817628 w_k1_MSFT_s017
                                 + 1.00439 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.03514 w_k1_ADM_s100
                                 + 0.779126 w_k1_AXP_s100
                                 + 0.823275 w_k1_DIS_s100
                                 + 0.969524 w_k1_FDX_s100
                                 + 0.756328 w_k1_IBM_s100
                                 + 1.11042 w_k1_INTC_s100
                                 + 0.721242 w_k1_KO_s100
                                 + 0.428979 w_k1_MAT_s100
                                 + 1.01759 w_k1_MSFT_s100
                                 + 0.884053 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.982061 w_k1_ADM_s100
                                 + 1.01752 w_k1_AXP_s100
                                 + 1.26171 w_k1_DIS_s100
                                 + 1.06377 w_k1_FDX_s100
                                 + 1.50747 w_k1_IBM_s100
                                 + 1.17989 w_k1_INTC_s100
                                 + 1.21742 w_k1_KO_s100 + 1.30418 w_k1_MAT_s100
                                 + 1.17876 w_k1_MSFT_s100
                                 + 1.04438 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.36934 w_k1_ADM_s100
                                 + 1.63002 w_k1_AXP_s100
                                 + 1.39705 w_k1_DIS_s100
                                 + 1.33154 w_k1_FDX_s100
                                 + 1.06454 w_k1_IBM_s100
                                 + 1.27159 w_k1_INTC_s100
                                 + 0.923208 w_k1_KO_s100
                                 + 1.27305 w_k1_MAT_s100
                                 + 1.28919 w_k1_MSFT_s100
                                 + 1.15881 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.21011 w_k1_ADM_s100
                                 + 1.1228 w_k1_AXP_s100 + 1.06126 w_k1_DIS_s100
                                 + 1.11268 w_k1_FDX_s100
                                 + 0.941042 w_k1_IBM_s100
                                 + 1.30027 w_k1_INTC_s100
                                 + 0.86951 w_k1_KO_s100
                                 + 0.851736 w_k1_MAT_s100
                                 + 1.06962 w_k1_MSFT_s100
                                 + 1.00985 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.0377 w_k1_ADM_s100
                                 + 0.815975 w_k1_AXP_s100
                                 + 0.840479 w_k1_DIS_s100
                                 + 0.82705 w_k1_FDX_s100
                                 + 0.902452 w_k1_IBM_s100
                                 + 0.771363 w_k1_INTC_s100
                                 + 0.71973 w_k1_KO_s100
                                 + 0.836012 w_k1_MAT_s100
                                 + 0.987232 w_k1_MSFT_s100
                                 + 0.846007 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.13109 w_k1_ADM_s100
                                 + 0.973715 w_k1_AXP_s100
                                 + 0.872031 w_k1_DIS_s100
                                 + 0.959028 w_k1_FDX_s100
                                 + 1.21972 w_k1_IBM_s100
                                 + 1.08349 w_k1_INTC_s100
                                 + 0.910122 w_k1_KO_s100
                                 + 0.739878 w_k1_MAT_s100
                                 + 1.11665 w_k1_MSFT_s100
                                 + 1.02122 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.12493 w_k1_ADM_s100
                                 + 0.647131 w_k1_AXP_s100
                                 + 0.681907 w_k1_DIS_s100
                                 + 0.775964 w_k1_FDX_s100
                                 + 0.69956 w_k1_IBM_s100
                                 + 0.754509 w_k1_INTC_s100
                                 + 0.634495 w_k1_KO_s100
                                 + 0.582352 w_k1_MAT_s100
                                 + 0.815512 w_k1_MSFT_s100
                                 + 0.778443 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.40367 w_k1_ADM_s100
                                 + 1.47779 w_k1_AXP_s100
                                 + 1.83772 w_k1_DIS_s100
                                 + 1.48635 w_k1_FDX_s100
                                 + 1.28569 w_k1_IBM_s100
                                 + 1.35419 w_k1_INTC_s100 + 1.2886 w_k1_KO_s100
                                 + 1.24625 w_k1_MAT_s100
                                 + 1.34087 w_k1_MSFT_s100
                                 + 1.28998 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.07931 w_k1_ADM_s101
                                 + 1.34475 w_k1_AXP_s101 + 1.3348 w_k1_DIS_s101
                                 + 1.1473 w_k1_FDX_s101 + 1.76339 w_k1_IBM_s101
                                 + 1.22486 w_k1_INTC_s101
                                 + 1.38436 w_k1_KO_s101 + 1.42179 w_k1_MAT_s101
                                 + 1.19639 w_k1_MSFT_s101
                                 + 1.14273 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.996181 w_k1_ADM_s101
                                 + 0.701117 w_k1_AXP_s101
                                 + 0.791193 w_k1_DIS_s101
                                 + 0.913534 w_k1_FDX_s101
                                 + 0.661819 w_k1_IBM_s101
                                 + 0.78703 w_k1_INTC_s101
                                 + 0.804634 w_k1_KO_s101
                                 + 0.508274 w_k1_MAT_s101
                                 + 0.987774 w_k1_MSFT_s101
                                 + 0.945426 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.670126 w_k1_ADM_s101
                                 + 0.728173 w_k1_AXP_s101
                                 + 0.705816 w_k1_DIS_s101
                                 + 0.825815 w_k1_FDX_s101
                                 + 0.957302 w_k1_IBM_s101
                                 + 0.987365 w_k1_INTC_s101
                                 + 0.946919 w_k1_KO_s101
                                 + 1.01264 w_k1_MAT_s101
                                 + 0.985391 w_k1_MSFT_s101
                                 + 0.928549 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.919127 w_k1_ADM_s101
                                 + 0.985218 w_k1_AXP_s101
                                 + 1.15275 w_k1_DIS_s101 + 1.1266 w_k1_FDX_s101
                                 + 0.775892 w_k1_IBM_s101
                                 + 1.03629 w_k1_INTC_s101
                                 + 1.15136 w_k1_KO_s101
                                 + 0.962824 w_k1_MAT_s101
                                 + 1.12585 w_k1_MSFT_s101
                                 + 1.26127 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.991455 w_k1_ADM_s101
                                 + 0.689022 w_k1_AXP_s101
                                 + 1.29821 w_k1_DIS_s101 + 1.0301 w_k1_FDX_s101
                                 + 0.891483 w_k1_IBM_s101
                                 + 1.07683 w_k1_INTC_s101
                                 + 1.13125 w_k1_KO_s101
                                 + 0.902604 w_k1_MAT_s101
                                 + 1.26052 w_k1_MSFT_s101
                                 + 1.04976 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.881318 w_k1_ADM_s101
                                 + 0.931435 w_k1_AXP_s101
                                 + 0.857954 w_k1_DIS_s101
                                 + 0.83952 w_k1_FDX_s101
                                 + 0.841971 w_k1_IBM_s101
                                 + 0.754536 w_k1_INTC_s101
                                 + 0.873745 w_k1_KO_s101
                                 + 0.919782 w_k1_MAT_s101
                                 + 0.895937 w_k1_MSFT_s101
                                 + 1.00539 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.951479 w_k1_ADM_s101
                                 + 0.766868 w_k1_AXP_s101
                                 + 0.862938 w_k1_DIS_s101
                                 + 0.844576 w_k1_FDX_s101
                                 + 1.32225 w_k1_IBM_s101
                                 + 0.9676 w_k1_INTC_s101 + 1.27009 w_k1_KO_s101
                                 + 1.05649 w_k1_MAT_s101
                                 + 0.934349 w_k1_MSFT_s101
                                 + 1.0453 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.841461 w_k1_ADM_s101
                                 + 1.07635 w_k1_AXP_s101
                                 + 1.26836 w_k1_DIS_s101
                                 + 1.11951 w_k1_FDX_s101
                                 + 1.11804 w_k1_IBM_s101
                                 + 0.924033 w_k1_INTC_s101
                                 + 1.19308 w_k1_KO_s101 + 1.57473 w_k1_MAT_s101
                                 + 1.12137 w_k1_MSFT_s101 + 1.0853 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.10652 w_k1_ADM_s102
                                 + 1.47382 w_k1_AXP_s102
                                 + 1.34018 w_k1_DIS_s102
                                 + 1.15476 w_k1_FDX_s102
                                 + 1.37786 w_k1_IBM_s102
                                 + 1.13209 w_k1_INTC_s102
                                 + 1.42492 w_k1_KO_s102 + 2.42032 w_k1_MAT_s102
                                 + 1.34018 w_k1_MSFT_s102
                                 + 1.34412 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.19769 w_k1_ADM_s102
                                 + 1.46475 w_k1_AXP_s102
                                 + 1.54887 w_k1_DIS_s102
                                 + 1.44565 w_k1_FDX_s102
                                 + 1.02461 w_k1_IBM_s102
                                 + 1.03316 w_k1_INTC_s102
                                 + 1.23145 w_k1_KO_s102 + 1.62239 w_k1_MAT_s102
                                 + 1.3367 w_k1_MSFT_s102 + 1.24976 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.17442 w_k1_ADM_s102
                                 + 1.28155 w_k1_AXP_s102
                                 + 1.38035 w_k1_DIS_s102
                                 + 1.29028 w_k1_FDX_s102
                                 + 0.900761 w_k1_IBM_s102
                                 + 1.13235 w_k1_INTC_s102
                                 + 1.09761 w_k1_KO_s102 + 1.18932 w_k1_MAT_s102
                                 + 1.31123 w_k1_MSFT_s102
                                 + 1.15114 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.839724 w_k1_ADM_s102
                                 + 0.933681 w_k1_AXP_s102
                                 + 0.833013 w_k1_DIS_s102
                                 + 0.933499 w_k1_FDX_s102
                                 + 0.724818 w_k1_IBM_s102
                                 + 1.09436 w_k1_INTC_s102
                                 + 0.782552 w_k1_KO_s102
                                 + 0.649632 w_k1_MAT_s102
                                 + 1.0188 w_k1_MSFT_s102
                                 + 0.924393 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.963985 w_k1_ADM_s102
                                 + 1.1889 w_k1_AXP_s102
                                 + 0.969965 w_k1_DIS_s102
                                 + 0.869967 w_k1_FDX_s102
                                 + 1.43412 w_k1_IBM_s102
                                 + 1.04264 w_k1_INTC_s102
                                 + 1.24854 w_k1_KO_s102 + 1.31097 w_k1_MAT_s102
                                 + 1.0205 w_k1_MSFT_s102
                                 + 0.971748 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.01594 w_k1_ADM_s102
                                 + 1.00453 w_k1_AXP_s102
                                 + 0.905373 w_k1_DIS_s102
                                 + 1.04563 w_k1_FDX_s102
                                 + 0.923551 w_k1_IBM_s102
                                 + 1.15088 w_k1_INTC_s102
                                 + 1.04396 w_k1_KO_s102 + 1.07865 w_k1_MAT_s102
                                 + 1.01186 w_k1_MSFT_s102
                                 + 1.05967 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.674653 w_k1_ADM_s102
                                 + 0.777057 w_k1_AXP_s102
                                 + 0.629664 w_k1_DIS_s102
                                 + 0.740851 w_k1_FDX_s102
                                 + 0.920014 w_k1_IBM_s102
                                 + 1.00215 w_k1_INTC_s102
                                 + 0.715365 w_k1_KO_s102
                                 + 0.942371 w_k1_MAT_s102
                                 + 0.886753 w_k1_MSFT_s102
                                 + 0.854683 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.959014 w_k1_ADM_s102
                                 + 1.04624 w_k1_AXP_s102
                                 + 1.18795 w_k1_DIS_s102
                                 + 1.10286 w_k1_FDX_s102
                                 + 1.18039 w_k1_IBM_s102
                                 + 1.03568 w_k1_INTC_s102
                                 + 1.23547 w_k1_KO_s102 + 1.41458 w_k1_MAT_s102
                                 + 1.15687 w_k1_MSFT_s102
                                 + 1.03183 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.34371 w_k1_ADM_s103
                                 + 1.03566 w_k1_AXP_s103
                                 + 1.01246 w_k1_DIS_s103
                                 + 1.12452 w_k1_FDX_s103
                                 + 0.702494 w_k1_IBM_s103
                                 + 1.29384 w_k1_INTC_s103
                                 + 0.935398 w_k1_KO_s103
                                 + 0.625168 w_k1_MAT_s103
                                 + 1.03688 w_k1_MSFT_s103
                                 + 1.14113 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.976619 w_k1_ADM_s103
                                 + 0.679555 w_k1_AXP_s103
                                 + 0.732514 w_k1_DIS_s103
                                 + 0.793681 w_k1_FDX_s103
                                 + 1.08109 w_k1_IBM_s103
                                 + 0.828391 w_k1_INTC_s103
                                 + 0.930316 w_k1_KO_s103
                                 + 1.03674 w_k1_MAT_s103
                                 + 0.828724 w_k1_MSFT_s103
                                 + 0.979608 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.20824 w_k1_ADM_s103
                                 + 0.724928 w_k1_AXP_s103
                                 + 0.919848 w_k1_DIS_s103
                                 + 0.877341 w_k1_FDX_s103
                                 + 1.27098 w_k1_IBM_s103
                                 + 1.07311 w_k1_INTC_s103 + 1.0746 w_k1_KO_s103
                                 + 0.943041 w_k1_MAT_s103
                                 + 0.823002 w_k1_MSFT_s103
                                 + 0.947784 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.26157 w_k1_ADM_s103
                                 + 1.44627 w_k1_AXP_s103
                                 + 1.34878 w_k1_DIS_s103
                                 + 1.19829 w_k1_FDX_s103 + 1.4088 w_k1_IBM_s103
                                 + 1.23009 w_k1_INTC_s103
                                 + 1.13023 w_k1_KO_s103 + 1.49957 w_k1_MAT_s103
                                 + 1.02936 w_k1_MSFT_s103
                                 + 1.21844 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.732281 w_k1_ADM_s103
                                 + 0.671855 w_k1_AXP_s103
                                 + 0.562322 w_k1_DIS_s103
                                 + 0.737359 w_k1_FDX_s103
                                 + 0.775726 w_k1_IBM_s103
                                 + 0.852551 w_k1_INTC_s103
                                 + 0.69354 w_k1_KO_s103
                                 + 0.485197 w_k1_MAT_s103
                                 + 0.688076 w_k1_MSFT_s103
                                 + 0.789004 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.39369 w_k1_ADM_s103
                                 + 1.18764 w_k1_AXP_s103
                                 + 1.23748 w_k1_DIS_s103
                                 + 1.19816 w_k1_FDX_s103
                                 + 0.929796 w_k1_IBM_s103
                                 + 1.29992 w_k1_INTC_s103 + 1.2879 w_k1_KO_s103
                                 + 0.776836 w_k1_MAT_s103
                                 + 1.14944 w_k1_MSFT_s103
                                 + 1.09949 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.945684 w_k1_ADM_s103
                                 + 0.850314 w_k1_AXP_s103
                                 + 0.717668 w_k1_DIS_s103
                                 + 0.917678 w_k1_FDX_s103
                                 + 0.677185 w_k1_IBM_s103
                                 + 1.06451 w_k1_INTC_s103
                                 + 0.771635 w_k1_KO_s103
                                 + 0.630098 w_k1_MAT_s103
                                 + 0.880476 w_k1_MSFT_s103
                                 + 0.915152 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01455 w_k1_ADM_s103
                                 + 0.984055 w_k1_AXP_s103
                                 + 0.805906 w_k1_DIS_s103
                                 + 1.02266 w_k1_FDX_s103
                                 + 0.842346 w_k1_IBM_s103
                                 + 1.04238 w_k1_INTC_s103
                                 + 0.811996 w_k1_KO_s103
                                 + 0.82782 w_k1_MAT_s103
                                 + 0.841796 w_k1_MSFT_s103
                                 + 0.964448 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.80164 w_k1_ADM_s104
                                 + 0.677368 w_k1_AXP_s104
                                 + 0.684462 w_k1_DIS_s104
                                 + 0.687152 w_k1_FDX_s104
                                 + 0.976285 w_k1_IBM_s104
                                 + 0.608149 w_k1_INTC_s104
                                 + 0.78511 w_k1_KO_s104 + 0.73061 w_k1_MAT_s104
                                 + 0.757524 w_k1_MSFT_s104
                                 + 0.953904 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.27373 w_k1_ADM_s104
                                 + 1.30753 w_k1_AXP_s104
                                 + 1.39617 w_k1_DIS_s104
                                 + 1.35748 w_k1_FDX_s104
                                 + 0.93604 w_k1_IBM_s104
                                 + 0.866917 w_k1_INTC_s104
                                 + 0.967881 w_k1_KO_s104
                                 + 0.922114 w_k1_MAT_s104
                                 + 1.11638 w_k1_MSFT_s104
                                 + 1.04717 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.878149 w_k1_ADM_s104
                                 + 0.932519 w_k1_AXP_s104
                                 + 0.878998 w_k1_DIS_s104
                                 + 0.94617 w_k1_FDX_s104
                                 + 1.06381 w_k1_IBM_s104
                                 + 0.742084 w_k1_INTC_s104
                                 + 0.903223 w_k1_KO_s104
                                 + 1.13482 w_k1_MAT_s104
                                 + 0.913613 w_k1_MSFT_s104
                                 + 1.05292 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.800299 w_k1_ADM_s104
                                 + 1.13459 w_k1_AXP_s104
                                 + 1.07234 w_k1_DIS_s104
                                 + 0.967904 w_k1_FDX_s104
                                 + 1.64357 w_k1_IBM_s104
                                 + 0.927731 w_k1_INTC_s104
                                 + 1.31819 w_k1_KO_s104 + 1.72216 w_k1_MAT_s104
                                 + 1.11549 w_k1_MSFT_s104
                                 + 1.15519 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.868497 w_k1_ADM_s104
                                 + 0.987786 w_k1_AXP_s104
                                 + 1.19614 w_k1_DIS_s104
                                 + 1.12578 w_k1_FDX_s104
                                 + 0.761642 w_k1_IBM_s104
                                 + 0.958742 w_k1_INTC_s104
                                 + 0.84917 w_k1_KO_s104
                                 + 0.555905 w_k1_MAT_s104
                                 + 0.990139 w_k1_MSFT_s104
                                 + 0.749256 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.10648 w_k1_ADM_s104
                                 + 1.42843 w_k1_AXP_s104
                                 + 1.09599 w_k1_DIS_s104
                                 + 1.13906 w_k1_FDX_s104
                                 + 1.14759 w_k1_IBM_s104
                                 + 1.21234 w_k1_INTC_s104
                                 + 1.14572 w_k1_KO_s104 + 1.21165 w_k1_MAT_s104
                                 + 1.14152 w_k1_MSFT_s104
                                 + 1.03025 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.896571 w_k1_ADM_s104
                                 + 0.998398 w_k1_AXP_s104
                                 + 1.56083 w_k1_DIS_s104
                                 + 1.23417 w_k1_FDX_s104
                                 + 1.26837 w_k1_IBM_s104
                                 + 1.06649 w_k1_INTC_s104
                                 + 1.15983 w_k1_KO_s104
                                 + 0.758566 w_k1_MAT_s104
                                 + 1.09526 w_k1_MSFT_s104
                                 + 1.01878 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.92241 w_k1_ADM_s104
                                 + 0.776919 w_k1_AXP_s104
                                 + 0.617279 w_k1_DIS_s104
                                 + 0.855917 w_k1_FDX_s104
                                 + 0.667903 w_k1_IBM_s104
                                 + 0.716456 w_k1_INTC_s104
                                 + 0.68045 w_k1_KO_s104 + 0.46407 w_k1_MAT_s104
                                 + 0.752875 w_k1_MSFT_s104
                                 + 0.84708 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.12336 w_k1_ADM_s105
                                 + 1.23665 w_k1_AXP_s105
                                 + 1.07683 w_k1_DIS_s105
                                 + 1.12392 w_k1_FDX_s105 + 1.2952 w_k1_IBM_s105
                                 + 1.00325 w_k1_INTC_s105 + 1.1303 w_k1_KO_s105
                                 + 1.1146 w_k1_MAT_s105
                                 + 1.13427 w_k1_MSFT_s105
                                 + 0.997267 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.25383 w_k1_ADM_s105
                                 + 1.46533 w_k1_AXP_s105
                                 + 0.84441 w_k1_DIS_s105
                                 + 1.18342 w_k1_FDX_s105
                                 + 0.733784 w_k1_IBM_s105
                                 + 0.990944 w_k1_INTC_s105
                                 + 0.780693 w_k1_KO_s105
                                 + 0.679069 w_k1_MAT_s105
                                 + 0.970033 w_k1_MSFT_s105
                                 + 0.985043 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.21932 w_k1_ADM_s105
                                 + 0.895431 w_k1_AXP_s105
                                 + 1.29849 w_k1_DIS_s105 + 1.1073 w_k1_FDX_s105
                                 + 0.81931 w_k1_IBM_s105
                                 + 1.03346 w_k1_INTC_s105
                                 + 0.983971 w_k1_KO_s105
                                 + 0.617173 w_k1_MAT_s105
                                 + 1.3277 w_k1_MSFT_s105 + 1.09553 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.775927 w_k1_ADM_s105
                                 + 0.768549 w_k1_AXP_s105
                                 + 0.868456 w_k1_DIS_s105
                                 + 0.900545 w_k1_FDX_s105
                                 + 0.781999 w_k1_IBM_s105
                                 + 1.05144 w_k1_INTC_s105
                                 + 0.735451 w_k1_KO_s105
                                 + 0.621906 w_k1_MAT_s105
                                 + 1.01928 w_k1_MSFT_s105
                                 + 0.858456 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.07371 w_k1_ADM_s105
                                 + 0.946827 w_k1_AXP_s105
                                 + 0.672568 w_k1_DIS_s105
                                 + 0.834736 w_k1_FDX_s105
                                 + 1.05786 w_k1_IBM_s105
                                 + 0.807233 w_k1_INTC_s105
                                 + 0.774596 w_k1_KO_s105
                                 + 0.575239 w_k1_MAT_s105
                                 + 0.808949 w_k1_MSFT_s105
                                 + 0.887241 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.294 w_k1_ADM_s105
                                 + 1.44994 w_k1_AXP_s105
                                 + 1.49801 w_k1_DIS_s105
                                 + 1.40521 w_k1_FDX_s105
                                 + 1.44994 w_k1_IBM_s105
                                 + 1.00448 w_k1_INTC_s105 + 1.1442 w_k1_KO_s105
                                 + 1.17778 w_k1_MAT_s105
                                 + 1.1812 w_k1_MSFT_s105 + 1.11702 w_k1_PG_s105
                                 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.886005 w_k1_ADM_s105
                                 + 1.07154 w_k1_AXP_s105
                                 + 1.00677 w_k1_DIS_s105
                                 + 0.857267 w_k1_FDX_s105
                                 + 1.47381 w_k1_IBM_s105
                                 + 1.0162 w_k1_INTC_s105
                                 + 0.999133 w_k1_KO_s105
                                 + 0.807736 w_k1_MAT_s105
                                 + 0.895135 w_k1_MSFT_s105
                                 + 0.976939 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.39841 w_k1_ADM_s105
                                 + 1.46952 w_k1_AXP_s105
                                 + 1.58614 w_k1_DIS_s105
                                 + 1.39825 w_k1_FDX_s105
                                 + 0.892848 w_k1_IBM_s105
                                 + 1.07945 w_k1_INTC_s105
                                 + 1.19414 w_k1_KO_s105
                                 + 0.821037 w_k1_MAT_s105
                                 + 1.41811 w_k1_MSFT_s105 + 1.2088 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.861115 w_k1_ADM_s106
                                 + 0.801711 w_k1_AXP_s106
                                 + 0.821157 w_k1_DIS_s106
                                 + 0.867274 w_k1_FDX_s106
                                 + 0.986344 w_k1_IBM_s106
                                 + 0.850083 w_k1_INTC_s106
                                 + 0.843274 w_k1_KO_s106
                                 + 0.739329 w_k1_MAT_s106
                                 + 0.837376 w_k1_MSFT_s106
                                 + 0.800928 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.22351 w_k1_ADM_s106
                                 + 1.02571 w_k1_AXP_s106
                                 + 1.06249 w_k1_DIS_s106
                                 + 0.99365 w_k1_FDX_s106
                                 + 0.984901 w_k1_IBM_s106
                                 + 0.916707 w_k1_INTC_s106
                                 + 0.990597 w_k1_KO_s106
                                 + 0.789387 w_k1_MAT_s106
                                 + 0.830021 w_k1_MSFT_s106
                                 + 0.99138 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.10904 w_k1_ADM_s106
                                 + 0.905009 w_k1_AXP_s106
                                 + 1.12332 w_k1_DIS_s106
                                 + 0.887008 w_k1_FDX_s106
                                 + 1.76519 w_k1_IBM_s106
                                 + 0.776367 w_k1_INTC_s106
                                 + 1.35967 w_k1_KO_s106 + 1.71121 w_k1_MAT_s106
                                 + 0.872702 w_k1_MSFT_s106
                                 + 1.05867 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.09482 w_k1_ADM_s106
                                 + 0.961092 w_k1_AXP_s106
                                 + 1.43818 w_k1_DIS_s106
                                 + 1.19528 w_k1_FDX_s106
                                 + 1.06708 w_k1_IBM_s106
                                 + 1.25783 w_k1_INTC_s106
                                 + 1.55519 w_k1_KO_s106
                                 + 0.884016 w_k1_MAT_s106
                                 + 1.41107 w_k1_MSFT_s106
                                 + 1.25919 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.35927 w_k1_ADM_s106
                                 + 1.29097 w_k1_AXP_s106
                                 + 1.15185 w_k1_DIS_s106
                                 + 1.12407 w_k1_FDX_s106
                                 + 1.18676 w_k1_IBM_s106
                                 + 1.35799 w_k1_INTC_s106
                                 + 1.37012 w_k1_KO_s106 + 1.43043 w_k1_MAT_s106
                                 + 1.12739 w_k1_MSFT_s106
                                 + 1.13792 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.38203 w_k1_ADM_s106
                                 + 1.09414 w_k1_AXP_s106
                                 + 1.16491 w_k1_DIS_s106
                                 + 1.11372 w_k1_FDX_s106
                                 + 1.01821 w_k1_IBM_s106
                                 + 1.15502 w_k1_INTC_s106
                                 + 1.11851 w_k1_KO_s106 + 0.90846 w_k1_MAT_s106
                                 + 1.17066 w_k1_MSFT_s106
                                 + 1.00329 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.5113 w_k1_ADM_s106
                                 + 1.30647 w_k1_AXP_s106 + 1.68 w_k1_DIS_s106
                                 + 1.26969 w_k1_FDX_s106
                                 + 1.43591 w_k1_IBM_s106
                                 + 0.947082 w_k1_INTC_s106
                                 + 1.48817 w_k1_KO_s106 + 1.51391 w_k1_MAT_s106
                                 + 1.04793 w_k1_MSFT_s106
                                 + 1.32653 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.957557 w_k1_ADM_s106
                                 + 0.835777 w_k1_AXP_s106
                                 + 1.13853 w_k1_DIS_s106
                                 + 1.08291 w_k1_FDX_s106
                                 + 0.690794 w_k1_IBM_s106
                                 + 1.0065 w_k1_INTC_s106
                                 + 0.978184 w_k1_KO_s106
                                 + 0.736446 w_k1_MAT_s106
                                 + 1.08178 w_k1_MSFT_s106
                                 + 0.968353 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.778022 w_k1_ADM_s107
                                 + 0.750065 w_k1_AXP_s107
                                 + 0.651988 w_k1_DIS_s107
                                 + 0.718999 w_k1_FDX_s107
                                 + 1.00988 w_k1_IBM_s107
                                 + 0.928995 w_k1_INTC_s107
                                 + 0.90232 w_k1_KO_s107
                                 + 0.741346 w_k1_MAT_s107
                                 + 0.761895 w_k1_MSFT_s107
                                 + 0.901251 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.40173 w_k1_ADM_s107
                                 + 1.29282 w_k1_AXP_s107
                                 + 1.74435 w_k1_DIS_s107
                                 + 1.33813 w_k1_FDX_s107
                                 + 1.41418 w_k1_IBM_s107
                                 + 1.16022 w_k1_INTC_s107
                                 + 1.36098 w_k1_KO_s107 + 1.06826 w_k1_MAT_s107
                                 + 1.16718 w_k1_MSFT_s107
                                 + 1.32938 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.859622 w_k1_ADM_s107
                                 + 0.894962 w_k1_AXP_s107
                                 + 0.895508 w_k1_DIS_s107
                                 + 0.913542 w_k1_FDX_s107
                                 + 0.741805 w_k1_IBM_s107
                                 + 1.0133 w_k1_INTC_s107
                                 + 0.831252 w_k1_KO_s107
                                 + 0.569174 w_k1_MAT_s107
                                 + 0.896051 w_k1_MSFT_s107
                                 + 0.914828 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.04712 w_k1_ADM_s107
                                 + 1.156 w_k1_AXP_s107 + 1.14792 w_k1_DIS_s107
                                 + 1.23424 w_k1_FDX_s107
                                 + 0.866078 w_k1_IBM_s107
                                 + 1.13706 w_k1_INTC_s107
                                 + 1.01613 w_k1_KO_s107
                                 + 0.661645 w_k1_MAT_s107
                                 + 0.989846 w_k1_MSFT_s107
                                 + 1.0827 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.807921 w_k1_ADM_s107
                                 + 1.01986 w_k1_AXP_s107
                                 + 1.00894 w_k1_DIS_s107
                                 + 0.864819 w_k1_FDX_s107
                                 + 1.3378 w_k1_IBM_s107
                                 + 1.14661 w_k1_INTC_s107
                                 + 1.16649 w_k1_KO_s107 + 1.3527 w_k1_MAT_s107
                                 + 0.871839 w_k1_MSFT_s107
                                 + 1.09885 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.91321 w_k1_ADM_s107
                                 + 1.108 w_k1_AXP_s107 + 1.14067 w_k1_DIS_s107
                                 + 0.971663 w_k1_FDX_s107
                                 + 1.39419 w_k1_IBM_s107
                                 + 0.994701 w_k1_INTC_s107
                                 + 1.29406 w_k1_KO_s107 + 1.30257 w_k1_MAT_s107
                                 + 0.951039 w_k1_MSFT_s107
                                 + 1.06629 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.745982 w_k1_ADM_s107
                                 + 0.72268 w_k1_AXP_s107
                                 + 0.902306 w_k1_DIS_s107
                                 + 0.870484 w_k1_FDX_s107
                                 + 0.984864 w_k1_IBM_s107
                                 + 1.04318 w_k1_INTC_s107
                                 + 1.01282 w_k1_KO_s107
                                 + 0.821952 w_k1_MAT_s107
                                 + 0.838095 w_k1_MSFT_s107
                                 + 0.958985 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.784799 w_k1_ADM_s107
                                 + 0.889186 w_k1_AXP_s107
                                 + 1.28859 w_k1_DIS_s107
                                 + 1.09141 w_k1_FDX_s107
                                 + 0.914825 w_k1_IBM_s107
                                 + 1.02642 w_k1_INTC_s107
                                 + 0.949543 w_k1_KO_s107
                                 + 0.793779 w_k1_MAT_s107
                                 + 1.05837 w_k1_MSFT_s107
                                 + 1.07815 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.992845 w_k1_ADM_s110
                                 + 1.40546 w_k1_AXP_s110
                                 + 1.58251 w_k1_DIS_s110
                                 + 1.12944 w_k1_FDX_s110
                                 + 1.76677 w_k1_IBM_s110 + 1.438 w_k1_INTC_s110
                                 + 1.63906 w_k1_KO_s110 + 2.41198 w_k1_MAT_s110
                                 + 1.19324 w_k1_MSFT_s110
                                 + 1.30898 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.17708 w_k1_ADM_s110
                                 + 1.18157 w_k1_AXP_s110
                                 + 1.36239 w_k1_DIS_s110
                                 + 1.20148 w_k1_FDX_s110
                                 + 1.32897 w_k1_IBM_s110
                                 + 1.07787 w_k1_INTC_s110
                                 + 1.39818 w_k1_KO_s110 + 2.01889 w_k1_MAT_s110
                                 + 1.16712 w_k1_MSFT_s110
                                 + 1.28244 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.994002 w_k1_ADM_s110
                                 + 1.21711 w_k1_AXP_s110
                                 + 1.11109 w_k1_DIS_s110
                                 + 1.08172 w_k1_FDX_s110
                                 + 1.03481 w_k1_IBM_s110
                                 + 1.18162 w_k1_INTC_s110
                                 + 1.18918 w_k1_KO_s110 + 1.38228 w_k1_MAT_s110
                                 + 1.02017 w_k1_MSFT_s110
                                 + 1.11968 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.801341 w_k1_ADM_s110
                                 + 0.817339 w_k1_AXP_s110
                                 + 0.675756 w_k1_DIS_s110
                                 + 0.725774 w_k1_FDX_s110
                                 + 0.839853 w_k1_IBM_s110
                                 + 0.862875 w_k1_INTC_s110
                                 + 0.869966 w_k1_KO_s110
                                 + 0.947229 w_k1_MAT_s110
                                 + 0.755675 w_k1_MSFT_s110
                                 + 0.925219 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.21668 w_k1_ADM_s110
                                 + 1.27143 w_k1_AXP_s110
                                 + 1.50014 w_k1_DIS_s110
                                 + 1.33256 w_k1_FDX_s110
                                 + 0.740866 w_k1_IBM_s110
                                 + 1.19899 w_k1_INTC_s110
                                 + 1.21913 w_k1_KO_s110 + 1.12568 w_k1_MAT_s110
                                 + 1.21191 w_k1_MSFT_s110
                                 + 1.09751 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.889283 w_k1_ADM_s110
                                 + 0.987015 w_k1_AXP_s110
                                 + 0.791454 w_k1_DIS_s110
                                 + 0.853825 w_k1_FDX_s110
                                 + 1.27263 w_k1_IBM_s110
                                 + 1.1349 w_k1_INTC_s110 + 1.01171 w_k1_KO_s110
                                 + 1.6006 w_k1_MAT_s110
                                 + 0.943585 w_k1_MSFT_s110
                                 + 1.06061 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.01992 w_k1_ADM_s110
                                 + 0.761921 w_k1_AXP_s110
                                 + 0.831347 w_k1_DIS_s110
                                 + 0.90711 w_k1_FDX_s110
                                 + 0.637373 w_k1_IBM_s110
                                 + 1.07336 w_k1_INTC_s110
                                 + 0.836709 w_k1_KO_s110
                                 + 0.468515 w_k1_MAT_s110
                                 + 0.96366 w_k1_MSFT_s110
                                 + 0.933913 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.823387 w_k1_ADM_s110
                                 + 0.832262 w_k1_AXP_s110
                                 + 1.0797 w_k1_DIS_s110
                                 + 0.916316 w_k1_FDX_s110
                                 + 1.08472 w_k1_IBM_s110
                                 + 1.07161 w_k1_INTC_s110
                                 + 1.07449 w_k1_KO_s110
                                 + 0.878146 w_k1_MAT_s110
                                 + 1.028 w_k1_MSFT_s110 + 1.10014 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.0887 w_k1_ADM_s111
                                 + 1.14347 w_k1_AXP_s111
                                 + 1.21396 w_k1_DIS_s111
                                 + 1.07724 w_k1_FDX_s111
                                 + 1.51286 w_k1_IBM_s111
                                 + 1.16896 w_k1_INTC_s111
                                 + 1.26802 w_k1_KO_s111 + 1.28452 w_k1_MAT_s111
                                 + 1.13304 w_k1_MSFT_s111
                                 + 1.09216 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.853015 w_k1_ADM_s111
                                 + 1.00039 w_k1_AXP_s111
                                 + 0.768243 w_k1_DIS_s111
                                 + 0.835214 w_k1_FDX_s111
                                 + 1.22404 w_k1_IBM_s111
                                 + 1.01324 w_k1_INTC_s111
                                 + 1.09743 w_k1_KO_s111 + 1.49198 w_k1_MAT_s111
                                 + 0.916635 w_k1_MSFT_s111
                                 + 0.942594 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.03999 w_k1_ADM_s111
                                 + 0.930868 w_k1_AXP_s111
                                 + 1.00051 w_k1_DIS_s111
                                 + 0.95699 w_k1_FDX_s111
                                 + 0.944611 w_k1_IBM_s111
                                 + 1.12775 w_k1_INTC_s111
                                 + 0.951305 w_k1_KO_s111
                                 + 1.18949 w_k1_MAT_s111
                                 + 1.08545 w_k1_MSFT_s111
                                 + 0.916765 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.35248 w_k1_ADM_s111
                                 + 1.41284 w_k1_AXP_s111
                                 + 1.21903 w_k1_DIS_s111
                                 + 1.09703 w_k1_FDX_s111
                                 + 1.20004 w_k1_IBM_s111
                                 + 1.22223 w_k1_INTC_s111
                                 + 1.22292 w_k1_KO_s111 + 1.38537 w_k1_MAT_s111
                                 + 1.14611 w_k1_MSFT_s111
                                 + 1.08898 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.728819 w_k1_ADM_s111
                                 + 0.728279 w_k1_AXP_s111
                                 + 0.670268 w_k1_DIS_s111
                                 + 0.776873 w_k1_FDX_s111
                                 + 0.67693 w_k1_IBM_s111
                                 + 1.06824 w_k1_INTC_s111
                                 + 0.687923 w_k1_KO_s111
                                 + 0.497553 w_k1_MAT_s111
                                 + 0.806969 w_k1_MSFT_s111
                                 + 0.785674 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.00783 w_k1_ADM_s111
                                 + 1.3029 w_k1_AXP_s111 + 1.61424 w_k1_DIS_s111
                                 + 1.1233 w_k1_FDX_s111 + 1.73394 w_k1_IBM_s111
                                 + 1.33637 w_k1_INTC_s111
                                 + 1.62748 w_k1_KO_s111 + 1.66189 w_k1_MAT_s111
                                 + 1.20098 w_k1_MSFT_s111
                                 + 1.07514 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.0773 w_k1_ADM_s111
                                 + 1.27638 w_k1_AXP_s111 + 1.071 w_k1_DIS_s111
                                 + 1.23633 w_k1_FDX_s111
                                 + 0.737614 w_k1_IBM_s111
                                 + 1.20522 w_k1_INTC_s111
                                 + 1.00931 w_k1_KO_s111 + 1.1885 w_k1_MAT_s111
                                 + 1.25942 w_k1_MSFT_s111
                                 + 1.10022 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.07517 w_k1_ADM_s111
                                 + 0.991112 w_k1_AXP_s111
                                 + 1.48775 w_k1_DIS_s111
                                 + 1.22632 w_k1_FDX_s111
                                 + 0.90119 w_k1_IBM_s111
                                 + 1.33284 w_k1_INTC_s111
                                 + 1.10831 w_k1_KO_s111 + 1.11887 w_k1_MAT_s111
                                 + 1.35315 w_k1_MSFT_s111
                                 + 1.04784 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.895302 w_k1_ADM_s112
                                 + 1.04193 w_k1_AXP_s112
                                 + 1.07479 w_k1_DIS_s112
                                 + 1.05688 w_k1_FDX_s112
                                 + 0.993719 w_k1_IBM_s112
                                 + 1.20938 w_k1_INTC_s112
                                 + 1.05401 w_k1_KO_s112 + 1.25613 w_k1_MAT_s112
                                 + 1.04742 w_k1_MSFT_s112
                                 + 1.09366 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.987124 w_k1_ADM_s112
                                 + 0.770543 w_k1_AXP_s112
                                 + 0.870865 w_k1_DIS_s112
                                 + 0.987695 w_k1_FDX_s112
                                 + 0.586233 w_k1_IBM_s112
                                 + 0.924381 w_k1_INTC_s112
                                 + 0.826878 w_k1_KO_s112
                                 + 0.635855 w_k1_MAT_s112
                                 + 0.912693 w_k1_MSFT_s112
                                 + 1.02899 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.988041 w_k1_ADM_s112
                                 + 1.55561 w_k1_AXP_s112
                                 + 1.53475 w_k1_DIS_s112
                                 + 1.12782 w_k1_FDX_s112
                                 + 1.61542 w_k1_IBM_s112
                                 + 1.18304 w_k1_INTC_s112 + 1.2636 w_k1_KO_s112
                                 + 2.11276 w_k1_MAT_s112
                                 + 1.21377 w_k1_MSFT_s112
                                 + 1.29009 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.0792 w_k1_ADM_s112
                                 + 1.09214 w_k1_AXP_s112
                                 + 1.38492 w_k1_DIS_s112
                                 + 1.29813 w_k1_FDX_s112
                                 + 0.764351 w_k1_IBM_s112
                                 + 1.31744 w_k1_INTC_s112
                                 + 0.958751 w_k1_KO_s112
                                 + 0.629673 w_k1_MAT_s112
                                 + 1.19835 w_k1_MSFT_s112
                                 + 1.17902 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.33444 w_k1_ADM_s112
                                 + 1.45116 w_k1_AXP_s112
                                 + 1.65735 w_k1_DIS_s112
                                 + 1.35082 w_k1_FDX_s112
                                 + 1.20661 w_k1_IBM_s112
                                 + 1.37142 w_k1_INTC_s112
                                 + 1.16115 w_k1_KO_s112 + 1.39044 w_k1_MAT_s112
                                 + 1.36865 w_k1_MSFT_s112
                                 + 1.22675 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.688753 w_k1_ADM_s112
                                 + 0.719887 w_k1_AXP_s112
                                 + 0.67511 w_k1_DIS_s112
                                 + 0.711485 w_k1_FDX_s112
                                 + 0.965797 w_k1_IBM_s112
                                 + 0.750336 w_k1_INTC_s112
                                 + 0.70484 w_k1_KO_s112 + 1.37243 w_k1_MAT_s112
                                 + 0.712814 w_k1_MSFT_s112
                                 + 0.90491 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.880425 w_k1_ADM_s112
                                 + 1.14492 w_k1_AXP_s112
                                 + 1.04376 w_k1_DIS_s112
                                 + 0.875858 w_k1_FDX_s112
                                 + 1.37137 w_k1_IBM_s112
                                 + 0.918123 w_k1_INTC_s112
                                 + 1.17608 w_k1_KO_s112 + 1.10265 w_k1_MAT_s112
                                 + 0.970015 w_k1_MSFT_s112
                                 + 1.10988 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.82892 w_k1_ADM_s112
                                 + 0.724153 w_k1_AXP_s112
                                 + 0.794412 w_k1_DIS_s112
                                 + 0.797781 w_k1_FDX_s112
                                 + 0.773975 w_k1_IBM_s112
                                 + 0.850308 w_k1_INTC_s112
                                 + 0.879976 w_k1_KO_s112
                                 + 0.804482 w_k1_MAT_s112
                                 + 0.820286 w_k1_MSFT_s112
                                 + 0.89525 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.09608 w_k1_ADM_s113
                                 + 0.991965 w_k1_AXP_s113
                                 + 1.31527 w_k1_DIS_s113 + 1.0447 w_k1_FDX_s113
                                 + 1.01051 w_k1_IBM_s113
                                 + 1.02578 w_k1_INTC_s113 + 1.0418 w_k1_KO_s113
                                 + 1.0255 w_k1_MAT_s113
                                 + 1.05022 w_k1_MSFT_s113
                                 + 1.04268 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.03615 w_k1_ADM_s113
                                 + 0.764808 w_k1_AXP_s113
                                 + 0.802557 w_k1_DIS_s113
                                 + 0.778187 w_k1_FDX_s113
                                 + 1.27974 w_k1_IBM_s113
                                 + 0.753344 w_k1_INTC_s113
                                 + 0.891508 w_k1_KO_s113
                                 + 1.04569 w_k1_MAT_s113
                                 + 0.677664 w_k1_MSFT_s113
                                 + 0.856357 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.76565 w_k1_ADM_s113
                                 + 0.672886 w_k1_AXP_s113
                                 + 0.733768 w_k1_DIS_s113
                                 + 0.788093 w_k1_FDX_s113
                                 + 0.681734 w_k1_IBM_s113
                                 + 0.733289 w_k1_INTC_s113
                                 + 0.693836 w_k1_KO_s113
                                 + 0.555748 w_k1_MAT_s113
                                 + 0.76286 w_k1_MSFT_s113
                                 + 0.812858 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.15728 w_k1_ADM_s113
                                 + 1.50216 w_k1_AXP_s113
                                 + 1.22069 w_k1_DIS_s113
                                 + 1.09992 w_k1_FDX_s113
                                 + 1.10999 w_k1_IBM_s113
                                 + 1.1496 w_k1_INTC_s113 + 1.0621 w_k1_KO_s113
                                 + 1.50478 w_k1_MAT_s113
                                 + 1.04886 w_k1_MSFT_s113
                                 + 1.00811 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.979397 w_k1_ADM_s113
                                 + 0.754285 w_k1_AXP_s113
                                 + 1.07917 w_k1_DIS_s113
                                 + 0.907435 w_k1_FDX_s113
                                 + 0.93761 w_k1_IBM_s113
                                 + 0.834515 w_k1_INTC_s113
                                 + 0.998316 w_k1_KO_s113
                                 + 0.907015 w_k1_MAT_s113
                                 + 0.951621 w_k1_MSFT_s113
                                 + 1.05163 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.01582 w_k1_ADM_s113
                                 + 1.07172 w_k1_AXP_s113
                                 + 0.826995 w_k1_DIS_s113
                                 + 0.933524 w_k1_FDX_s113
                                 + 0.882058 w_k1_IBM_s113
                                 + 0.851377 w_k1_INTC_s113
                                 + 0.855285 w_k1_KO_s113
                                 + 0.817627 w_k1_MAT_s113
                                 + 0.813542 w_k1_MSFT_s113
                                 + 0.933383 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.22457 w_k1_ADM_s113
                                 + 1.17247 w_k1_AXP_s113
                                 + 1.52058 w_k1_DIS_s113
                                 + 1.21184 w_k1_FDX_s113
                                 + 0.837485 w_k1_IBM_s113
                                 + 1.05227 w_k1_INTC_s113
                                 + 1.12723 w_k1_KO_s113
                                 + 0.978852 w_k1_MAT_s113
                                 + 1.11056 w_k1_MSFT_s113
                                 + 1.14735 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.20552 w_k1_ADM_s113
                                 + 1.33741 w_k1_AXP_s113
                                 + 1.70792 w_k1_DIS_s113
                                 + 1.12123 w_k1_FDX_s113
                                 + 1.47464 w_k1_IBM_s113
                                 + 1.15874 w_k1_INTC_s113
                                 + 1.46778 w_k1_KO_s113 + 1.576 w_k1_MAT_s113
                                 + 1.11844 w_k1_MSFT_s113
                                 + 1.15259 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.21284 w_k1_ADM_s114
                                 + 1.31879 w_k1_AXP_s114
                                 + 1.37106 w_k1_DIS_s114
                                 + 1.23402 w_k1_FDX_s114
                                 + 1.39329 w_k1_IBM_s114
                                 + 0.941508 w_k1_INTC_s114
                                 + 1.26866 w_k1_KO_s114 + 1.33173 w_k1_MAT_s114
                                 + 1.20598 w_k1_MSFT_s114
                                 + 1.10744 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.86977 w_k1_ADM_s114
                                 + 0.95932 w_k1_AXP_s114
                                 + 1.12526 w_k1_DIS_s114
                                 + 1.00538 w_k1_FDX_s114
                                 + 1.00034 w_k1_IBM_s114
                                 + 1.03222 w_k1_INTC_s114
                                 + 0.999562 w_k1_KO_s114
                                 + 0.898428 w_k1_MAT_s114
                                 + 0.989508 w_k1_MSFT_s114
                                 + 0.944115 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.05671 w_k1_ADM_s114
                                 + 0.81049 w_k1_AXP_s114
                                 + 0.943312 w_k1_DIS_s114
                                 + 0.871775 w_k1_FDX_s114
                                 + 1.31712 w_k1_IBM_s114
                                 + 0.727722 w_k1_INTC_s114
                                 + 1.17773 w_k1_KO_s114 + 1.10354 w_k1_MAT_s114
                                 + 0.929165 w_k1_MSFT_s114
                                 + 0.933478 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.819047 w_k1_ADM_s114
                                 + 0.658601 w_k1_AXP_s114
                                 + 0.6093 w_k1_DIS_s114
                                 + 0.772475 w_k1_FDX_s114
                                 + 0.690022 w_k1_IBM_s114
                                 + 0.781742 w_k1_INTC_s114
                                 + 0.718065 w_k1_KO_s114
                                 + 0.496267 w_k1_MAT_s114
                                 + 0.75676 w_k1_MSFT_s114
                                 + 0.671692 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.884253 w_k1_ADM_s114
                                 + 0.814816 w_k1_AXP_s114
                                 + 0.709006 w_k1_DIS_s114
                                 + 0.758251 w_k1_FDX_s114
                                 + 0.945133 w_k1_IBM_s114
                                 + 0.850097 w_k1_INTC_s114
                                 + 0.883258 w_k1_KO_s114
                                 + 0.877493 w_k1_MAT_s114
                                 + 0.772199 w_k1_MSFT_s114
                                 + 0.818895 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.883459 w_k1_ADM_s114
                                 + 1.11214 w_k1_AXP_s114
                                 + 0.911705 w_k1_DIS_s114
                                 + 0.833563 w_k1_FDX_s114
                                 + 1.59761 w_k1_IBM_s114
                                 + 0.983987 w_k1_INTC_s114
                                 + 1.02546 w_k1_KO_s114 + 1.31207 w_k1_MAT_s114
                                 + 0.907283 w_k1_MSFT_s114
                                 + 0.965278 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.04157 w_k1_ADM_s114
                                 + 0.852424 w_k1_AXP_s114
                                 + 0.906436 w_k1_DIS_s114
                                 + 1.01555 w_k1_FDX_s114
                                 + 0.651855 w_k1_IBM_s114
                                 + 0.960083 w_k1_INTC_s114
                                 + 0.819233 w_k1_KO_s114
                                 + 0.441988 w_k1_MAT_s114
                                 + 0.968174 w_k1_MSFT_s114
                                 + 0.819548 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.37731 w_k1_ADM_s114
                                 + 1.19321 w_k1_AXP_s114
                                 + 1.22066 w_k1_DIS_s114 + 1.3715 w_k1_FDX_s114
                                 + 0.599551 w_k1_IBM_s114
                                 + 1.05835 w_k1_INTC_s114
                                 + 0.857608 w_k1_KO_s114
                                 + 0.613356 w_k1_MAT_s114
                                 + 1.19036 w_k1_MSFT_s114
                                 + 0.923315 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.988868 w_k1_ADM_s115
                                 + 0.771166 w_k1_AXP_s115
                                 + 0.662607 w_k1_DIS_s115
                                 + 0.845022 w_k1_FDX_s115
                                 + 0.732988 w_k1_IBM_s115
                                 + 0.992053 w_k1_INTC_s115
                                 + 0.79194 w_k1_KO_s115
                                 + 0.658974 w_k1_MAT_s115
                                 + 0.880145 w_k1_MSFT_s115
                                 + 0.924279 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.828478 w_k1_ADM_s115
                                 + 0.90629 w_k1_AXP_s115
                                 + 0.699598 w_k1_DIS_s115
                                 + 0.805618 w_k1_FDX_s115
                                 + 1.23651 w_k1_IBM_s115
                                 + 0.788119 w_k1_INTC_s115
                                 + 0.953089 w_k1_KO_s115
                                 + 0.893247 w_k1_MAT_s115
                                 + 0.716074 w_k1_MSFT_s115
                                 + 0.931106 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.838547 w_k1_ADM_s115
                                 + 0.821331 w_k1_AXP_s115
                                 + 0.643365 w_k1_DIS_s115
                                 + 0.723256 w_k1_FDX_s115
                                 + 0.990361 w_k1_IBM_s115
                                 + 0.716532 w_k1_INTC_s115
                                 + 0.805769 w_k1_KO_s115
                                 + 0.896727 w_k1_MAT_s115
                                 + 0.729241 w_k1_MSFT_s115
                                 + 0.882434 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.782402 w_k1_ADM_s115
                                 + 1.35918 w_k1_AXP_s115
                                 + 1.16244 w_k1_DIS_s115
                                 + 1.02065 w_k1_FDX_s115
                                 + 1.59064 w_k1_IBM_s115
                                 + 0.988823 w_k1_INTC_s115
                                 + 1.18548 w_k1_KO_s115 + 1.54602 w_k1_MAT_s115
                                 + 1.03229 w_k1_MSFT_s115
                                 + 1.12593 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.12271 w_k1_ADM_s115
                                 + 1.20266 w_k1_AXP_s115
                                 + 0.953838 w_k1_DIS_s115
                                 + 1.06921 w_k1_FDX_s115
                                 + 0.997612 w_k1_IBM_s115
                                 + 1.07542 w_k1_INTC_s115
                                 + 0.936425 w_k1_KO_s115
                                 + 0.941831 w_k1_MAT_s115
                                 + 0.954208 w_k1_MSFT_s115
                                 + 1.02548 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.928664 w_k1_ADM_s115
                                 + 1.03063 w_k1_AXP_s115
                                 + 0.899889 w_k1_DIS_s115
                                 + 0.959251 w_k1_FDX_s115
                                 + 0.947535 w_k1_IBM_s115
                                 + 1.04592 w_k1_INTC_s115
                                 + 0.985858 w_k1_KO_s115
                                 + 0.930212 w_k1_MAT_s115
                                 + 0.89271 w_k1_MSFT_s115
                                 + 0.987572 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.107 w_k1_ADM_s115
                                 + 1.34951 w_k1_AXP_s115
                                 + 1.22633 w_k1_DIS_s115
                                 + 1.23304 w_k1_FDX_s115
                                 + 0.977425 w_k1_IBM_s115
                                 + 1.32488 w_k1_INTC_s115
                                 + 1.18891 w_k1_KO_s115
                                 + 0.837052 w_k1_MAT_s115
                                 + 1.14966 w_k1_MSFT_s115
                                 + 1.15147 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.51203 w_k1_ADM_s115
                                 + 1.79284 w_k1_AXP_s115
                                 + 1.96998 w_k1_DIS_s115
                                 + 1.60722 w_k1_FDX_s115
                                 + 1.18652 w_k1_IBM_s115
                                 + 1.43451 w_k1_INTC_s115
                                 + 1.35064 w_k1_KO_s115 + 1.52511 w_k1_MAT_s115
                                 + 1.48922 w_k1_MSFT_s115
                                 + 1.26336 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.86871 w_k1_ADM_s116
                                 + 1.14665 w_k1_AXP_s116
                                 + 0.900989 w_k1_DIS_s116
                                 + 1.01631 w_k1_FDX_s116
                                 + 0.988338 w_k1_IBM_s116
                                 + 1.14173 w_k1_INTC_s116
                                 + 0.874943 w_k1_KO_s116
                                 + 1.22482 w_k1_MAT_s116
                                 + 1.02962 w_k1_MSFT_s116
                                 + 0.929597 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.876267 w_k1_ADM_s116
                                 + 1.07316 w_k1_AXP_s116
                                 + 1.20068 w_k1_DIS_s116
                                 + 1.15774 w_k1_FDX_s116
                                 + 1.16075 w_k1_IBM_s116
                                 + 1.18359 w_k1_INTC_s116
                                 + 1.05925 w_k1_KO_s116 + 1.1914 w_k1_MAT_s116
                                 + 1.25039 w_k1_MSFT_s116
                                 + 1.04196 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.83175 w_k1_ADM_s116
                                 + 0.737674 w_k1_AXP_s116
                                 + 0.680132 w_k1_DIS_s116
                                 + 0.791116 w_k1_FDX_s116
                                 + 0.695536 w_k1_IBM_s116
                                 + 0.903566 w_k1_INTC_s116
                                 + 0.799613 w_k1_KO_s116
                                 + 0.921848 w_k1_MAT_s116
                                 + 0.906779 w_k1_MSFT_s116
                                 + 0.866926 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.769247 w_k1_ADM_s116
                                 + 0.738799 w_k1_AXP_s116
                                 + 0.880309 w_k1_DIS_s116
                                 + 0.860992 w_k1_FDX_s116
                                 + 0.929607 w_k1_IBM_s116
                                 + 0.881891 w_k1_INTC_s116
                                 + 0.879019 w_k1_KO_s116
                                 + 0.867184 w_k1_MAT_s116
                                 + 0.948038 w_k1_MSFT_s116
                                 + 0.941609 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.749126 w_k1_ADM_s116
                                 + 1.02086 w_k1_AXP_s116
                                 + 1.02624 w_k1_DIS_s116
                                 + 0.939985 w_k1_FDX_s116
                                 + 1.5657 w_k1_IBM_s116 + 1.1111 w_k1_INTC_s116
                                 + 1.12852 w_k1_KO_s116 + 1.77564 w_k1_MAT_s116
                                 + 0.987433 w_k1_MSFT_s116
                                 + 1.07135 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.802578 w_k1_ADM_s116
                                 + 0.779475 w_k1_AXP_s116
                                 + 0.886302 w_k1_DIS_s116
                                 + 0.902185 w_k1_FDX_s116
                                 + 1.14227 w_k1_IBM_s116
                                 + 0.793236 w_k1_INTC_s116
                                 + 0.958697 w_k1_KO_s116
                                 + 1.31565 w_k1_MAT_s116
                                 + 0.992088 w_k1_MSFT_s116
                                 + 0.974162 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.21145 w_k1_ADM_s116
                                 + 1.46421 w_k1_AXP_s116
                                 + 1.58208 w_k1_DIS_s116
                                 + 1.42127 w_k1_FDX_s116
                                 + 1.27727 w_k1_IBM_s116
                                 + 1.33725 w_k1_INTC_s116
                                 + 1.41655 w_k1_KO_s116 + 2.05582 w_k1_MAT_s116
                                 + 1.21747 w_k1_MSFT_s116
                                 + 1.29061 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.948867 w_k1_ADM_s116
                                 + 1.01105 w_k1_AXP_s116
                                 + 1.12657 w_k1_DIS_s116
                                 + 1.11012 w_k1_FDX_s116
                                 + 0.789083 w_k1_IBM_s116
                                 + 1.25828 w_k1_INTC_s116
                                 + 0.855621 w_k1_KO_s116
                                 + 0.784255 w_k1_MAT_s116
                                 + 1.16603 w_k1_MSFT_s116
                                 + 1.01606 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.09237 w_k1_ADM_s117
                                 + 0.949905 w_k1_AXP_s117
                                 + 1.16487 w_k1_DIS_s117
                                 + 1.04431 w_k1_FDX_s117
                                 + 1.02663 w_k1_IBM_s117
                                 + 1.12631 w_k1_INTC_s117
                                 + 1.08257 w_k1_KO_s117 + 1.04905 w_k1_MAT_s117
                                 + 1.09339 w_k1_MSFT_s117
                                 + 1.13825 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.917522 w_k1_ADM_s117
                                 + 1.04049 w_k1_AXP_s117
                                 + 0.857381 w_k1_DIS_s117
                                 + 0.820843 w_k1_FDX_s117
                                 + 1.7618 w_k1_IBM_s117
                                 + 1.06339 w_k1_INTC_s117 + 1.1837 w_k1_KO_s117
                                 + 1.33839 w_k1_MAT_s117
                                 + 0.888454 w_k1_MSFT_s117
                                 + 0.933626 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.14663 w_k1_ADM_s117
                                 + 1.27952 w_k1_AXP_s117 + 1.4833 w_k1_DIS_s117
                                 + 1.39618 w_k1_FDX_s117
                                 + 0.747612 w_k1_IBM_s117
                                 + 1.44578 w_k1_INTC_s117
                                 + 1.20753 w_k1_KO_s117
                                 + 0.766345 w_k1_MAT_s117
                                 + 1.3391 w_k1_MSFT_s117 + 1.25221 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.665593 w_k1_ADM_s117
                                 + 0.752062 w_k1_AXP_s117
                                 + 0.685159 w_k1_DIS_s117
                                 + 0.821656 w_k1_FDX_s117
                                 + 0.780181 w_k1_IBM_s117
                                 + 0.982762 w_k1_INTC_s117
                                 + 0.754083 w_k1_KO_s117
                                 + 0.675807 w_k1_MAT_s117
                                 + 0.897666 w_k1_MSFT_s117
                                 + 0.928687 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.840381 w_k1_ADM_s117
                                 + 0.888126 w_k1_AXP_s117
                                 + 1.00718 w_k1_DIS_s117
                                 + 1.04174 w_k1_FDX_s117
                                 + 0.72396 w_k1_IBM_s117
                                 + 1.03269 w_k1_INTC_s117
                                 + 0.996144 w_k1_KO_s117
                                 + 0.762696 w_k1_MAT_s117
                                 + 1.01152 w_k1_MSFT_s117
                                 + 0.909689 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.928132 w_k1_ADM_s117
                                 + 0.873429 w_k1_AXP_s117
                                 + 0.794003 w_k1_DIS_s117
                                 + 0.730412 w_k1_FDX_s117
                                 + 1.40935 w_k1_IBM_s117
                                 + 0.898298 w_k1_INTC_s117
                                 + 1.16507 w_k1_KO_s117 + 1.08913 w_k1_MAT_s117
                                 + 0.841964 w_k1_MSFT_s117
                                 + 1.00187 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.632355 w_k1_ADM_s117
                                 + 0.652834 w_k1_AXP_s117
                                 + 0.73528 w_k1_DIS_s117
                                 + 0.733277 w_k1_FDX_s117
                                 + 1.10481 w_k1_IBM_s117
                                 + 0.997419 w_k1_INTC_s117
                                 + 0.924044 w_k1_KO_s117
                                 + 0.925864 w_k1_MAT_s117
                                 + 0.688591 w_k1_MSFT_s117
                                 + 0.925545 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.71441 w_k1_ADM_s117
                                 + 1.16149 w_k1_AXP_s117
                                 + 1.60164 w_k1_DIS_s117
                                 + 1.26141 w_k1_FDX_s117
                                 + 1.55149 w_k1_IBM_s117
                                 + 1.6057 w_k1_INTC_s117 + 1.41177 w_k1_KO_s117
                                 + 1.68846 w_k1_MAT_s117
                                 + 1.27051 w_k1_MSFT_s117
                                 + 1.15184 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0177003747957308 w_k1_ADM_s000
                                 + 0.0297596718312096 w_k1_ADM_s001
                                 + 0.0592570277358765 w_k1_ADM_s002
                                 + 0.0199016634156218 w_k1_ADM_s003
                                 + 0.0111641235185932 w_k1_ADM_s004
                                 + 0.0235408794691318 w_k1_ADM_s005
                                 + 0.0437094691769814 w_k1_ADM_s006
                                 + 0.0162731975854051 w_k1_ADM_s007
                                 + 0.0380259902252762 w_k1_ADM_s010
                                 + 0.0186305195382592 w_k1_ADM_s011
                                 + 0.0147036724041576 w_k1_ADM_s012
                                 + 0.0118138968425682 w_k1_ADM_s013
                                 + 0.0313755293897255 w_k1_ADM_s014
                                 + 0.0376651798471636 w_k1_ADM_s015
                                 + 0.0432363549895346 w_k1_ADM_s016
                                 + 0.0139964952121367 w_k1_ADM_s017
                                 + 0.0182534850200254 w_k1_ADM_s100
                                 + 0.00564505800496989 w_k1_ADM_s101
                                 + 0.0191778702204891 w_k1_ADM_s102
                                 + 0.0269935413974928 w_k1_ADM_s103
                                 + 0.00536110336762155 w_k1_ADM_s104
                                 + 0.025186105332803 w_k1_ADM_s105
                                 + 0.0172820504532626 w_k1_ADM_s106
                                 + 0.021774637321719 w_k1_ADM_s107
                                 + 0.086612139989388 w_k1_ADM_s110
                                 + 0.0274749789628599 w_k1_ADM_s111
                                 + 0.0366106839728773 w_k1_ADM_s112
                                 + 0.0465655705414717 w_k1_ADM_s113
                                 + 0.117569148853823 w_k1_ADM_s114
                                 + 0.0958854935279264 w_k1_ADM_s115
                                 + 0.0234726099055871 w_k1_ADM_s116
                                 + 0.0309101654140706 w_k1_ADM_s117
                                 + 0.0169217589950759 w_k1_AXP_s000
                                 + 0.0249292312271421 w_k1_AXP_s001
                                 + 0.0474482761765343 w_k1_AXP_s002
                                 + 0.0193179375650366 w_k1_AXP_s003
                                 + 0.0134301812467793 w_k1_AXP_s004
                                 + 0.0230877983530888 w_k1_AXP_s005
                                 + 0.0395525050363462 w_k1_AXP_s006
                                 + 0.0161103654934851 w_k1_AXP_s007
                                 + 0.0389020634543311 w_k1_AXP_s010
                                 + 0.0203269558086446 w_k1_AXP_s011
                                 + 0.0137167841579663 w_k1_AXP_s012
                                 + 0.0131204444009033 w_k1_AXP_s013
                                 + 0.0233930339831674 w_k1_AXP_s014
                                 + 0.0389939751154342 w_k1_AXP_s015
                                 + 0.0421279936860857 w_k1_AXP_s016
                                 + 0.0127786284112143 w_k1_AXP_s017
                                 + 0.0163763139370993 w_k1_AXP_s100
                                 + 0.00587292947883545 w_k1_AXP_s101
                                 + 0.0214101200647868 w_k1_AXP_s102
                                 + 0.0246219029387084 w_k1_AXP_s103
                                 + 0.00591438456251801 w_k1_AXP_s104
                                 + 0.02584830136683 w_k1_AXP_s105
                                 + 0.0151976061544944 w_k1_AXP_s106
                                 + 0.0230185118396879 w_k1_AXP_s107
                                 + 0.0901427138091343 w_k1_AXP_s110
                                 + 0.02833613027862 w_k1_AXP_s111
                                 + 0.0407515430266494 w_k1_AXP_s112
                                 + 0.0458508422407298 w_k1_AXP_s113
                                 + 0.116637156503799 w_k1_AXP_s114
                                 + 0.108501414873514 w_k1_AXP_s115
                                 + 0.025978840061543 w_k1_AXP_s116
                                 + 0.0328259060826778 w_k1_AXP_s117
                                 + 0.0183237462091014 w_k1_DIS_s000
                                 + 0.0262762353057542 w_k1_DIS_s001
                                 + 0.052647438374036 w_k1_DIS_s002
                                 + 0.017927012190956 w_k1_DIS_s003
                                 + 0.0126078232233937 w_k1_DIS_s004
                                 + 0.026412673763215 w_k1_DIS_s005
                                 + 0.0405435031353089 w_k1_DIS_s006
                                 + 0.0164458603750634 w_k1_DIS_s007
                                 + 0.0357046268478539 w_k1_DIS_s010
                                 + 0.0211110515756937 w_k1_DIS_s011
                                 + 0.0136122415456944 w_k1_DIS_s012
                                 + 0.0123552821566925 w_k1_DIS_s013
                                 + 0.0231812616378743 w_k1_DIS_s014
                                 + 0.036355356689713 w_k1_DIS_s015
                                 + 0.0433749448246418 w_k1_DIS_s016
                                 + 0.0124304176576199 w_k1_DIS_s017
                                 + 0.016879847216029 w_k1_DIS_s100
                                 + 0.00658040791587188 w_k1_DIS_s101
                                 + 0.0203125832698468 w_k1_DIS_s102
                                 + 0.0239268105744784 w_k1_DIS_s103
                                 + 0.00621045259860306 w_k1_DIS_s104
                                 + 0.0235174569437934 w_k1_DIS_s105
                                 + 0.0176674476125683 w_k1_DIS_s106
                                 + 0.0258913824944565 w_k1_DIS_s107
                                 + 0.0922921778878593 w_k1_DIS_s110
                                 + 0.028328506993765 w_k1_DIS_s111
                                 + 0.0423527467352592 w_k1_DIS_s112
                                 + 0.0519909586244554 w_k1_DIS_s113
                                 + 0.119810565877463 w_k1_DIS_s114
                                 + 0.0954335204281289 w_k1_DIS_s115
                                 + 0.0274472929402246 w_k1_DIS_s116
                                 + 0.0368220008161926 w_k1_DIS_s117
                                 + 0.0172542603886452 w_k1_FDX_s000
                                 + 0.0251107821577472 w_k1_FDX_s001
                                 + 0.0506139354329606 w_k1_FDX_s002
                                 + 0.0183931302506082 w_k1_FDX_s003
                                 + 0.0129221505468594 w_k1_FDX_s004
                                 + 0.0239009672647618 w_k1_FDX_s005
                                 + 0.0381987629247523 w_k1_FDX_s006
                                 + 0.0160930915717745 w_k1_FDX_s007
                                 + 0.0366707140237744 w_k1_FDX_s010
                                 + 0.0198807114792951 w_k1_FDX_s011
                                 + 0.0132717152254084 w_k1_FDX_s012
                                 + 0.0130143153313379 w_k1_FDX_s013
                                 + 0.0227894053083664 w_k1_FDX_s014
                                 + 0.0356066245735413 w_k1_FDX_s015
                                 + 0.0438017824164301 w_k1_FDX_s016
                                 + 0.0117804559910474 w_k1_FDX_s017
                                 + 0.0166631790630677 w_k1_FDX_s100
                                 + 0.00621877963106173 w_k1_FDX_s101
                                 + 0.0206747534988741 w_k1_FDX_s102
                                 + 0.024161094340891 w_k1_FDX_s103
                                 + 0.00604873801214291 w_k1_FDX_s104
                                 + 0.0241784780509301 w_k1_FDX_s105
                                 + 0.0162597821516317 w_k1_FDX_s106
                                 + 0.0236828559281289 w_k1_FDX_s107
                                 + 0.0874114352904186 w_k1_FDX_s110
                                 + 0.0269491749665924 w_k1_FDX_s111
                                 + 0.0392510115887636 w_k1_FDX_s112
                                 + 0.0435856907201725 w_k1_FDX_s113
                                 + 0.116358004832091 w_k1_FDX_s114
                                 + 0.0984281695310841 w_k1_FDX_s115
                                 + 0.027340969970973 w_k1_FDX_s116
                                 + 0.0349449267911408 w_k1_FDX_s117
                                 + 0.0179824789928817 w_k1_IBM_s000
                                 + 0.0257703052086062 w_k1_IBM_s001
                                 + 0.052318895232269 w_k1_IBM_s002
                                 + 0.0178518977846044 w_k1_IBM_s003
                                 + 0.0131027256219244 w_k1_IBM_s004
                                 + 0.0233821596335049 w_k1_IBM_s005
                                 + 0.0407163019793528 w_k1_IBM_s006
                                 + 0.0155402286558617 w_k1_IBM_s007
                                 + 0.0368684522124093 w_k1_IBM_s010
                                 + 0.0193090782650786 w_k1_IBM_s011
                                 + 0.013384578495826 w_k1_IBM_s012
                                 + 0.0125835787398156 w_k1_IBM_s013
                                 + 0.0222044099485777 w_k1_IBM_s014
                                 + 0.035816453585146 w_k1_IBM_s015
                                 + 0.0433984369378113 w_k1_IBM_s016
                                 + 0.0121503423915202 w_k1_IBM_s017
                                 + 0.0173856290167976 w_k1_IBM_s100
                                 + 0.00681885129147493 w_k1_IBM_s101
                                 + 0.0205757868437045 w_k1_IBM_s102
                                 + 0.0244567450380093 w_k1_IBM_s103
                                 + 0.00599166775169955 w_k1_IBM_s104
                                 + 0.0250949084868847 w_k1_IBM_s105
                                 + 0.016493209812333 w_k1_IBM_s106
                                 + 0.0251474487436557 w_k1_IBM_s107
                                 + 0.0904400711042167 w_k1_IBM_s110
                                 + 0.0284590969998645 w_k1_IBM_s111
                                 + 0.0410252592641243 w_k1_IBM_s112
                                 + 0.0463817264656915 w_k1_IBM_s113
                                 + 0.123288802837702 w_k1_IBM_s114
                                 + 0.100322833897798 w_k1_IBM_s115
                                 + 0.028090735645736 w_k1_IBM_s116
                                 + 0.0362680303409217 w_k1_IBM_s117
                                 + 0.0165953282854128 w_k1_INTC_s000
                                 + 0.0252737099681391 w_k1_INTC_s001
                                 + 0.045202415902047 w_k1_INTC_s002
                                 + 0.0194664935714075 w_k1_INTC_s003
                                 + 0.0141688493413023 w_k1_INTC_s004
                                 + 0.0227476891235947 w_k1_INTC_s005
                                 + 0.0329210915735343 w_k1_INTC_s006
                                 + 0.0149346159645061 w_k1_INTC_s007
                                 + 0.0388316869083557 w_k1_INTC_s010
                                 + 0.0223183885726907 w_k1_INTC_s011
                                 + 0.0124407599681315 w_k1_INTC_s012
                                 + 0.0129305171304842 w_k1_INTC_s013
                                 + 0.0234174323998256 w_k1_INTC_s014
                                 + 0.0381996044163098 w_k1_INTC_s015
                                 + 0.0437087705709759 w_k1_INTC_s016
                                 + 0.0116717956384205 w_k1_INTC_s017
                                 + 0.0179149327440437 w_k1_INTC_s100
                                 + 0.00608254021031518 w_k1_INTC_s101
                                 + 0.0218924661621587 w_k1_INTC_s102
                                 + 0.0261584522711013 w_k1_INTC_s103
                                 + 0.00521313217577292 w_k1_INTC_s104
                                 + 0.0230773079924203 w_k1_INTC_s105
                                 + 0.0156834928183723 w_k1_INTC_s106
                                 + 0.0248891315780842 w_k1_INTC_s107
                                 + 0.0978275225054558 w_k1_INTC_s110
                                 + 0.0311832844953237 w_k1_INTC_s111
                                 + 0.041681904419782 w_k1_INTC_s112
                                 + 0.0421271459087478 w_k1_INTC_s113
                                 + 0.106006376745754 w_k1_INTC_s114
                                 + 0.103155721320847 w_k1_INTC_s115
                                 + 0.0286467679339825 w_k1_INTC_s116
                                 + 0.0395733030094715 w_k1_INTC_s117
                                 + 0.0179798502936618 w_k1_KO_s000
                                 + 0.0261451407264438 w_k1_KO_s001
                                 + 0.0482797071990595 w_k1_KO_s002
                                 + 0.0183728530125079 w_k1_KO_s003
                                 + 0.0122827471390107 w_k1_KO_s004
                                 + 0.0242580919648369 w_k1_KO_s005
                                 + 0.0362791202333625 w_k1_KO_s006
                                 + 0.0166549959331832 w_k1_KO_s007
                                 + 0.0361156399582665 w_k1_KO_s010
                                 + 0.0232687945011571 w_k1_KO_s011
                                 + 0.0138592392269998 w_k1_KO_s012
                                 + 0.0128573994763872 w_k1_KO_s013
                                 + 0.02213605374621 w_k1_KO_s014
                                 + 0.0405418443647691 w_k1_KO_s015
                                 + 0.0402248694940388 w_k1_KO_s016
                                 + 0.0116882987182932 w_k1_KO_s017
                                 + 0.014750803985821 w_k1_KO_s100
                                 + 0.0070155425384904 w_k1_KO_s101
                                 + 0.0210256988268802 w_k1_KO_s102
                                 + 0.0239215723067812 w_k1_KO_s103
                                 + 0.0055858880681981 w_k1_KO_s104
                                 + 0.0215978015369891 w_k1_KO_s105
                                 + 0.0175050424390353 w_k1_KO_s106
                                 + 0.0248879690813129 w_k1_KO_s107
                                 + 0.0972995115657533 w_k1_KO_s110
                                 + 0.0285426098703853 w_k1_KO_s111
                                 + 0.0400372702984823 w_k1_KO_s112
                                 + 0.0459597279152484 w_k1_KO_s113
                                 + 0.116866897649743 w_k1_KO_s114
                                 + 0.0978974904206431 w_k1_KO_s115
                                 + 0.026299642389982 w_k1_KO_s116
                                 + 0.0369328501651936 w_k1_KO_s117
                                 + 0.0192122783813494 w_k1_MAT_s000
                                 + 0.0257648591112404 w_k1_MAT_s001
                                 + 0.0517289992377807 w_k1_MAT_s002
                                 + 0.0162793301627988 w_k1_MAT_s003
                                 + 0.012642173504575 w_k1_MAT_s004
                                 + 0.0230488452818595 w_k1_MAT_s005
                                 + 0.0441823606619603 w_k1_MAT_s006
                                 + 0.0118502681820438 w_k1_MAT_s007
                                 + 0.0344636786606324 w_k1_MAT_s010
                                 + 0.0213431999047558 w_k1_MAT_s011
                                 + 0.0118154501725497 w_k1_MAT_s012
                                 + 0.0125309044537675 w_k1_MAT_s013
                                 + 0.0184238981033853 w_k1_MAT_s014
                                 + 0.0331188331091471 w_k1_MAT_s015
                                 + 0.0452440745807321 w_k1_MAT_s016
                                 + 0.0115830659538778 w_k1_MAT_s017
                                 + 0.0145014555276649 w_k1_MAT_s100
                                 + 0.00711084734279564 w_k1_MAT_s101
                                 + 0.024475181984897 w_k1_MAT_s102
                                 + 0.0225793819863498 w_k1_MAT_s103
                                 + 0.00527074215374709 w_k1_MAT_s104
                                 + 0.0177870994177075 w_k1_MAT_s105
                                 + 0.0156236511283674 w_k1_MAT_s106
                                 + 0.0209216148085083 w_k1_MAT_s107
                                 + 0.108906933346245 w_k1_MAT_s110
                                 + 0.0323598983606422 w_k1_MAT_s111
                                 + 0.0458945239759019 w_k1_MAT_s112
                                 + 0.0471603301931564 w_k1_MAT_s113
                                 + 0.110365089632485 w_k1_MAT_s114
                                 + 0.0937148274811298 w_k1_MAT_s115
                                 + 0.032039134685404 w_k1_MAT_s116
                                 + 0.0343793239038741 w_k1_MAT_s117
                                 + 0.0178207828559575 w_k1_MSFT_s000
                                 + 0.025943671510021 w_k1_MSFT_s001
                                 + 0.0466829819791249 w_k1_MSFT_s002
                                 + 0.0185953327862426 w_k1_MSFT_s003
                                 + 0.0126117159755838 w_k1_MSFT_s004
                                 + 0.0241148422505571 w_k1_MSFT_s005
                                 + 0.0376715542592023 w_k1_MSFT_s006
                                 + 0.0138379009720372 w_k1_MSFT_s007
                                 + 0.0368712755077961 w_k1_MSFT_s010
                                 + 0.0201036482523982 w_k1_MSFT_s011
                                 + 0.0135654291300933 w_k1_MSFT_s012
                                 + 0.0139857899306012 w_k1_MSFT_s013
                                 + 0.0218592053184923 w_k1_MSFT_s014
                                 + 0.0382658559383405 w_k1_MSFT_s015
                                 + 0.0425985370810117 w_k1_MSFT_s016
                                 + 0.0113240185010322 w_k1_MSFT_s017
                                 + 0.0176081211547922 w_k1_MSFT_s100
                                 + 0.00665354470960376 w_k1_MSFT_s101
                                 + 0.0219150808549 w_k1_MSFT_s102
                                 + 0.022198318011157 w_k1_MSFT_s103
                                 + 0.00567046311796676 w_k1_MSFT_s104
                                 + 0.0241594742437914 w_k1_MSFT_s105
                                 + 0.015905924169002 w_k1_MSFT_s106
                                 + 0.0223065770187641 w_k1_MSFT_s107
                                 + 0.088980640677795 w_k1_MSFT_s110
                                 + 0.0291095859211381 w_k1_MSFT_s111
                                 + 0.0397001420989567 w_k1_MSFT_s112
                                 + 0.0419120946028793 w_k1_MSFT_s113
                                 + 0.114291739688087 w_k1_MSFT_s114
                                 + 0.0946907623025855 w_k1_MSFT_s115
                                 + 0.0289040455127114 w_k1_MSFT_s116
                                 + 0.0358533751597962 w_k1_MSFT_s117
                                 + 0.0172153531748315 w_k1_PG_s000
                                 + 0.02430253482851 w_k1_PG_s001
                                 + 0.0504739755010165 w_k1_PG_s002
                                 + 0.0174520671742684 w_k1_PG_s003
                                 + 0.0133316324964177 w_k1_PG_s004
                                 + 0.0219795283974368 w_k1_PG_s005
                                 + 0.036380923538569 w_k1_PG_s006
                                 + 0.0148822887394203 w_k1_PG_s007
                                 + 0.0354425363505368 w_k1_PG_s010
                                 + 0.0197199303273378 w_k1_PG_s011
                                 + 0.0126512685671424 w_k1_PG_s012
                                 + 0.0122376226632613 w_k1_PG_s013
                                 + 0.0245181219222711 w_k1_PG_s014
                                 + 0.0332720868071223 w_k1_PG_s015
                                 + 0.0445462980024893 w_k1_PG_s016
                                 + 0.0117644370692844 w_k1_PG_s017
                                 + 0.0159980738544237 w_k1_PG_s100
                                 + 0.0066553380176056 w_k1_PG_s101
                                 + 0.0209111354530508 w_k1_PG_s102
                                 + 0.0244588330674172 w_k1_PG_s103
                                 + 0.00558628089113748 w_k1_PG_s104
                                 + 0.0230289906684455 w_k1_PG_s105
                                 + 0.0158071351915635 w_k1_PG_s106
                                 + 0.0248311370645895 w_k1_PG_s107
                                 + 0.0954181311485664 w_k1_PG_s110
                                 + 0.026357871706321 w_k1_PG_s111
                                 + 0.0425565241742918 w_k1_PG_s112
                                 + 0.0443618040651669 w_k1_PG_s113
                                 + 0.106994861741332 w_k1_PG_s114
                                 + 0.0998758673343896 w_k1_PG_s115
                                 + 0.0270978387316253 w_k1_PG_s116
                                 + 0.0361020598703964 w_k1_PG_s117 >= 1.07
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
