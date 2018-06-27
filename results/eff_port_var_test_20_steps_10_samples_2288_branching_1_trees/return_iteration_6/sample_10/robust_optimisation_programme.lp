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
 w_k1_ADM_s0_return_constraint:  1.09942 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.895082 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.07502 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.968606 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.03393 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.16436 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.977239 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.28821 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.16784 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.770011 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.762445 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.2489 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.625744 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.00058 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.14042 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.980859 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.779696 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.687416 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.925999 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.86018 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.44472 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.984895 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.82836 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.82492 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.967449 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.09125 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.560836 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.34277 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.20372 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.789458 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.890955 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.01824 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.966431 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.1482 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.753086 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.751855 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.87383 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.826109 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.19988 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.819312 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.05875 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.938144 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.01458 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.34961 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.03604 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.41464 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.50031 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.72594 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.835227 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.58857 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.533224 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.29838 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.19902 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.845097 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.712251 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.501329 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.01573 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.79189 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.41411 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.744295 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.711796 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.71992 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.10655 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.06677 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.511949 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.7858 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.29414 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.05043 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.956661 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.45579 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.00361 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.2659 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.754548 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.567266 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.747873 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.00675 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.19112 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.815769 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.10586 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.990482 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.940439 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.32785 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.10291 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.65818 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.09134 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.760031 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.721468 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.90632 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.445411 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.29612 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.346 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.729042 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.58721 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.481602 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.99557 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 2.26626 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.38747 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.900039 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.690064 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.770195 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.27829 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.26895 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.477942 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.76026 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.38905 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.925645 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.02915 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.65739 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.01619 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.41262 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.68562 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.51297 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.730858 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.866101 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.1311 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.862669 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.00335 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.03279 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.00069 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.12953 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.05088 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.4422 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.2299 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.913392 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.790158 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.49783 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.632458 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.07483 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.0935 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.892842 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.710171 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.61359 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.00834 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.67198 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.41748 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.830583 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.842721 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.83106 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.06342 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.082 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.558524 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.45888 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.30513 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.878203 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.01893 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.27358 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.915827 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.36782 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.738913 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.62657 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.832165 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.843922 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13293 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.901303 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.21635 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.715717 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.908129 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.54985 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.930003 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.976142 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.02621 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.593276 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.03655 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.52414 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.618676 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.86449 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.32617 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.720979 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.791759 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.905134 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.891114 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.57156 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.997784 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.08569 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.687118 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.14209 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.70593 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.12706 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.783528 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.58433 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.943263 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.163 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.825623 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.40755 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.21871 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.902936 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.04129 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.733314 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.716165 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.64865 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10426 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.844609 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.96578 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.03017 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.02453 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.13868 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.989345 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.39666 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.31818 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.918952 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.779079 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.52689 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.71417 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.14869 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.11798 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.762428 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.787425 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.732842 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.993732 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.18901 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.19515 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.891003 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.798453 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.921739 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.2184 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.0478 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.59848 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.28062 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.01902 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.96845 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.12404 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.45861 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.0639 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.37073 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.7745 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.783506 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.94011 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.1001 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18512 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.865035 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.01631 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.827713 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.909644 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.36531 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.06407 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.37095 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.11312 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.720754 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.939509 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.72181 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.578881 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.59271 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.36467 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.777252 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.691626 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.69096 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.97418 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.65437 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.1799 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.05993 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.746355 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.977771 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.47126 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.23699 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.640679 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.56501 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.22759 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.1011 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.02569 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.35859 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.10034 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.0741 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.832775 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.69076 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.805221 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.15016 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.19182 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.884123 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.18064 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.763946 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.937894 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.94917 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.910943 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.05345 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.40385 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.464469 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.982722 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.66159 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.575486 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 2.09213 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.39387 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.707994 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.640426 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.755783 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.04272 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.92128 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.31202 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.874895 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.55293 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.804429 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.49319 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.15592 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.589427 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.54911 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.15385 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.27468 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.745701 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.15944 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.973499 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.693463 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.737991 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.411051 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.570167 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.22181 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.90248 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.05806 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.01917 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.917395 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.0926 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.971344 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.43808 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.10954 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.861404 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.784592 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.42334 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.647448 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.01501 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.17008 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.800523 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.742273 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.660728 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.02816 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.55392 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.23858 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.923119 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.892628 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.967604 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.25741 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.14379 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.609027 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.32472 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.32181 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.979441 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.07339 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.32704 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.06699 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.35762 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.696453 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.694634 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.907195 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.995719 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.07645 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.924389 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.00371 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.965865 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.965708 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.18692 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.10738 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.36764 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.13458 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.822819 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.987105 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.46533 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.705722 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.15758 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.09556 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.899357 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.821984 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.71701 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.0017 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.36227 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.16527 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.899326 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.839 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.985547 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.21677 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.06682 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.7547 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.21606 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.09238 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.03558 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.01239 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.17242 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.939281 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.01416 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.847528 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.689682 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.844694 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.02777 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.530000576931217 z_k1_s0000
                                 - 0.554001007496696 z_k1_s0001
                                 - 0.775000903542917 z_k1_s0002
                                 - 1.41600238022517 z_k1_s0003
                                 - 1.21900370864574 z_k1_s0004
                                 - 0.961001660246109 z_k1_s0005
                                 - 1.67300411297157 z_k1_s0006
                                 - 0.81100194634179 z_k1_s0007
                                 - 0.564999664694828 z_k1_s0010
                                 - 0.566000482479953 z_k1_s0011
                                 - 0.374999427487793 z_k1_s0012
                                 - 0.196999877704215 z_k1_s0013
                                 - 0.27299997258703 z_k1_s0014
                                 - 0.264999818504659 z_k1_s0015
                                 - 0.385999297126126 z_k1_s0016
                                 - 0.414998676899972 z_k1_s0017
                                 - 0.165000010295779 z_k1_s0020
                                 - 0.213999961026056 z_k1_s0021
                                 - 0.521999980244843 z_k1_s0022
                                 - 0.210999934675231 z_k1_s0023
                                 - 0.325000936006579 z_k1_s0024
                                 - 0.342001277194532 z_k1_s0025
                                 - 0.731998946803404 z_k1_s0026
                                 - 0.366999578118616 z_k1_s0027
                                 - 0.252999174804369 z_k1_s0030
                                 - 0.30100054666732 z_k1_s0031
                                 - 0.608998471212553 z_k1_s0032
                                 - 0.152999970203879 z_k1_s0033
                                 - 0.199999886999897 z_k1_s0034
                                 - 0.320000114759618 z_k1_s0035
                                 - 0.179999849039943 z_k1_s0036
                                 - 0.446999690795811 z_k1_s0037
                                 - 2.2029965356234 z_k1_s0040
                                 - 1.16500356848047 z_k1_s0041
                                 - 1.01600084215646 z_k1_s0042
                                 - 0.774000227259724 z_k1_s0043
                                 - 2.0890065110865 z_k1_s0044
                                 - 1.04700050184756 z_k1_s0045
                                 - 1.16900570557078 z_k1_s0046
                                 - 1.44200161815634 z_k1_s0047
                                 - 0.340000711587429 z_k1_s0050
                                 - 0.52699886346169 z_k1_s0051
                                 - 0.220000003756811 z_k1_s0052
                                 - 0.178999856471301 z_k1_s0053
                                 - 0.353999072241904 z_k1_s0054
                                 - 0.240999933537271 z_k1_s0055
                                 - 0.609000352432084 z_k1_s0056
                                 - 0.515998445234068 z_k1_s0057
                                 - 0.176000109439642 z_k1_s0060
                                 - 0.411999704831474 z_k1_s0061
                                 - 0.194999954994502 z_k1_s0062
                                 - 0.420000403265045 z_k1_s0063
                                 - 0.36300011630052 z_k1_s0064
                                 - 0.102999926207869 z_k1_s0065
                                 - 0.443999997565935 z_k1_s0066
                                 - 0.388000110530584 z_k1_s0067
                                 - 0.565000312200993 z_k1_s0070
                                 - 0.437001043238362 z_k1_s0071
                                 - 0.30700061897985 z_k1_s0072
                                 - 0.195999969536145 z_k1_s0073
                                 - 0.33800110307213 z_k1_s0074
                                 - 0.311000440572487 z_k1_s0075
                                 - 0.187999828551047 z_k1_s0076
                                 - 0.147000039377087 z_k1_s0077
                                 - 0.295000538246936 z_k1_s0100
                                 - 0.365000740458242 z_k1_s0101
                                 - 0.941002391811243 z_k1_s0102
                                 - 0.325000876052349 z_k1_s0103
                                 - 0.41100111192621 z_k1_s0104
                                 - 0.764999824018163 z_k1_s0105
                                 - 0.459001652414871 z_k1_s0106
                                 - 0.834001479972599 z_k1_s0107
                                 - 0.264002117248207 z_k1_s0110
                                 - 0.424002480061466 z_k1_s0111
                                 - 0.29100040676556 z_k1_s0112
                                 - 0.28200013325626 z_k1_s0113
                                 - 0.109000500267588 z_k1_s0114
                                 - 0.525002455986281 z_k1_s0115
                                 - 0.152000597773651 z_k1_s0116
                                 - 0.484002616116438 z_k1_s0117
                                 - 0.457998255851622 z_k1_s0120
                                 - 0.359999676941811 z_k1_s0121
                                 - 0.371997996077501 z_k1_s0122
                                 - 0.210999539627703 z_k1_s0123
                                 - 0.147999662087716 z_k1_s0124
                                 - 0.231999498807698 z_k1_s0125
                                 - 0.339998083393633 z_k1_s0126
                                 - 0.515998921651512 z_k1_s0127
                                 - 0.147999733629204 z_k1_s0130
                                 - 0.382000162015984 z_k1_s0131
                                 - 0.194000105243721 z_k1_s0132
                                 - 0.204000186029486 z_k1_s0133
                                 - 0.12199996218625 z_k1_s0134
                                 - 0.101999946814732 z_k1_s0135
                                 - 0.109000003364767 z_k1_s0136
                                 - 0.201000161793757 z_k1_s0137
                                 - 0.270999970853964 z_k1_s0140
                                 - 0.714000001762896 z_k1_s0141
                                 - 0.213000332219407 z_k1_s0142
                                 - 0.450002013633436 z_k1_s0143
                                 - 0.709001753390389 z_k1_s0144
                                 - 0.747000859029827 z_k1_s0145
                                 - 0.736999927281819 z_k1_s0146
                                 - 0.594002125795776 z_k1_s0147
                                 - 0.229999703247417 z_k1_s0150
                                 - 0.0670000499807961 z_k1_s0151
                                 - 0.113999542426919 z_k1_s0152
                                 - 0.110000469640189 z_k1_s0153
                                 - 0.0639999879905994 z_k1_s0154
                                 - 0.161000549673341 z_k1_s0155
                                 - 0.111000237836872 z_k1_s0156
                                 - 0.224999780263791 z_k1_s0157
                                 - 0.356999336185338 z_k1_s0160
                                 - 0.421997934149841 z_k1_s0161
                                 - 0.299999783818655 z_k1_s0162
                                 - 0.116999763229702 z_k1_s0163
                                 - 0.40999936575311 z_k1_s0164
                                 - 0.308999980612645 z_k1_s0165
                                 - 0.375998339976604 z_k1_s0166
                                 - 0.249999396350065 z_k1_s0167
                                 - 0.289000124351909 z_k1_s0170
                                 - 0.442998415091647 z_k1_s0171
                                 - 0.369998637327229 z_k1_s0172
                                 - 0.251998967023195 z_k1_s0173
                                 - 0.102999777990606 z_k1_s0174
                                 - 0.133999512318492 z_k1_s0175
                                 - 0.137999526721939 z_k1_s0176
                                 - 0.426998788676521 z_k1_s0177
                                 - 1.97099495694663 z_k1_s1000
                                 - 0.734998170375267 z_k1_s1001
                                 - 0.433999145214425 z_k1_s1002
                                 - 0.909001440715985 z_k1_s1003
                                 - 0.792998443790631 z_k1_s1004
                                 - 1.38399311914083 z_k1_s1005
                                 - 1.54299609347462 z_k1_s1006
                                 - 0.397999088156366 z_k1_s1007
                                 - 0.688999977648476 z_k1_s1010
                                 - 0.873003946701491 z_k1_s1011
                                 - 0.931004142161462 z_k1_s1012
                                 - 0.301000813431461 z_k1_s1013
                                 - 0.613003405398933 z_k1_s1014
                                 - 0.54499982726586 z_k1_s1015
                                 - 0.468000974344604 z_k1_s1016
                                 - 0.436000833014889 z_k1_s1017
                                 - 0.591998503547517 z_k1_s1020
                                 - 0.329000361829293 z_k1_s1021
                                 - 0.313999886994951 z_k1_s1022
                                 - 0.582999984986433 z_k1_s1023
                                 - 0.534000418267045 z_k1_s1024
                                 - 0.545000788285557 z_k1_s1025
                                 - 0.193000069090171 z_k1_s1026
                                 - 0.181000081742591 z_k1_s1027
                                 - 0.327001427909734 z_k1_s1030
                                 - 0.663000412955389 z_k1_s1031
                                 - 0.184000043429573 z_k1_s1032
                                 - 0.13499985062719 z_k1_s1033
                                 - 0.657000992865109 z_k1_s1034
                                 - 0.370000942222238 z_k1_s1035
                                 - 0.540001290108158 z_k1_s1036
                                 - 0.270000015854279 z_k1_s1037
                                 - 0.328000483558739 z_k1_s1040
                                 - 0.253999822213115 z_k1_s1041
                                 - 0.0929999197315721 z_k1_s1042
                                 - 0.379000470621217 z_k1_s1043
                                 - 0.210000626845566 z_k1_s1044
                                 - 0.140000032497006 z_k1_s1045
                                 - 0.169999956874927 z_k1_s1046
                                 - 0.352999457040245 z_k1_s1047
                                 - 0.11400004202891 z_k1_s1050
                                 - 0.13700004018036 z_k1_s1051
                                 - 0.378999911034772 z_k1_s1052
                                 - 0.194000145294792 z_k1_s1053
                                 - 0.439000198734126 z_k1_s1054
                                 - 0.127999997025457 z_k1_s1055
                                 - 0.169999693815142 z_k1_s1056
                                 - 0.121000019527183 z_k1_s1057
                                 - 1.36399878674543 z_k1_s1060
                                 - 0.932997577623792 z_k1_s1061
                                 - 1.13599604404203 z_k1_s1062
                                 - 0.516999750872732 z_k1_s1063
                                 - 1.66800244368329 z_k1_s1064
                                 - 0.557999577904902 z_k1_s1065
                                 - 0.829999884356432 z_k1_s1066
                                 - 1.15999590295845 z_k1_s1067
                                 - 0.220999933211928 z_k1_s1070
                                 - 0.711001497458026 z_k1_s1071
                                 - 0.255999756158274 z_k1_s1072
                                 - 0.785999096558711 z_k1_s1073
                                 - 0.563998117226172 z_k1_s1074
                                 - 0.288999912661752 z_k1_s1075
                                 - 0.591001295899675 z_k1_s1076
                                 - 0.624998859826233 z_k1_s1077
                                 - 0.242001074102442 z_k1_s1100
                                 - 0.332000994097833 z_k1_s1101
                                 - 0.113000386940065 z_k1_s1102
                                 - 0.278000193098146 z_k1_s1103
                                 - 0.0860002694410394 z_k1_s1104
                                 - 0.195000298324709 z_k1_s1105
                                 - 0.0740001857747148 z_k1_s1106
                                 - 0.0950002614405785 z_k1_s1107
                                 - 0.237999613406589 z_k1_s1110
                                 - 0.115999773189772 z_k1_s1111
                                 - 0.105000030973462 z_k1_s1112
                                 - 0.132999582433126 z_k1_s1113
                                 - 0.193999502541535 z_k1_s1114
                                 - 0.14000004129795 z_k1_s1115
                                 - 0.138999649460171 z_k1_s1116
                                 - 0.0770000227138724 z_k1_s1117
                                 - 0.221000757171584 z_k1_s1120
                                 - 0.186000340079694 z_k1_s1121
                                 - 0.0670003112736617 z_k1_s1122
                                 - 0.159001064813624 z_k1_s1123
                                 - 0.219001418969489 z_k1_s1124
                                 - 0.208000638337619 z_k1_s1125
                                 - 0.111000528887769 z_k1_s1126
                                 - 0.0920003746382179 z_k1_s1127
                                 - 0.157999938697935 z_k1_s1130
                                 - 0.10799997323656 z_k1_s1131
                                 - 0.0659999304890199 z_k1_s1132
                                 - 0.0639999797105549 z_k1_s1133
                                 - 0.163000413643971 z_k1_s1134
                                 - 0.260999436988458 z_k1_s1135
                                 - 0.295000513821961 z_k1_s1136
                                 - 0.0810000397274072 z_k1_s1137
                                 - 0.134999889828611 z_k1_s1140
                                 - 0.192000554780702 z_k1_s1141
                                 - 0.0770001891400727 z_k1_s1142
                                 - 0.164001140877277 z_k1_s1143
                                 - 0.217001127269343 z_k1_s1144
                                 - 0.211000776431399 z_k1_s1145
                                 - 0.201001081037618 z_k1_s1146
                                 - 0.137000896110717 z_k1_s1147
                                 - 0.0470001374224952 z_k1_s1150
                                 - 0.0360000970555065 z_k1_s1151
                                 - 0.083000138077746 z_k1_s1152
                                 - 0.0380001131586186 z_k1_s1153
                                 - 0.0690004109569838 z_k1_s1154
                                 - 0.0460001293709391 z_k1_s1155
                                 - 0.0710002342595847 z_k1_s1156
                                 - 0.092000065941367 z_k1_s1157
                                 - 0.154999742454211 z_k1_s1160
                                 - 0.0869999913775272 z_k1_s1161
                                 - 0.18399990291339 z_k1_s1162
                                 - 0.0879999833933608 z_k1_s1163
                                 - 0.137999682185038 z_k1_s1164
                                 - 0.191000043024227 z_k1_s1165
                                 - 0.058000026918352 z_k1_s1166
                                 - 0.0789999572508575 z_k1_s1167
                                 - 0.0500001316344245 z_k1_s1170
                                 - 0.029000066587939 z_k1_s1171
                                 - 0.029000066587939 z_k1_s1172
                                 - 0.0689999766949334 z_k1_s1173
                                 - 0.0960001160977134 z_k1_s1174
                                 - 0.0350000921440971 z_k1_s1175
                                 - 0.0820003330007833 z_k1_s1176
                                 - 0.0980004336839626 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.08597 w_k1_ADM_s000
                                 + 1.41267 w_k1_AXP_s000 + 1.3008 w_k1_DIS_s000
                                 + 1.16717 w_k1_FDX_s000
                                 + 1.25585 w_k1_IBM_s000
                                 + 1.42669 w_k1_INTC_s000
                                 + 1.16486 w_k1_KO_s000 + 1.55847 w_k1_MAT_s000
                                 + 1.27417 w_k1_MSFT_s000
                                 + 1.08574 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.16453 w_k1_ADM_s000
                                 + 1.30755 w_k1_AXP_s000
                                 + 1.80172 w_k1_DIS_s000
                                 + 1.36822 w_k1_FDX_s000
                                 + 1.10803 w_k1_IBM_s000
                                 + 1.34489 w_k1_INTC_s000
                                 + 1.36936 w_k1_KO_s000 + 1.52089 w_k1_MAT_s000
                                 + 1.48024 w_k1_MSFT_s000
                                 + 1.35345 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.18017 w_k1_ADM_s000
                                 + 1.01899 w_k1_AXP_s000
                                 + 1.15957 w_k1_DIS_s000
                                 + 0.96678 w_k1_FDX_s000
                                 + 1.66146 w_k1_IBM_s000
                                 + 1.03154 w_k1_INTC_s000
                                 + 1.27275 w_k1_KO_s000 + 1.37131 w_k1_MAT_s000
                                 + 1.07051 w_k1_MSFT_s000
                                 + 1.08195 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.923764 w_k1_ADM_s000
                                 + 0.961845 w_k1_AXP_s000
                                 + 0.848208 w_k1_DIS_s000
                                 + 0.878781 w_k1_FDX_s000
                                 + 1.10149 w_k1_IBM_s000
                                 + 1.12721 w_k1_INTC_s000
                                 + 1.02293 w_k1_KO_s000
                                 + 0.885517 w_k1_MAT_s000
                                 + 0.939784 w_k1_MSFT_s000
                                 + 0.988867 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.995876 w_k1_ADM_s000
                                 + 1.07421 w_k1_AXP_s000
                                 + 1.14968 w_k1_DIS_s000
                                 + 1.18936 w_k1_FDX_s000
                                 + 0.898585 w_k1_IBM_s000
                                 + 1.35258 w_k1_INTC_s000 + 0.9867 w_k1_KO_s000
                                 + 0.947351 w_k1_MAT_s000
                                 + 1.20349 w_k1_MSFT_s000
                                 + 0.975921 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.928976 w_k1_ADM_s000
                                 + 0.908789 w_k1_AXP_s000
                                 + 1.24309 w_k1_DIS_s000
                                 + 1.05478 w_k1_FDX_s000
                                 + 1.23746 w_k1_IBM_s000
                                 + 1.1416 w_k1_INTC_s000 + 1.14163 w_k1_KO_s000
                                 + 1.35925 w_k1_MAT_s000
                                 + 1.28923 w_k1_MSFT_s000
                                 + 1.01527 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.824219 w_k1_ADM_s000
                                 + 0.711337 w_k1_AXP_s000
                                 + 0.719994 w_k1_DIS_s000
                                 + 0.865063 w_k1_FDX_s000
                                 + 0.800647 w_k1_IBM_s000
                                 + 1.07289 w_k1_INTC_s000
                                 + 0.766963 w_k1_KO_s000
                                 + 0.638984 w_k1_MAT_s000
                                 + 0.878522 w_k1_MSFT_s000
                                 + 0.79933 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.01217 w_k1_ADM_s000
                                 + 0.97301 w_k1_AXP_s000
                                 + 0.858722 w_k1_DIS_s000
                                 + 1.06624 w_k1_FDX_s000
                                 + 0.671306 w_k1_IBM_s000
                                 + 1.0921 w_k1_INTC_s000
                                 + 0.816835 w_k1_KO_s000
                                 + 0.835006 w_k1_MAT_s000
                                 + 1.04874 w_k1_MSFT_s000
                                 + 0.968436 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.17818 w_k1_ADM_s001
                                 + 1.01288 w_k1_AXP_s001
                                 + 0.935501 w_k1_DIS_s001
                                 + 1.03399 w_k1_FDX_s001
                                 + 0.936481 w_k1_IBM_s001
                                 + 0.967425 w_k1_INTC_s001
                                 + 0.854106 w_k1_KO_s001
                                 + 0.825928 w_k1_MAT_s001
                                 + 0.90583 w_k1_MSFT_s001
                                 + 0.953706 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.816217 w_k1_ADM_s001
                                 + 0.682417 w_k1_AXP_s001
                                 + 0.841047 w_k1_DIS_s001
                                 + 0.786255 w_k1_FDX_s001
                                 + 1.13944 w_k1_IBM_s001
                                 + 0.700092 w_k1_INTC_s001
                                 + 0.877526 w_k1_KO_s001
                                 + 0.879428 w_k1_MAT_s001
                                 + 0.810271 w_k1_MSFT_s001
                                 + 0.859479 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.897165 w_k1_ADM_s001
                                 + 0.765789 w_k1_AXP_s001
                                 + 0.605835 w_k1_DIS_s001
                                 + 0.770625 w_k1_FDX_s001
                                 + 0.709325 w_k1_IBM_s001
                                 + 0.92512 w_k1_INTC_s001
                                 + 0.673341 w_k1_KO_s001
                                 + 0.668876 w_k1_MAT_s001
                                 + 0.717778 w_k1_MSFT_s001
                                 + 0.747303 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.19933 w_k1_ADM_s001
                                 + 1.23049 w_k1_AXP_s001
                                 + 1.31606 w_k1_DIS_s001
                                 + 1.14812 w_k1_FDX_s001
                                 + 0.843791 w_k1_IBM_s001
                                 + 1.0611 w_k1_INTC_s001
                                 + 0.992781 w_k1_KO_s001
                                 + 0.833531 w_k1_MAT_s001
                                 + 1.03423 w_k1_MSFT_s001
                                 + 1.07945 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.975369 w_k1_ADM_s001
                                 + 0.861152 w_k1_AXP_s001
                                 + 0.949301 w_k1_DIS_s001
                                 + 1.02449 w_k1_FDX_s001
                                 + 0.65639 w_k1_IBM_s001
                                 + 1.01464 w_k1_INTC_s001
                                 + 0.734679 w_k1_KO_s001
                                 + 0.665415 w_k1_MAT_s001
                                 + 1.00812 w_k1_MSFT_s001
                                 + 0.926647 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.884668 w_k1_ADM_s001
                                 + 1.33509 w_k1_AXP_s001
                                 + 1.12319 w_k1_DIS_s001
                                 + 1.00634 w_k1_FDX_s001
                                 + 1.51121 w_k1_IBM_s001
                                 + 0.973877 w_k1_INTC_s001
                                 + 1.16596 w_k1_KO_s001 + 1.78198 w_k1_MAT_s001
                                 + 0.946037 w_k1_MSFT_s001
                                 + 0.89788 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.04507 w_k1_ADM_s001
                                 + 0.994138 w_k1_AXP_s001
                                 + 1.56632 w_k1_DIS_s001
                                 + 1.24543 w_k1_FDX_s001
                                 + 1.30863 w_k1_IBM_s001
                                 + 1.06976 w_k1_INTC_s001 + 1.335 w_k1_KO_s001
                                 + 1.48236 w_k1_MAT_s001
                                 + 1.26063 w_k1_MSFT_s001
                                 + 1.06683 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.11577 w_k1_ADM_s001
                                 + 1.50837 w_k1_AXP_s001
                                 + 1.32256 w_k1_DIS_s001
                                 + 1.27345 w_k1_FDX_s001
                                 + 1.09464 w_k1_IBM_s001
                                 + 1.08403 w_k1_INTC_s001
                                 + 1.09112 w_k1_KO_s001 + 1.31957 w_k1_MAT_s001
                                 + 1.077 w_k1_MSFT_s001 + 0.94602 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.894917 w_k1_ADM_s002
                                 + 1.3025 w_k1_AXP_s002
                                 + 0.957578 w_k1_DIS_s002
                                 + 1.03142 w_k1_FDX_s002
                                 + 0.828397 w_k1_IBM_s002
                                 + 0.867873 w_k1_INTC_s002
                                 + 0.799553 w_k1_KO_s002
                                 + 0.897038 w_k1_MAT_s002
                                 + 0.899116 w_k1_MSFT_s002
                                 + 1.0051 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.67119 w_k1_ADM_s002
                                 + 0.708902 w_k1_AXP_s002
                                 + 0.662518 w_k1_DIS_s002
                                 + 0.67619 w_k1_FDX_s002
                                 + 1.01051 w_k1_IBM_s002
                                 + 0.649145 w_k1_INTC_s002
                                 + 0.861795 w_k1_KO_s002
                                 + 0.747884 w_k1_MAT_s002
                                 + 0.677108 w_k1_MSFT_s002
                                 + 0.885078 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.09541 w_k1_ADM_s002
                                 + 1.10037 w_k1_AXP_s002
                                 + 1.24319 w_k1_DIS_s002
                                 + 1.25419 w_k1_FDX_s002
                                 + 0.98095 w_k1_IBM_s002
                                 + 1.10146 w_k1_INTC_s002
                                 + 0.91733 w_k1_KO_s002
                                 + 0.983034 w_k1_MAT_s002
                                 + 0.999744 w_k1_MSFT_s002
                                 + 1.03175 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.16835 w_k1_ADM_s002
                                 + 1.44787 w_k1_AXP_s002
                                 + 1.71337 w_k1_DIS_s002
                                 + 1.30391 w_k1_FDX_s002
                                 + 1.49653 w_k1_IBM_s002
                                 + 0.942952 w_k1_INTC_s002
                                 + 1.37867 w_k1_KO_s002 + 1.64342 w_k1_MAT_s002
                                 + 1.29634 w_k1_MSFT_s002
                                 + 1.28104 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.750149 w_k1_ADM_s002
                                 + 1.0767 w_k1_AXP_s002 + 0.96708 w_k1_DIS_s002
                                 + 0.892939 w_k1_FDX_s002
                                 + 1.19383 w_k1_IBM_s002
                                 + 0.709315 w_k1_INTC_s002
                                 + 0.951367 w_k1_KO_s002 + 1.2936 w_k1_MAT_s002
                                 + 0.794227 w_k1_MSFT_s002
                                 + 0.97832 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.917102 w_k1_ADM_s002
                                 + 1.03081 w_k1_AXP_s002
                                 + 1.11564 w_k1_DIS_s002
                                 + 0.922898 w_k1_FDX_s002
                                 + 1.62694 w_k1_IBM_s002
                                 + 0.792849 w_k1_INTC_s002
                                 + 1.1849 w_k1_KO_s002 + 1.35405 w_k1_MAT_s002
                                 + 1.03385 w_k1_MSFT_s002
                                 + 1.06177 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.982106 w_k1_ADM_s002
                                 + 0.783142 w_k1_AXP_s002
                                 + 0.739556 w_k1_DIS_s002
                                 + 0.87637 w_k1_FDX_s002
                                 + 0.670307 w_k1_IBM_s002
                                 + 0.864643 w_k1_INTC_s002
                                 + 0.818398 w_k1_KO_s002
                                 + 0.640401 w_k1_MAT_s002
                                 + 0.965455 w_k1_MSFT_s002
                                 + 0.9065 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.84969 w_k1_ADM_s002
                                 + 0.877792 w_k1_AXP_s002
                                 + 1.15025 w_k1_DIS_s002
                                 + 1.00805 w_k1_FDX_s002
                                 + 0.845518 w_k1_IBM_s002
                                 + 0.945009 w_k1_INTC_s002
                                 + 1.00529 w_k1_KO_s002
                                 + 0.709413 w_k1_MAT_s002
                                 + 0.905709 w_k1_MSFT_s002
                                 + 1.05273 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.18103 w_k1_ADM_s003
                                 + 0.940043 w_k1_AXP_s003
                                 + 1.10067 w_k1_DIS_s003
                                 + 1.05849 w_k1_FDX_s003
                                 + 0.787613 w_k1_IBM_s003
                                 + 1.2145 w_k1_INTC_s003 + 1.07004 w_k1_KO_s003
                                 + 0.817641 w_k1_MAT_s003
                                 + 1.056 w_k1_MSFT_s003 + 1.12718 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.906081 w_k1_ADM_s003
                                 + 1.04792 w_k1_AXP_s003
                                 + 0.768896 w_k1_DIS_s003
                                 + 0.883984 w_k1_FDX_s003
                                 + 0.846667 w_k1_IBM_s003
                                 + 0.947399 w_k1_INTC_s003
                                 + 0.8906 w_k1_KO_s003 + 1.14759 w_k1_MAT_s003
                                 + 0.890332 w_k1_MSFT_s003
                                 + 0.931421 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.34593 w_k1_ADM_s003
                                 + 1.52156 w_k1_AXP_s003 + 1.1875 w_k1_DIS_s003
                                 + 1.24843 w_k1_FDX_s003
                                 + 1.09474 w_k1_IBM_s003
                                 + 0.975795 w_k1_INTC_s003
                                 + 1.2315 w_k1_KO_s003 + 1.57951 w_k1_MAT_s003
                                 + 1.03854 w_k1_MSFT_s003
                                 + 1.10027 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.941994 w_k1_ADM_s003
                                 + 0.666504 w_k1_AXP_s003
                                 + 0.700413 w_k1_DIS_s003
                                 + 0.754565 w_k1_FDX_s003
                                 + 0.630039 w_k1_IBM_s003
                                 + 0.795354 w_k1_INTC_s003
                                 + 0.808846 w_k1_KO_s003
                                 + 0.612023 w_k1_MAT_s003
                                 + 0.857832 w_k1_MSFT_s003
                                 + 0.935084 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.759431 w_k1_ADM_s003
                                 + 0.74275 w_k1_AXP_s003
                                 + 0.598207 w_k1_DIS_s003
                                 + 0.755866 w_k1_FDX_s003
                                 + 0.986668 w_k1_IBM_s003
                                 + 0.788457 w_k1_INTC_s003
                                 + 0.751236 w_k1_KO_s003
                                 + 1.09463 w_k1_MAT_s003
                                 + 0.699467 w_k1_MSFT_s003
                                 + 0.834079 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.965665 w_k1_ADM_s003
                                 + 0.982021 w_k1_AXP_s003
                                 + 0.956402 w_k1_DIS_s003
                                 + 0.866157 w_k1_FDX_s003
                                 + 1.38794 w_k1_IBM_s003
                                 + 0.855335 w_k1_INTC_s003
                                 + 1.11322 w_k1_KO_s003 + 1.52495 w_k1_MAT_s003
                                 + 1.01032 w_k1_MSFT_s003
                                 + 1.14561 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.17497 w_k1_ADM_s003
                                 + 0.843787 w_k1_AXP_s003
                                 + 0.812561 w_k1_DIS_s003
                                 + 0.952046 w_k1_FDX_s003
                                 + 0.911463 w_k1_IBM_s003
                                 + 1.09261 w_k1_INTC_s003
                                 + 1.07029 w_k1_KO_s003 + 1.12633 w_k1_MAT_s003
                                 + 0.997729 w_k1_MSFT_s003
                                 + 1.06867 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.21934 w_k1_ADM_s003
                                 + 1.1215 w_k1_AXP_s003 + 1.2835 w_k1_DIS_s003
                                 + 1.0622 w_k1_FDX_s003 + 1.38675 w_k1_IBM_s003
                                 + 1.14989 w_k1_INTC_s003
                                 + 1.38539 w_k1_KO_s003 + 1.56363 w_k1_MAT_s003
                                 + 1.26378 w_k1_MSFT_s003
                                 + 1.16631 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.954692 w_k1_ADM_s004
                                 + 0.915547 w_k1_AXP_s004
                                 + 1.04645 w_k1_DIS_s004 + 1.0711 w_k1_FDX_s004
                                 + 0.861017 w_k1_IBM_s004
                                 + 1.26763 w_k1_INTC_s004
                                 + 0.972571 w_k1_KO_s004
                                 + 0.921149 w_k1_MAT_s004
                                 + 1.13822 w_k1_MSFT_s004
                                 + 0.919235 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.804259 w_k1_ADM_s004
                                 + 0.691333 w_k1_AXP_s004
                                 + 0.556577 w_k1_DIS_s004
                                 + 0.781223 w_k1_FDX_s004
                                 + 0.710327 w_k1_IBM_s004
                                 + 0.98055 w_k1_INTC_s004
                                 + 0.69007 w_k1_KO_s004
                                 + 0.815044 w_k1_MAT_s004
                                 + 0.867622 w_k1_MSFT_s004
                                 + 0.813803 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.930924 w_k1_ADM_s004
                                 + 0.660385 w_k1_AXP_s004
                                 + 0.816816 w_k1_DIS_s004
                                 + 0.846497 w_k1_FDX_s004
                                 + 0.631624 w_k1_IBM_s004
                                 + 1.0116 w_k1_INTC_s004
                                 + 0.788822 w_k1_KO_s004
                                 + 0.680272 w_k1_MAT_s004
                                 + 0.939033 w_k1_MSFT_s004
                                 + 0.83559 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.988749 w_k1_ADM_s004
                                 + 0.960113 w_k1_AXP_s004
                                 + 1.16877 w_k1_DIS_s004
                                 + 1.02456 w_k1_FDX_s004
                                 + 1.21148 w_k1_IBM_s004
                                 + 1.22856 w_k1_INTC_s004
                                 + 1.11885 w_k1_KO_s004 + 1.06923 w_k1_MAT_s004
                                 + 1.11907 w_k1_MSFT_s004
                                 + 0.963795 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.21064 w_k1_ADM_s004
                                 + 1.28047 w_k1_AXP_s004
                                 + 1.39925 w_k1_DIS_s004
                                 + 1.26243 w_k1_FDX_s004
                                 + 1.44427 w_k1_IBM_s004
                                 + 1.21237 w_k1_INTC_s004
                                 + 1.29342 w_k1_KO_s004 + 1.3102 w_k1_MAT_s004
                                 + 1.13228 w_k1_MSFT_s004
                                 + 1.02778 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.15395 w_k1_ADM_s004
                                 + 1.20326 w_k1_AXP_s004
                                 + 1.03501 w_k1_DIS_s004
                                 + 1.05419 w_k1_FDX_s004
                                 + 1.04478 w_k1_IBM_s004
                                 + 1.05539 w_k1_INTC_s004
                                 + 1.05976 w_k1_KO_s004 + 1.37182 w_k1_MAT_s004
                                 + 0.972089 w_k1_MSFT_s004
                                 + 0.980451 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.3318 w_k1_ADM_s004
                                 + 1.05856 w_k1_AXP_s004
                                 + 1.17698 w_k1_DIS_s004 + 1.3188 w_k1_FDX_s004
                                 + 0.712938 w_k1_IBM_s004
                                 + 1.26343 w_k1_INTC_s004
                                 + 0.925776 w_k1_KO_s004
                                 + 0.75133 w_k1_MAT_s004
                                 + 1.20164 w_k1_MSFT_s004
                                 + 0.963085 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.744956 w_k1_ADM_s004
                                 + 0.791006 w_k1_AXP_s004
                                 + 0.817812 w_k1_DIS_s004
                                 + 0.764373 w_k1_FDX_s004
                                 + 1.1756 w_k1_IBM_s004
                                 + 0.992017 w_k1_INTC_s004
                                 + 0.937641 w_k1_KO_s004
                                 + 0.922298 w_k1_MAT_s004
                                 + 0.889987 w_k1_MSFT_s004
                                 + 0.865794 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.756169 w_k1_ADM_s005
                                 + 0.712317 w_k1_AXP_s005
                                 + 0.664453 w_k1_DIS_s005
                                 + 0.708665 w_k1_FDX_s005
                                 + 0.837161 w_k1_IBM_s005
                                 + 0.72268 w_k1_INTC_s005
                                 + 0.653463 w_k1_KO_s005
                                 + 0.678874 w_k1_MAT_s005
                                 + 0.778034 w_k1_MSFT_s005
                                 + 0.753296 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.42602 w_k1_ADM_s005
                                 + 1.25394 w_k1_AXP_s005 + 1.2009 w_k1_DIS_s005
                                 + 1.04833 w_k1_FDX_s005
                                 + 1.22391 w_k1_IBM_s005
                                 + 1.02734 w_k1_INTC_s005
                                 + 1.15943 w_k1_KO_s005 + 1.18766 w_k1_MAT_s005
                                 + 1.05394 w_k1_MSFT_s005
                                 + 1.12647 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.939848 w_k1_ADM_s005
                                 + 0.60421 w_k1_AXP_s005
                                 + 0.881041 w_k1_DIS_s005
                                 + 0.770913 w_k1_FDX_s005
                                 + 0.976628 w_k1_IBM_s005
                                 + 1.02935 w_k1_INTC_s005
                                 + 1.07387 w_k1_KO_s005
                                 + 0.660505 w_k1_MAT_s005
                                 + 0.942569 w_k1_MSFT_s005
                                 + 0.875261 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.05727 w_k1_ADM_s005
                                 + 0.977807 w_k1_AXP_s005
                                 + 0.830643 w_k1_DIS_s005
                                 + 0.823109 w_k1_FDX_s005
                                 + 1.37032 w_k1_IBM_s005
                                 + 0.801405 w_k1_INTC_s005
                                 + 1.07085 w_k1_KO_s005 + 1.37583 w_k1_MAT_s005
                                 + 0.971893 w_k1_MSFT_s005
                                 + 1.01943 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.738173 w_k1_ADM_s005
                                 + 1.00854 w_k1_AXP_s005
                                 + 0.797706 w_k1_DIS_s005
                                 + 0.847158 w_k1_FDX_s005
                                 + 1.0592 w_k1_IBM_s005
                                 + 0.867568 w_k1_INTC_s005
                                 + 0.965262 w_k1_KO_s005
                                 + 1.11056 w_k1_MAT_s005
                                 + 0.886317 w_k1_MSFT_s005
                                 + 0.889168 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.2188 w_k1_ADM_s005
                                 + 1.1455 w_k1_AXP_s005 + 1.46955 w_k1_DIS_s005
                                 + 1.23092 w_k1_FDX_s005
                                 + 1.69642 w_k1_IBM_s005
                                 + 1.29426 w_k1_INTC_s005
                                 + 1.50247 w_k1_KO_s005 + 1.71188 w_k1_MAT_s005
                                 + 1.40507 w_k1_MSFT_s005
                                 + 1.27705 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.41769 w_k1_ADM_s005
                                 + 1.32064 w_k1_AXP_s005
                                 + 1.39811 w_k1_DIS_s005
                                 + 1.27182 w_k1_FDX_s005
                                 + 0.789754 w_k1_IBM_s005
                                 + 0.830436 w_k1_INTC_s005
                                 + 0.987225 w_k1_KO_s005
                                 + 0.813098 w_k1_MAT_s005
                                 + 1.08995 w_k1_MSFT_s005
                                 + 0.936609 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.750222 w_k1_ADM_s005
                                 + 0.753748 w_k1_AXP_s005
                                 + 0.742458 w_k1_DIS_s005
                                 + 0.840321 w_k1_FDX_s005
                                 + 0.701755 w_k1_IBM_s005
                                 + 0.793367 w_k1_INTC_s005
                                 + 0.736653 w_k1_KO_s005
                                 + 0.759878 w_k1_MAT_s005
                                 + 0.874122 w_k1_MSFT_s005
                                 + 0.735103 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.24843 w_k1_ADM_s006
                                 + 0.948063 w_k1_AXP_s006
                                 + 1.49533 w_k1_DIS_s006
                                 + 1.17792 w_k1_FDX_s006
                                 + 0.97091 w_k1_IBM_s006
                                 + 1.25411 w_k1_INTC_s006
                                 + 1.34816 w_k1_KO_s006
                                 + 0.545407 w_k1_MAT_s006
                                 + 1.30346 w_k1_MSFT_s006
                                 + 1.17189 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.31665 w_k1_ADM_s006
                                 + 1.32261 w_k1_AXP_s006
                                 + 1.57353 w_k1_DIS_s006
                                 + 1.19468 w_k1_FDX_s006
                                 + 1.25584 w_k1_IBM_s006
                                 + 1.15791 w_k1_INTC_s006
                                 + 1.36379 w_k1_KO_s006 + 1.56116 w_k1_MAT_s006
                                 + 1.32292 w_k1_MSFT_s006
                                 + 1.22489 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.12635 w_k1_ADM_s006
                                 + 1.1516 w_k1_AXP_s006 + 1.35133 w_k1_DIS_s006
                                 + 1.27667 w_k1_FDX_s006
                                 + 0.700117 w_k1_IBM_s006
                                 + 1.35789 w_k1_INTC_s006
                                 + 0.986675 w_k1_KO_s006
                                 + 0.707934 w_k1_MAT_s006
                                 + 1.32919 w_k1_MSFT_s006
                                 + 1.12582 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.06507 w_k1_ADM_s006
                                 + 1.12967 w_k1_AXP_s006
                                 + 1.02902 w_k1_DIS_s006
                                 + 0.970621 w_k1_FDX_s006
                                 + 1.41753 w_k1_IBM_s006
                                 + 0.893111 w_k1_INTC_s006
                                 + 1.19703 w_k1_KO_s006 + 1.70755 w_k1_MAT_s006
                                 + 0.961054 w_k1_MSFT_s006
                                 + 1.12946 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.77209 w_k1_ADM_s006
                                 + 0.609171 w_k1_AXP_s006
                                 + 0.795736 w_k1_DIS_s006
                                 + 0.775752 w_k1_FDX_s006
                                 + 0.792469 w_k1_IBM_s006
                                 + 0.755257 w_k1_INTC_s006
                                 + 0.835544 w_k1_KO_s006
                                 + 0.477516 w_k1_MAT_s006
                                 + 0.794225 w_k1_MSFT_s006
                                 + 0.893039 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.54815 w_k1_ADM_s006
                                 + 1.75583 w_k1_AXP_s006
                                 + 1.94326 w_k1_DIS_s006
                                 + 1.49672 w_k1_FDX_s006
                                 + 0.99338 w_k1_IBM_s006
                                 + 1.57315 w_k1_INTC_s006
                                 + 1.43124 w_k1_KO_s006 + 1.10208 w_k1_MAT_s006
                                 + 1.39557 w_k1_MSFT_s006
                                 + 1.35757 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.07352 w_k1_ADM_s006
                                 + 1.02428 w_k1_AXP_s006
                                 + 0.947115 w_k1_DIS_s006
                                 + 0.99967 w_k1_FDX_s006
                                 + 0.742278 w_k1_IBM_s006
                                 + 1.15865 w_k1_INTC_s006
                                 + 0.873255 w_k1_KO_s006
                                 + 0.668266 w_k1_MAT_s006
                                 + 0.868095 w_k1_MSFT_s006
                                 + 0.92531 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.998216 w_k1_ADM_s006
                                 + 1.03532 w_k1_AXP_s006
                                 + 0.828951 w_k1_DIS_s006
                                 + 0.820691 w_k1_FDX_s006
                                 + 0.988449 w_k1_IBM_s006
                                 + 1.05794 w_k1_INTC_s006
                                 + 1.03209 w_k1_KO_s006 + 1.16747 w_k1_MAT_s006
                                 + 0.896283 w_k1_MSFT_s006
                                 + 1.0958 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.925282 w_k1_ADM_s007
                                 + 1.43809 w_k1_AXP_s007
                                 + 1.14436 w_k1_DIS_s007
                                 + 1.15765 w_k1_FDX_s007
                                 + 1.01755 w_k1_IBM_s007
                                 + 1.2184 w_k1_INTC_s007 + 1.04004 w_k1_KO_s007
                                 + 1.19282 w_k1_MAT_s007
                                 + 1.00783 w_k1_MSFT_s007
                                 + 1.10232 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.10812 w_k1_ADM_s007
                                 + 0.930506 w_k1_AXP_s007
                                 + 0.708996 w_k1_DIS_s007
                                 + 0.840977 w_k1_FDX_s007
                                 + 0.759622 w_k1_IBM_s007
                                 + 0.993058 w_k1_INTC_s007
                                 + 0.695494 w_k1_KO_s007
                                 + 0.690455 w_k1_MAT_s007
                                 + 0.872977 w_k1_MSFT_s007
                                 + 0.879765 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.636264 w_k1_ADM_s007
                                 + 0.60011 w_k1_AXP_s007
                                 + 0.715856 w_k1_DIS_s007
                                 + 0.700548 w_k1_FDX_s007
                                 + 0.928291 w_k1_IBM_s007
                                 + 0.78332 w_k1_INTC_s007
                                 + 0.888994 w_k1_KO_s007
                                 + 0.810362 w_k1_MAT_s007
                                 + 0.837887 w_k1_MSFT_s007
                                 + 0.901806 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.46801 w_k1_ADM_s007
                                 + 1.40284 w_k1_AXP_s007
                                 + 1.60681 w_k1_DIS_s007
                                 + 1.20979 w_k1_FDX_s007
                                 + 1.80827 w_k1_IBM_s007
                                 + 1.14459 w_k1_INTC_s007
                                 + 1.45326 w_k1_KO_s007 + 1.9249 w_k1_MAT_s007
                                 + 1.22833 w_k1_MSFT_s007
                                 + 1.27114 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.985808 w_k1_ADM_s007
                                 + 1.00461 w_k1_AXP_s007
                                 + 0.886922 w_k1_DIS_s007
                                 + 0.888681 w_k1_FDX_s007
                                 + 1.1554 w_k1_IBM_s007
                                 + 0.906018 w_k1_INTC_s007
                                 + 0.988231 w_k1_KO_s007
                                 + 1.09164 w_k1_MAT_s007
                                 + 0.949559 w_k1_MSFT_s007
                                 + 0.996907 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.15134 w_k1_ADM_s007
                                 + 0.964472 w_k1_AXP_s007
                                 + 1.21008 w_k1_DIS_s007 + 1.1701 w_k1_FDX_s007
                                 + 0.764389 w_k1_IBM_s007
                                 + 1.09667 w_k1_INTC_s007
                                 + 0.931209 w_k1_KO_s007
                                 + 0.727387 w_k1_MAT_s007
                                 + 1.24625 w_k1_MSFT_s007
                                 + 1.09401 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.751637 w_k1_ADM_s007
                                 + 0.886493 w_k1_AXP_s007
                                 + 0.810485 w_k1_DIS_s007
                                 + 0.77821 w_k1_FDX_s007
                                 + 1.36796 w_k1_IBM_s007
                                 + 0.897499 w_k1_INTC_s007
                                 + 1.06668 w_k1_KO_s007 + 1.40268 w_k1_MAT_s007
                                 + 0.914014 w_k1_MSFT_s007
                                 + 1.20282 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.49897 w_k1_ADM_s007
                                 + 0.536972 w_k1_AXP_s007
                                 + 0.468035 w_k1_DIS_s007
                                 + 0.629215 w_k1_FDX_s007
                                 + 0.604919 w_k1_IBM_s007
                                 + 0.688202 w_k1_INTC_s007
                                 + 0.605228 w_k1_KO_s007
                                 + 0.704629 w_k1_MAT_s007
                                 + 0.66987 w_k1_MSFT_s007
                                 + 0.81479 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.901502 w_k1_ADM_s010
                                 + 1.11879 w_k1_AXP_s010
                                 + 1.12034 w_k1_DIS_s010
                                 + 0.992678 w_k1_FDX_s010
                                 + 1.83102 w_k1_IBM_s010
                                 + 0.818517 w_k1_INTC_s010
                                 + 1.10829 w_k1_KO_s010 + 1.97233 w_k1_MAT_s010
                                 + 0.873638 w_k1_MSFT_s010
                                 + 1.27354 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.72978 w_k1_ADM_s010
                                 + 0.598325 w_k1_AXP_s010
                                 + 0.546328 w_k1_DIS_s010
                                 + 0.706071 w_k1_FDX_s010
                                 + 0.808777 w_k1_IBM_s010
                                 + 0.793017 w_k1_INTC_s010
                                 + 0.621442 w_k1_KO_s010
                                 + 0.579068 w_k1_MAT_s010
                                 + 0.693698 w_k1_MSFT_s010
                                 + 0.874602 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.48795 w_k1_ADM_s010
                                 + 1.18547 w_k1_AXP_s010
                                 + 1.46198 w_k1_DIS_s010
                                 + 1.33112 w_k1_FDX_s010
                                 + 1.15463 w_k1_IBM_s010
                                 + 1.11792 w_k1_INTC_s010 + 1.0945 w_k1_KO_s010
                                 + 0.935806 w_k1_MAT_s010
                                 + 1.25527 w_k1_MSFT_s010
                                 + 1.18672 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.0348 w_k1_ADM_s010
                                 + 0.80242 w_k1_AXP_s010
                                 + 0.675534 w_k1_DIS_s010
                                 + 0.906847 w_k1_FDX_s010
                                 + 0.763017 w_k1_IBM_s010
                                 + 0.91169 w_k1_INTC_s010
                                 + 0.785916 w_k1_KO_s010
                                 + 0.596493 w_k1_MAT_s010
                                 + 0.870316 w_k1_MSFT_s010
                                 + 0.858502 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.778315 w_k1_ADM_s010
                                 + 0.796865 w_k1_AXP_s010
                                 + 0.760076 w_k1_DIS_s010
                                 + 0.772734 w_k1_FDX_s010
                                 + 1.24459 w_k1_IBM_s010
                                 + 0.957242 w_k1_INTC_s010
                                 + 0.908517 w_k1_KO_s010
                                 + 1.09799 w_k1_MAT_s010
                                 + 0.831605 w_k1_MSFT_s010
                                 + 0.976345 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.929637 w_k1_ADM_s010
                                 + 1.00342 w_k1_AXP_s010
                                 + 0.969629 w_k1_DIS_s010
                                 + 1.00875 w_k1_FDX_s010
                                 + 0.93221 w_k1_IBM_s010
                                 + 0.912311 w_k1_INTC_s010
                                 + 0.91178 w_k1_KO_s010
                                 + 0.991032 w_k1_MAT_s010
                                 + 1.01688 w_k1_MSFT_s010
                                 + 1.02229 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.719297 w_k1_ADM_s010
                                 + 0.96775 w_k1_AXP_s010
                                 + 0.894855 w_k1_DIS_s010
                                 + 1.01159 w_k1_FDX_s010
                                 + 0.790757 w_k1_IBM_s010
                                 + 1.13488 w_k1_INTC_s010
                                 + 0.877392 w_k1_KO_s010
                                 + 0.794067 w_k1_MAT_s010
                                 + 0.944164 w_k1_MSFT_s010
                                 + 0.986878 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.970493 w_k1_ADM_s010
                                 + 0.928154 w_k1_AXP_s010
                                 + 1.03197 w_k1_DIS_s010 + 1.0184 w_k1_FDX_s010
                                 + 1.25957 w_k1_IBM_s010
                                 + 1.00473 w_k1_INTC_s010
                                 + 0.979084 w_k1_KO_s010
                                 + 1.03559 w_k1_MAT_s010
                                 + 1.06832 w_k1_MSFT_s010
                                 + 1.05764 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.823781 w_k1_ADM_s011
                                 + 0.703388 w_k1_AXP_s011
                                 + 0.805272 w_k1_DIS_s011
                                 + 0.789864 w_k1_FDX_s011
                                 + 0.839689 w_k1_IBM_s011
                                 + 1.03237 w_k1_INTC_s011
                                 + 0.782532 w_k1_KO_s011
                                 + 0.637768 w_k1_MAT_s011
                                 + 0.964075 w_k1_MSFT_s011
                                 + 0.809335 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.1325 w_k1_ADM_s011
                                 + 0.923632 w_k1_AXP_s011
                                 + 1.0573 w_k1_DIS_s011 + 1.072 w_k1_FDX_s011
                                 + 0.76612 w_k1_IBM_s011
                                 + 1.12844 w_k1_INTC_s011
                                 + 1.03658 w_k1_KO_s011
                                 + 0.764087 w_k1_MAT_s011
                                 + 1.01263 w_k1_MSFT_s011
                                 + 0.942373 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.15115 w_k1_ADM_s011
                                 + 1.46593 w_k1_AXP_s011 + 1.4684 w_k1_DIS_s011
                                 + 1.3337 w_k1_FDX_s011 + 1.45734 w_k1_IBM_s011
                                 + 1.38692 w_k1_INTC_s011
                                 + 1.30786 w_k1_KO_s011 + 1.44983 w_k1_MAT_s011
                                 + 1.30125 w_k1_MSFT_s011
                                 + 1.18176 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.954812 w_k1_ADM_s011
                                 + 0.726103 w_k1_AXP_s011
                                 + 0.608934 w_k1_DIS_s011
                                 + 0.843126 w_k1_FDX_s011
                                 + 0.680212 w_k1_IBM_s011
                                 + 1.02878 w_k1_INTC_s011
                                 + 0.691628 w_k1_KO_s011
                                 + 0.422978 w_k1_MAT_s011
                                 + 0.814632 w_k1_MSFT_s011
                                 + 0.819174 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.27955 w_k1_ADM_s011
                                 + 1.46274 w_k1_AXP_s011
                                 + 1.06059 w_k1_DIS_s011
                                 + 1.25381 w_k1_FDX_s011
                                 + 1.15393 w_k1_IBM_s011
                                 + 1.19935 w_k1_INTC_s011
                                 + 1.26776 w_k1_KO_s011 + 1.23808 w_k1_MAT_s011
                                 + 1.28582 w_k1_MSFT_s011
                                 + 1.20361 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.09558 w_k1_ADM_s011
                                 + 1.07758 w_k1_AXP_s011
                                 + 1.09727 w_k1_DIS_s011
                                 + 1.18256 w_k1_FDX_s011
                                 + 0.970592 w_k1_IBM_s011
                                 + 1.21808 w_k1_INTC_s011
                                 + 1.06718 w_k1_KO_s011
                                 + 0.868209 w_k1_MAT_s011
                                 + 1.08709 w_k1_MSFT_s011
                                 + 1.04661 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.20751 w_k1_ADM_s011
                                 + 1.10147 w_k1_AXP_s011
                                 + 1.52146 w_k1_DIS_s011 + 1.2991 w_k1_FDX_s011
                                 + 0.911219 w_k1_IBM_s011
                                 + 1.53109 w_k1_INTC_s011 + 1.1475 w_k1_KO_s011
                                 + 0.693806 w_k1_MAT_s011
                                 + 1.62034 w_k1_MSFT_s011
                                 + 1.14774 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.914765 w_k1_ADM_s011
                                 + 0.851817 w_k1_AXP_s011
                                 + 0.919484 w_k1_DIS_s011
                                 + 0.888602 w_k1_FDX_s011
                                 + 1.30503 w_k1_IBM_s011
                                 + 0.936624 w_k1_INTC_s011
                                 + 1.03372 w_k1_KO_s011 + 1.51156 w_k1_MAT_s011
                                 + 1.08856 w_k1_MSFT_s011
                                 + 1.01003 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.06275 w_k1_ADM_s012
                                 + 0.843684 w_k1_AXP_s012
                                 + 0.860601 w_k1_DIS_s012
                                 + 0.815602 w_k1_FDX_s012
                                 + 1.19891 w_k1_IBM_s012
                                 + 0.958471 w_k1_INTC_s012
                                 + 1.02453 w_k1_KO_s012 + 1.04245 w_k1_MAT_s012
                                 + 0.926814 w_k1_MSFT_s012
                                 + 0.90841 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.20755 w_k1_ADM_s012
                                 + 0.936721 w_k1_AXP_s012
                                 + 1.07233 w_k1_DIS_s012
                                 + 1.11316 w_k1_FDX_s012
                                 + 0.896738 w_k1_IBM_s012
                                 + 0.903197 w_k1_INTC_s012
                                 + 0.955799 w_k1_KO_s012
                                 + 0.996912 w_k1_MAT_s012
                                 + 1.07106 w_k1_MSFT_s012
                                 + 0.99298 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.09767 w_k1_ADM_s012
                                 + 0.921594 w_k1_AXP_s012
                                 + 0.805207 w_k1_DIS_s012
                                 + 0.956543 w_k1_FDX_s012
                                 + 0.799068 w_k1_IBM_s012
                                 + 0.797401 w_k1_INTC_s012
                                 + 0.90137 w_k1_KO_s012 + 1.02198 w_k1_MAT_s012
                                 + 0.876398 w_k1_MSFT_s012
                                 + 0.954373 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.862028 w_k1_ADM_s012
                                 + 0.558341 w_k1_AXP_s012
                                 + 0.88914 w_k1_DIS_s012
                                 + 0.930342 w_k1_FDX_s012
                                 + 0.696482 w_k1_IBM_s012
                                 + 0.846459 w_k1_INTC_s012
                                 + 0.821399 w_k1_KO_s012
                                 + 0.634633 w_k1_MAT_s012
                                 + 0.894103 w_k1_MSFT_s012
                                 + 0.931172 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.764625 w_k1_ADM_s012
                                 + 0.515366 w_k1_AXP_s012
                                 + 0.542016 w_k1_DIS_s012
                                 + 0.723991 w_k1_FDX_s012
                                 + 0.61614 w_k1_IBM_s012
                                 + 0.738478 w_k1_INTC_s012
                                 + 0.637435 w_k1_KO_s012
                                 + 0.707075 w_k1_MAT_s012
                                 + 0.701096 w_k1_MSFT_s012
                                 + 0.753715 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.48427 w_k1_ADM_s012
                                 + 1.18747 w_k1_AXP_s012
                                 + 1.48348 w_k1_DIS_s012
                                 + 1.26974 w_k1_FDX_s012
                                 + 0.918049 w_k1_IBM_s012
                                 + 1.10123 w_k1_INTC_s012
                                 + 1.28701 w_k1_KO_s012 + 1.03527 w_k1_MAT_s012
                                 + 1.31319 w_k1_MSFT_s012
                                 + 1.10854 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.25443 w_k1_ADM_s012
                                 + 1.26474 w_k1_AXP_s012
                                 + 1.43621 w_k1_DIS_s012
                                 + 1.17907 w_k1_FDX_s012
                                 + 1.65064 w_k1_IBM_s012
                                 + 1.1909 w_k1_INTC_s012 + 1.64073 w_k1_KO_s012
                                 + 2.01064 w_k1_MAT_s012
                                 + 1.15312 w_k1_MSFT_s012 + 1.2481 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.12604 w_k1_ADM_s012
                                 + 1.14337 w_k1_AXP_s012
                                 + 1.10495 w_k1_DIS_s012
                                 + 1.11622 w_k1_FDX_s012
                                 + 1.29388 w_k1_IBM_s012
                                 + 1.10738 w_k1_INTC_s012
                                 + 1.17289 w_k1_KO_s012 + 1.7061 w_k1_MAT_s012
                                 + 1.07235 w_k1_MSFT_s012
                                 + 1.05384 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.0484 w_k1_ADM_s013
                                 + 0.913292 w_k1_AXP_s013
                                 + 0.86841 w_k1_DIS_s013
                                 + 0.95726 w_k1_FDX_s013
                                 + 0.664051 w_k1_IBM_s013
                                 + 0.655625 w_k1_INTC_s013
                                 + 0.720534 w_k1_KO_s013
                                 + 0.54763 w_k1_MAT_s013
                                 + 0.856047 w_k1_MSFT_s013
                                 + 0.877106 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.66682 w_k1_ADM_s013
                                 + 0.789847 w_k1_AXP_s013
                                 + 0.636132 w_k1_DIS_s013
                                 + 0.764562 w_k1_FDX_s013
                                 + 0.769514 w_k1_IBM_s013
                                 + 0.82379 w_k1_INTC_s013
                                 + 0.763614 w_k1_KO_s013
                                 + 0.574577 w_k1_MAT_s013
                                 + 0.813762 w_k1_MSFT_s013
                                 + 0.842356 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.05949 w_k1_ADM_s013
                                 + 1.06776 w_k1_AXP_s013
                                 + 1.28366 w_k1_DIS_s013
                                 + 1.01387 w_k1_FDX_s013
                                 + 1.12723 w_k1_IBM_s013
                                 + 0.926007 w_k1_INTC_s013
                                 + 1.27382 w_k1_KO_s013 + 1.11147 w_k1_MAT_s013
                                 + 1.07656 w_k1_MSFT_s013
                                 + 1.08897 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.33051 w_k1_ADM_s013
                                 + 1.11058 w_k1_AXP_s013
                                 + 1.00505 w_k1_DIS_s013
                                 + 1.05258 w_k1_FDX_s013
                                 + 0.857218 w_k1_IBM_s013
                                 + 1.01745 w_k1_INTC_s013
                                 + 0.97537 w_k1_KO_s013
                                 + 0.796828 w_k1_MAT_s013
                                 + 1.08732 w_k1_MSFT_s013
                                 + 1.00243 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.19973 w_k1_ADM_s013
                                 + 1.60114 w_k1_AXP_s013 + 1.4579 w_k1_DIS_s013
                                 + 1.24887 w_k1_FDX_s013
                                 + 1.64279 w_k1_IBM_s013
                                 + 1.04253 w_k1_INTC_s013
                                 + 1.39777 w_k1_KO_s013 + 1.77849 w_k1_MAT_s013
                                 + 1.25412 w_k1_MSFT_s013
                                 + 1.33084 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.31177 w_k1_ADM_s013
                                 + 1.35911 w_k1_AXP_s013
                                 + 1.65347 w_k1_DIS_s013 + 1.2946 w_k1_FDX_s013
                                 + 1.28458 w_k1_IBM_s013
                                 + 1.18603 w_k1_INTC_s013 + 1.5807 w_k1_KO_s013
                                 + 1.09687 w_k1_MAT_s013
                                 + 1.39075 w_k1_MSFT_s013
                                 + 1.23637 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.968113 w_k1_ADM_s013
                                 + 1.04818 w_k1_AXP_s013 + 1.7035 w_k1_DIS_s013
                                 + 1.09419 w_k1_FDX_s013
                                 + 1.65172 w_k1_IBM_s013
                                 + 1.22997 w_k1_INTC_s013
                                 + 1.55165 w_k1_KO_s013 + 1.12156 w_k1_MAT_s013
                                 + 1.33494 w_k1_MSFT_s013
                                 + 1.32466 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.931461 w_k1_ADM_s013
                                 + 1.52968 w_k1_AXP_s013
                                 + 0.931711 w_k1_DIS_s013
                                 + 0.981809 w_k1_FDX_s013
                                 + 1.22375 w_k1_IBM_s013
                                 + 0.78443 w_k1_INTC_s013
                                 + 1.03318 w_k1_KO_s013 + 1.45222 w_k1_MAT_s013
                                 + 0.833708 w_k1_MSFT_s013
                                 + 1.12347 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.0528 w_k1_ADM_s014
                                 + 1.30408 w_k1_AXP_s014
                                 + 0.96319 w_k1_DIS_s014
                                 + 0.944957 w_k1_FDX_s014
                                 + 1.7379 w_k1_IBM_s014
                                 + 0.929017 w_k1_INTC_s014
                                 + 1.12385 w_k1_KO_s014 + 1.28245 w_k1_MAT_s014
                                 + 0.953968 w_k1_MSFT_s014
                                 + 0.999052 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.930235 w_k1_ADM_s014
                                 + 0.75798 w_k1_AXP_s014
                                 + 0.747258 w_k1_DIS_s014
                                 + 0.740667 w_k1_FDX_s014
                                 + 0.989196 w_k1_IBM_s014
                                 + 0.854612 w_k1_INTC_s014
                                 + 0.845985 w_k1_KO_s014
                                 + 0.764602 w_k1_MAT_s014
                                 + 0.753246 w_k1_MSFT_s014
                                 + 0.802141 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.62445 w_k1_ADM_s014
                                 + 1.37424 w_k1_AXP_s014
                                 + 1.77373 w_k1_DIS_s014
                                 + 1.46514 w_k1_FDX_s014
                                 + 1.21471 w_k1_IBM_s014
                                 + 1.41457 w_k1_INTC_s014
                                 + 1.40608 w_k1_KO_s014
                                 + 0.939201 w_k1_MAT_s014
                                 + 1.53917 w_k1_MSFT_s014
                                 + 1.18155 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.00132 w_k1_ADM_s014
                                 + 0.88634 w_k1_AXP_s014
                                 + 0.688754 w_k1_DIS_s014
                                 + 0.890463 w_k1_FDX_s014
                                 + 0.747215 w_k1_IBM_s014
                                 + 0.954984 w_k1_INTC_s014
                                 + 0.718956 w_k1_KO_s014
                                 + 0.698856 w_k1_MAT_s014
                                 + 0.968096 w_k1_MSFT_s014
                                 + 0.858887 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.32156 w_k1_ADM_s014
                                 + 1.30712 w_k1_AXP_s014 + 1.2225 w_k1_DIS_s014
                                 + 1.10536 w_k1_FDX_s014
                                 + 1.33772 w_k1_IBM_s014
                                 + 1.18284 w_k1_INTC_s014
                                 + 1.14529 w_k1_KO_s014 + 1.20969 w_k1_MAT_s014
                                 + 1.05433 w_k1_MSFT_s014
                                 + 1.09787 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.18013 w_k1_ADM_s014
                                 + 0.82132 w_k1_AXP_s014
                                 + 1.02223 w_k1_DIS_s014
                                 + 0.978442 w_k1_FDX_s014
                                 + 1.12857 w_k1_IBM_s014
                                 + 1.00034 w_k1_INTC_s014
                                 + 1.10849 w_k1_KO_s014
                                 + 0.972083 w_k1_MAT_s014
                                 + 1.07157 w_k1_MSFT_s014
                                 + 1.01805 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.09477 w_k1_ADM_s014
                                 + 1.12352 w_k1_AXP_s014 + 1.214 w_k1_DIS_s014
                                 + 1.21755 w_k1_FDX_s014
                                 + 0.97065 w_k1_IBM_s014
                                 + 1.23512 w_k1_INTC_s014
                                 + 1.04814 w_k1_KO_s014
                                 + 0.732931 w_k1_MAT_s014
                                 + 1.1492 w_k1_MSFT_s014 + 1.13022 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.30177 w_k1_ADM_s014
                                 + 0.927325 w_k1_AXP_s014
                                 + 0.921087 w_k1_DIS_s014
                                 + 1.02993 w_k1_FDX_s014
                                 + 0.792368 w_k1_IBM_s014
                                 + 1.00911 w_k1_INTC_s014
                                 + 0.921148 w_k1_KO_s014
                                 + 0.732053 w_k1_MAT_s014
                                 + 0.985336 w_k1_MSFT_s014
                                 + 0.936913 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.780978 w_k1_ADM_s015
                                 + 1.09555 w_k1_AXP_s015
                                 + 0.779163 w_k1_DIS_s015
                                 + 0.84421 w_k1_FDX_s015
                                 + 0.990233 w_k1_IBM_s015
                                 + 1.06009 w_k1_INTC_s015
                                 + 0.93302 w_k1_KO_s015 + 1.21744 w_k1_MAT_s015
                                 + 0.892716 w_k1_MSFT_s015
                                 + 0.864574 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.729311 w_k1_ADM_s015
                                 + 1.20014 w_k1_AXP_s015
                                 + 0.981439 w_k1_DIS_s015
                                 + 0.99175 w_k1_FDX_s015
                                 + 1.70479 w_k1_IBM_s015
                                 + 1.40355 w_k1_INTC_s015
                                 + 1.52046 w_k1_KO_s015 + 1.95492 w_k1_MAT_s015
                                 + 1.00505 w_k1_MSFT_s015
                                 + 1.03222 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.537421 w_k1_ADM_s015
                                 + 0.8885 w_k1_AXP_s015
                                 + 0.591778 w_k1_DIS_s015
                                 + 0.676023 w_k1_FDX_s015
                                 + 0.9477 w_k1_IBM_s015
                                 + 0.772394 w_k1_INTC_s015
                                 + 0.799781 w_k1_KO_s015
                                 + 1.09056 w_k1_MAT_s015
                                 + 0.689922 w_k1_MSFT_s015
                                 + 0.885589 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.17834 w_k1_ADM_s015
                                 + 1.59508 w_k1_AXP_s015
                                 + 1.17706 w_k1_DIS_s015
                                 + 1.26377 w_k1_FDX_s015
                                 + 0.890141 w_k1_IBM_s015
                                 + 1.38197 w_k1_INTC_s015
                                 + 1.04427 w_k1_KO_s015
                                 + 0.940562 w_k1_MAT_s015
                                 + 1.06659 w_k1_MSFT_s015
                                 + 1.11879 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.984611 w_k1_ADM_s015
                                 + 1.03718 w_k1_AXP_s015
                                 + 1.49479 w_k1_DIS_s015
                                 + 1.20673 w_k1_FDX_s015
                                 + 0.76508 w_k1_IBM_s015
                                 + 1.29411 w_k1_INTC_s015
                                 + 1.23993 w_k1_KO_s015
                                 + 0.821785 w_k1_MAT_s015
                                 + 1.48688 w_k1_MSFT_s015
                                 + 1.16585 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.06746 w_k1_ADM_s015
                                 + 1.21554 w_k1_AXP_s015
                                 + 1.07721 w_k1_DIS_s015
                                 + 1.03266 w_k1_FDX_s015
                                 + 1.11116 w_k1_IBM_s015
                                 + 1.23284 w_k1_INTC_s015 + 1.1041 w_k1_KO_s015
                                 + 1.05897 w_k1_MAT_s015
                                 + 0.984927 w_k1_MSFT_s015
                                 + 1.0581 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.02408 w_k1_ADM_s015
                                 + 1.74772 w_k1_AXP_s015
                                 + 1.84612 w_k1_DIS_s015 + 1.5249 w_k1_FDX_s015
                                 + 1.38277 w_k1_IBM_s015
                                 + 1.40095 w_k1_INTC_s015
                                 + 1.69934 w_k1_KO_s015 + 1.99178 w_k1_MAT_s015
                                 + 1.36365 w_k1_MSFT_s015 + 1.6097 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.904982 w_k1_ADM_s015
                                 + 0.947877 w_k1_AXP_s015
                                 + 0.855945 w_k1_DIS_s015
                                 + 0.988985 w_k1_FDX_s015
                                 + 0.773004 w_k1_IBM_s015
                                 + 1.15433 w_k1_INTC_s015
                                 + 0.781911 w_k1_KO_s015
                                 + 0.714855 w_k1_MAT_s015
                                 + 0.890107 w_k1_MSFT_s015
                                 + 0.956169 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.15279 w_k1_ADM_s016
                                 + 1.40235 w_k1_AXP_s016
                                 + 1.20691 w_k1_DIS_s016
                                 + 1.13671 w_k1_FDX_s016
                                 + 0.957313 w_k1_IBM_s016
                                 + 1.10646 w_k1_INTC_s016
                                 + 1.02286 w_k1_KO_s016
                                 + 0.891534 w_k1_MAT_s016
                                 + 1.05791 w_k1_MSFT_s016
                                 + 1.04128 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.791175 w_k1_ADM_s016
                                 + 0.80885 w_k1_AXP_s016
                                 + 0.664009 w_k1_DIS_s016
                                 + 0.753678 w_k1_FDX_s016
                                 + 0.961777 w_k1_IBM_s016
                                 + 0.864784 w_k1_INTC_s016
                                 + 0.886225 w_k1_KO_s016
                                 + 1.03591 w_k1_MAT_s016
                                 + 0.943806 w_k1_MSFT_s016
                                 + 0.963008 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.17348 w_k1_ADM_s016
                                 + 1.23038 w_k1_AXP_s016
                                 + 1.73348 w_k1_DIS_s016
                                 + 1.28539 w_k1_FDX_s016
                                 + 1.39094 w_k1_IBM_s016
                                 + 1.19269 w_k1_INTC_s016
                                 + 1.32065 w_k1_KO_s016 + 1.13383 w_k1_MAT_s016
                                 + 1.51723 w_k1_MSFT_s016
                                 + 1.21795 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.686916 w_k1_ADM_s016
                                 + 0.62311 w_k1_AXP_s016
                                 + 0.564032 w_k1_DIS_s016
                                 + 0.702984 w_k1_FDX_s016
                                 + 0.518443 w_k1_IBM_s016
                                 + 0.830581 w_k1_INTC_s016
                                 + 0.560372 w_k1_KO_s016 + 0.347 w_k1_MAT_s016
                                 + 0.709282 w_k1_MSFT_s016 + 0.773 w_k1_PG_s016
                                 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.939753 w_k1_ADM_s016
                                 + 0.998552 w_k1_AXP_s016
                                 + 1.03124 w_k1_DIS_s016
                                 + 0.90465 w_k1_FDX_s016 + 1.1313 w_k1_IBM_s016
                                 + 1.00856 w_k1_INTC_s016
                                 + 0.944881 w_k1_KO_s016
                                 + 0.943643 w_k1_MAT_s016
                                 + 1.00503 w_k1_MSFT_s016
                                 + 1.04815 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.04008 w_k1_ADM_s016
                                 + 1.07754 w_k1_AXP_s016
                                 + 0.940992 w_k1_DIS_s016
                                 + 1.0572 w_k1_FDX_s016
                                 + 0.928634 w_k1_IBM_s016
                                 + 0.866669 w_k1_INTC_s016
                                 + 0.958197 w_k1_KO_s016
                                 + 0.797573 w_k1_MAT_s016
                                 + 0.955713 w_k1_MSFT_s016
                                 + 0.998089 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.957356 w_k1_ADM_s016
                                 + 0.766563 w_k1_AXP_s016
                                 + 0.819612 w_k1_DIS_s016
                                 + 0.928545 w_k1_FDX_s016
                                 + 0.676695 w_k1_IBM_s016
                                 + 0.794688 w_k1_INTC_s016
                                 + 0.770167 w_k1_KO_s016
                                 + 0.486147 w_k1_MAT_s016
                                 + 0.938681 w_k1_MSFT_s016
                                 + 0.940987 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.14438 w_k1_ADM_s016
                                 + 1.12467 w_k1_AXP_s016
                                 + 1.54324 w_k1_DIS_s016
                                 + 1.15591 w_k1_FDX_s016
                                 + 0.936645 w_k1_IBM_s016
                                 + 1.26045 w_k1_INTC_s016
                                 + 1.30049 w_k1_KO_s016
                                 + 0.765678 w_k1_MAT_s016
                                 + 1.25038 w_k1_MSFT_s016 + 1.1159 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.0641 w_k1_ADM_s017
                                 + 1.14955 w_k1_AXP_s017 + 1.1433 w_k1_DIS_s017
                                 + 1.09684 w_k1_FDX_s017
                                 + 1.21316 w_k1_IBM_s017
                                 + 1.03134 w_k1_INTC_s017
                                 + 1.13118 w_k1_KO_s017 + 1.27318 w_k1_MAT_s017
                                 + 1.05638 w_k1_MSFT_s017
                                 + 1.08164 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.783432 w_k1_ADM_s017
                                 + 0.767142 w_k1_AXP_s017
                                 + 0.938909 w_k1_DIS_s017
                                 + 0.929668 w_k1_FDX_s017
                                 + 0.899588 w_k1_IBM_s017
                                 + 0.828307 w_k1_INTC_s017
                                 + 0.842606 w_k1_KO_s017
                                 + 1.00511 w_k1_MAT_s017
                                 + 1.03592 w_k1_MSFT_s017
                                 + 1.05132 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.749447 w_k1_ADM_s017
                                 + 0.93534 w_k1_AXP_s017
                                 + 0.827588 w_k1_DIS_s017
                                 + 0.981015 w_k1_FDX_s017
                                 + 0.735466 w_k1_IBM_s017
                                 + 1.02234 w_k1_INTC_s017
                                 + 0.834033 w_k1_KO_s017
                                 + 1.04143 w_k1_MAT_s017
                                 + 0.9404 w_k1_MSFT_s017
                                 + 0.971153 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.20047 w_k1_ADM_s017
                                 + 1.24118 w_k1_AXP_s017
                                 + 1.37226 w_k1_DIS_s017
                                 + 1.31723 w_k1_FDX_s017 + 1.0349 w_k1_IBM_s017
                                 + 1.24585 w_k1_INTC_s017
                                 + 1.12247 w_k1_KO_s017 + 1.12699 w_k1_MAT_s017
                                 + 1.41547 w_k1_MSFT_s017
                                 + 1.14654 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.833107 w_k1_ADM_s017
                                 + 1.16892 w_k1_AXP_s017
                                 + 1.06661 w_k1_DIS_s017
                                 + 1.06758 w_k1_FDX_s017
                                 + 0.955611 w_k1_IBM_s017
                                 + 1.15028 w_k1_INTC_s017
                                 + 0.882708 w_k1_KO_s017
                                 + 1.04432 w_k1_MAT_s017
                                 + 1.10717 w_k1_MSFT_s017
                                 + 1.03696 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.15406 w_k1_ADM_s017
                                 + 1.34758 w_k1_AXP_s017 + 1.3086 w_k1_DIS_s017
                                 + 1.18788 w_k1_FDX_s017
                                 + 1.76694 w_k1_IBM_s017
                                 + 0.953625 w_k1_INTC_s017
                                 + 1.42075 w_k1_KO_s017 + 2.30812 w_k1_MAT_s017
                                 + 1.02793 w_k1_MSFT_s017
                                 + 1.34548 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.11792 w_k1_ADM_s017
                                 + 1.52625 w_k1_AXP_s017
                                 + 1.20802 w_k1_DIS_s017
                                 + 1.26248 w_k1_FDX_s017
                                 + 1.38125 w_k1_IBM_s017
                                 + 1.05093 w_k1_INTC_s017
                                 + 1.11434 w_k1_KO_s017 + 1.76051 w_k1_MAT_s017
                                 + 1.06919 w_k1_MSFT_s017
                                 + 1.22852 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.818618 w_k1_ADM_s017
                                 + 0.856034 w_k1_AXP_s017
                                 + 0.734166 w_k1_DIS_s017
                                 + 0.849691 w_k1_FDX_s017
                                 + 1.14413 w_k1_IBM_s017
                                 + 0.788428 w_k1_INTC_s017
                                 + 0.883029 w_k1_KO_s017
                                 + 1.30431 w_k1_MAT_s017
                                 + 0.882947 w_k1_MSFT_s017
                                 + 0.995034 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.12734 w_k1_ADM_s100
                                 + 1.05516 w_k1_AXP_s100 + 1.2355 w_k1_DIS_s100
                                 + 1.01354 w_k1_FDX_s100
                                 + 1.12448 w_k1_IBM_s100
                                 + 1.20328 w_k1_INTC_s100
                                 + 1.10061 w_k1_KO_s100 + 1.66447 w_k1_MAT_s100
                                 + 1.09384 w_k1_MSFT_s100
                                 + 1.07727 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.870957 w_k1_ADM_s100
                                 + 0.968178 w_k1_AXP_s100
                                 + 1.00803 w_k1_DIS_s100
                                 + 1.11544 w_k1_FDX_s100
                                 + 0.666396 w_k1_IBM_s100
                                 + 1.04035 w_k1_INTC_s100
                                 + 0.941849 w_k1_KO_s100
                                 + 0.843373 w_k1_MAT_s100
                                 + 1.0405 w_k1_MSFT_s100
                                 + 0.971407 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.910296 w_k1_ADM_s100
                                 + 1.09637 w_k1_AXP_s100
                                 + 1.10404 w_k1_DIS_s100
                                 + 0.812697 w_k1_FDX_s100
                                 + 1.6543 w_k1_IBM_s100
                                 + 1.11057 w_k1_INTC_s100
                                 + 1.21211 w_k1_KO_s100 + 1.52614 w_k1_MAT_s100
                                 + 1.02015 w_k1_MSFT_s100
                                 + 0.949209 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.34069 w_k1_ADM_s100
                                 + 1.64706 w_k1_AXP_s100
                                 + 1.68758 w_k1_DIS_s100
                                 + 1.21148 w_k1_FDX_s100
                                 + 1.46612 w_k1_IBM_s100
                                 + 1.19659 w_k1_INTC_s100
                                 + 1.53457 w_k1_KO_s100 + 2.16731 w_k1_MAT_s100
                                 + 1.1175 w_k1_MSFT_s100 + 1.30646 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.21701 w_k1_ADM_s100
                                 + 1.26631 w_k1_AXP_s100
                                 + 1.34401 w_k1_DIS_s100
                                 + 1.24537 w_k1_FDX_s100
                                 + 0.971556 w_k1_IBM_s100
                                 + 1.31329 w_k1_INTC_s100
                                 + 1.07315 w_k1_KO_s100 + 1.00245 w_k1_MAT_s100
                                 + 1.13966 w_k1_MSFT_s100
                                 + 1.09205 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.935599 w_k1_ADM_s100
                                 + 0.878229 w_k1_AXP_s100
                                 + 0.78355 w_k1_DIS_s100
                                 + 0.861315 w_k1_FDX_s100
                                 + 0.645172 w_k1_IBM_s100
                                 + 1.11542 w_k1_INTC_s100
                                 + 0.882856 w_k1_KO_s100
                                 + 0.637729 w_k1_MAT_s100
                                 + 0.911597 w_k1_MSFT_s100
                                 + 0.821782 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.817613 w_k1_ADM_s100
                                 + 0.827152 w_k1_AXP_s100
                                 + 0.823294 w_k1_DIS_s100
                                 + 0.793812 w_k1_FDX_s100
                                 + 1.06808 w_k1_IBM_s100
                                 + 1.07472 w_k1_INTC_s100
                                 + 0.923046 w_k1_KO_s100
                                 + 1.12136 w_k1_MAT_s100
                                 + 0.855804 w_k1_MSFT_s100
                                 + 0.898385 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.27597 w_k1_ADM_s100
                                 + 1.62738 w_k1_AXP_s100
                                 + 1.82025 w_k1_DIS_s100
                                 + 1.49494 w_k1_FDX_s100
                                 + 0.741121 w_k1_IBM_s100
                                 + 1.61155 w_k1_INTC_s100
                                 + 1.16536 w_k1_KO_s100
                                 + 0.834132 w_k1_MAT_s100
                                 + 1.43248 w_k1_MSFT_s100
                                 + 1.20594 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.28419 w_k1_ADM_s101
                                 + 1.36535 w_k1_AXP_s101
                                 + 1.28223 w_k1_DIS_s101
                                 + 1.13182 w_k1_FDX_s101
                                 + 1.08935 w_k1_IBM_s101
                                 + 1.10832 w_k1_INTC_s101
                                 + 1.17458 w_k1_KO_s101 + 1.51946 w_k1_MAT_s101
                                 + 1.06971 w_k1_MSFT_s101
                                 + 1.09531 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.05757 w_k1_ADM_s101
                                 + 1.02528 w_k1_AXP_s101
                                 + 1.05369 w_k1_DIS_s101
                                 + 0.983188 w_k1_FDX_s101
                                 + 0.89723 w_k1_IBM_s101
                                 + 1.04016 w_k1_INTC_s101
                                 + 0.999348 w_k1_KO_s101 + 1.0687 w_k1_MAT_s101
                                 + 1.08512 w_k1_MSFT_s101
                                 + 0.947085 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.23565 w_k1_ADM_s101
                                 + 1.21456 w_k1_AXP_s101 + 1.501 w_k1_DIS_s101
                                 + 1.29793 w_k1_FDX_s101
                                 + 0.84225 w_k1_IBM_s101
                                 + 1.19851 w_k1_INTC_s101
                                 + 1.16387 w_k1_KO_s101 + 1.11544 w_k1_MAT_s101
                                 + 1.18276 w_k1_MSFT_s101 + 1.1307 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.801182 w_k1_ADM_s101
                                 + 1.08678 w_k1_AXP_s101 + 1.0991 w_k1_DIS_s101
                                 + 0.842987 w_k1_FDX_s101
                                 + 1.63483 w_k1_IBM_s101
                                 + 1.10584 w_k1_INTC_s101
                                 + 1.29386 w_k1_KO_s101 + 1.51828 w_k1_MAT_s101
                                 + 0.946149 w_k1_MSFT_s101
                                 + 0.982271 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.970847 w_k1_ADM_s101
                                 + 0.898732 w_k1_AXP_s101
                                 + 0.784675 w_k1_DIS_s101
                                 + 0.988518 w_k1_FDX_s101
                                 + 0.620541 w_k1_IBM_s101
                                 + 0.968011 w_k1_INTC_s101
                                 + 0.809093 w_k1_KO_s101
                                 + 0.893427 w_k1_MAT_s101
                                 + 0.94757 w_k1_MSFT_s101
                                 + 0.897415 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.79101 w_k1_ADM_s101
                                 + 0.804499 w_k1_AXP_s101
                                 + 0.593175 w_k1_DIS_s101
                                 + 0.678042 w_k1_FDX_s101
                                 + 0.993492 w_k1_IBM_s101
                                 + 0.733723 w_k1_INTC_s101
                                 + 0.76655 w_k1_KO_s101 + 1.28343 w_k1_MAT_s101
                                 + 0.64629 w_k1_MSFT_s101
                                 + 0.776971 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.982274 w_k1_ADM_s101
                                 + 1.27475 w_k1_AXP_s101
                                 + 1.38683 w_k1_DIS_s101 + 1.0686 w_k1_FDX_s101
                                 + 1.40788 w_k1_IBM_s101
                                 + 1.12345 w_k1_INTC_s101
                                 + 1.23569 w_k1_KO_s101 + 1.60826 w_k1_MAT_s101
                                 + 1.1199 w_k1_MSFT_s101
                                 + 0.949873 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.88179 w_k1_ADM_s101
                                 + 0.862292 w_k1_AXP_s101
                                 + 0.863038 w_k1_DIS_s101
                                 + 0.80233 w_k1_FDX_s101
                                 + 0.948147 w_k1_IBM_s101
                                 + 0.843232 w_k1_INTC_s101
                                 + 0.949291 w_k1_KO_s101
                                 + 0.886218 w_k1_MAT_s101
                                 + 0.729913 w_k1_MSFT_s101
                                 + 0.839062 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.00111 w_k1_ADM_s102
                                 + 1.28149 w_k1_AXP_s102
                                 + 1.05596 w_k1_DIS_s102
                                 + 1.04349 w_k1_FDX_s102
                                 + 1.14843 w_k1_IBM_s102
                                 + 1.0031 w_k1_INTC_s102 + 1.04241 w_k1_KO_s102
                                 + 1.22828 w_k1_MAT_s102
                                 + 0.829479 w_k1_MSFT_s102
                                 + 0.961876 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.9096 w_k1_ADM_s102
                                 + 0.705644 w_k1_AXP_s102
                                 + 0.695461 w_k1_DIS_s102
                                 + 0.764933 w_k1_FDX_s102
                                 + 0.653023 w_k1_IBM_s102
                                 + 0.691835 w_k1_INTC_s102
                                 + 0.734644 w_k1_KO_s102
                                 + 0.541125 w_k1_MAT_s102
                                 + 0.775407 w_k1_MSFT_s102
                                 + 0.789111 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.01531 w_k1_ADM_s102
                                 + 1.03941 w_k1_AXP_s102
                                 + 1.04518 w_k1_DIS_s102
                                 + 0.851356 w_k1_FDX_s102
                                 + 1.26856 w_k1_IBM_s102
                                 + 0.974215 w_k1_INTC_s102
                                 + 1.14206 w_k1_KO_s102 + 1.25797 w_k1_MAT_s102
                                 + 0.963405 w_k1_MSFT_s102
                                 + 0.873099 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.896342 w_k1_ADM_s102
                                 + 0.916161 w_k1_AXP_s102
                                 + 0.949475 w_k1_DIS_s102
                                 + 0.882699 w_k1_FDX_s102
                                 + 0.81698 w_k1_IBM_s102
                                 + 1.00185 w_k1_INTC_s102
                                 + 0.864441 w_k1_KO_s102
                                 + 0.678963 w_k1_MAT_s102
                                 + 0.864228 w_k1_MSFT_s102
                                 + 0.888069 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.47335 w_k1_ADM_s102
                                 + 1.57621 w_k1_AXP_s102 + 1.6786 w_k1_DIS_s102
                                 + 1.29237 w_k1_FDX_s102
                                 + 1.28956 w_k1_IBM_s102
                                 + 1.31267 w_k1_INTC_s102
                                 + 1.46131 w_k1_KO_s102 + 1.3328 w_k1_MAT_s102
                                 + 1.2215 w_k1_MSFT_s102 + 1.17523 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.781666 w_k1_ADM_s102
                                 + 0.789342 w_k1_AXP_s102
                                 + 0.580298 w_k1_DIS_s102
                                 + 0.666848 w_k1_FDX_s102
                                 + 1.08247 w_k1_IBM_s102
                                 + 0.726024 w_k1_INTC_s102
                                 + 0.835796 w_k1_KO_s102
                                 + 1.11258 w_k1_MAT_s102
                                 + 0.701978 w_k1_MSFT_s102
                                 + 0.817494 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.13264 w_k1_ADM_s102
                                 + 1.04325 w_k1_AXP_s102
                                 + 1.32999 w_k1_DIS_s102
                                 + 1.09119 w_k1_FDX_s102
                                 + 0.924119 w_k1_IBM_s102
                                 + 1.12399 w_k1_INTC_s102
                                 + 1.19226 w_k1_KO_s102
                                 + 0.784907 w_k1_MAT_s102
                                 + 1.07613 w_k1_MSFT_s102
                                 + 0.99097 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.45291 w_k1_ADM_s102
                                 + 1.22054 w_k1_AXP_s102
                                 + 1.20581 w_k1_DIS_s102
                                 + 1.24884 w_k1_FDX_s102
                                 + 0.733506 w_k1_IBM_s102
                                 + 0.883905 w_k1_INTC_s102
                                 + 0.937845 w_k1_KO_s102
                                 + 0.584872 w_k1_MAT_s102
                                 + 1.0178 w_k1_MSFT_s102 + 1.0251 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.16932 w_k1_ADM_s103
                                 + 1.35813 w_k1_AXP_s103
                                 + 1.59383 w_k1_DIS_s103
                                 + 1.36938 w_k1_FDX_s103
                                 + 1.40581 w_k1_IBM_s103
                                 + 1.3384 w_k1_INTC_s103 + 1.44852 w_k1_KO_s103
                                 + 1.34083 w_k1_MAT_s103
                                 + 1.36967 w_k1_MSFT_s103 + 1.2975 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.853167 w_k1_ADM_s103
                                 + 0.772729 w_k1_AXP_s103
                                 + 0.920045 w_k1_DIS_s103
                                 + 0.945426 w_k1_FDX_s103
                                 + 0.722509 w_k1_IBM_s103
                                 + 0.944297 w_k1_INTC_s103
                                 + 0.751366 w_k1_KO_s103
                                 + 0.661557 w_k1_MAT_s103
                                 + 1.02053 w_k1_MSFT_s103
                                 + 0.941739 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.954104 w_k1_ADM_s103
                                 + 1.00642 w_k1_AXP_s103
                                 + 0.674423 w_k1_DIS_s103
                                 + 0.889427 w_k1_FDX_s103
                                 + 0.842957 w_k1_IBM_s103
                                 + 0.85039 w_k1_INTC_s103
                                 + 0.865653 w_k1_KO_s103
                                 + 1.05519 w_k1_MAT_s103
                                 + 0.93414 w_k1_MSFT_s103
                                 + 1.01375 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.862506 w_k1_ADM_s103
                                 + 0.853434 w_k1_AXP_s103
                                 + 0.811828 w_k1_DIS_s103
                                 + 0.761299 w_k1_FDX_s103
                                 + 1.34269 w_k1_IBM_s103
                                 + 0.876262 w_k1_INTC_s103
                                 + 0.912436 w_k1_KO_s103
                                 + 0.877973 w_k1_MAT_s103
                                 + 0.808796 w_k1_MSFT_s103
                                 + 0.970448 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.939006 w_k1_ADM_s103
                                 + 1.2413 w_k1_AXP_s103 + 1.29792 w_k1_DIS_s103
                                 + 1.14711 w_k1_FDX_s103
                                 + 1.19024 w_k1_IBM_s103
                                 + 1.25522 w_k1_INTC_s103
                                 + 1.23358 w_k1_KO_s103 + 1.36051 w_k1_MAT_s103
                                 + 1.36126 w_k1_MSFT_s103
                                 + 1.15155 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.871094 w_k1_ADM_s103
                                 + 0.837635 w_k1_AXP_s103
                                 + 0.951617 w_k1_DIS_s103
                                 + 0.884757 w_k1_FDX_s103
                                 + 0.968561 w_k1_IBM_s103
                                 + 1.06952 w_k1_INTC_s103
                                 + 0.949172 w_k1_KO_s103
                                 + 0.706345 w_k1_MAT_s103
                                 + 1.00535 w_k1_MSFT_s103
                                 + 0.89618 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.942602 w_k1_ADM_s103
                                 + 1.09724 w_k1_AXP_s103
                                 + 0.870978 w_k1_DIS_s103
                                 + 0.957938 w_k1_FDX_s103
                                 + 1.0956 w_k1_IBM_s103
                                 + 1.07482 w_k1_INTC_s103
                                 + 0.997477 w_k1_KO_s103
                                 + 1.17223 w_k1_MAT_s103
                                 + 0.955952 w_k1_MSFT_s103
                                 + 1.07168 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.741189 w_k1_ADM_s103
                                 + 0.599442 w_k1_AXP_s103
                                 + 0.591315 w_k1_DIS_s103
                                 + 0.725346 w_k1_FDX_s103
                                 + 0.742883 w_k1_IBM_s103
                                 + 0.738269 w_k1_INTC_s103
                                 + 0.782261 w_k1_KO_s103
                                 + 0.712314 w_k1_MAT_s103
                                 + 0.791772 w_k1_MSFT_s103
                                 + 0.815007 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.11333 w_k1_ADM_s104
                                 + 0.773251 w_k1_AXP_s104
                                 + 0.830238 w_k1_DIS_s104
                                 + 0.917549 w_k1_FDX_s104
                                 + 0.924324 w_k1_IBM_s104
                                 + 1.04968 w_k1_INTC_s104 + 1.0082 w_k1_KO_s104
                                 + 0.914389 w_k1_MAT_s104
                                 + 1.03003 w_k1_MSFT_s104
                                 + 0.863315 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.32849 w_k1_ADM_s104
                                 + 0.843814 w_k1_AXP_s104
                                 + 1.15458 w_k1_DIS_s104
                                 + 1.19682 w_k1_FDX_s104
                                 + 0.645432 w_k1_IBM_s104
                                 + 1.1761 w_k1_INTC_s104 + 1.05685 w_k1_KO_s104
                                 + 0.527709 w_k1_MAT_s104
                                 + 1.33265 w_k1_MSFT_s104
                                 + 0.95228 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.886698 w_k1_ADM_s104
                                 + 0.879437 w_k1_AXP_s104
                                 + 0.708348 w_k1_DIS_s104
                                 + 0.749546 w_k1_FDX_s104
                                 + 1.25132 w_k1_IBM_s104
                                 + 0.980011 w_k1_INTC_s104
                                 + 1.05332 w_k1_KO_s104 + 1.02033 w_k1_MAT_s104
                                 + 0.845972 w_k1_MSFT_s104
                                 + 0.834163 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.817098 w_k1_ADM_s104
                                 + 0.728348 w_k1_AXP_s104
                                 + 0.627364 w_k1_DIS_s104
                                 + 0.797171 w_k1_FDX_s104
                                 + 0.616061 w_k1_IBM_s104
                                 + 1.08765 w_k1_INTC_s104
                                 + 0.71951 w_k1_KO_s104
                                 + 0.579932 w_k1_MAT_s104
                                 + 0.933656 w_k1_MSFT_s104
                                 + 0.733844 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.15795 w_k1_ADM_s104
                                 + 1.44249 w_k1_AXP_s104
                                 + 1.18839 w_k1_DIS_s104
                                 + 1.12955 w_k1_FDX_s104
                                 + 0.977505 w_k1_IBM_s104
                                 + 1.12419 w_k1_INTC_s104 + 1.0373 w_k1_KO_s104
                                 + 1.39019 w_k1_MAT_s104
                                 + 1.16004 w_k1_MSFT_s104 + 1.0439 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.53531 w_k1_ADM_s104
                                 + 1.361 w_k1_AXP_s104 + 1.58971 w_k1_DIS_s104
                                 + 1.46873 w_k1_FDX_s104
                                 + 1.28445 w_k1_IBM_s104
                                 + 1.92115 w_k1_INTC_s104
                                 + 1.33456 w_k1_KO_s104 + 1.13508 w_k1_MAT_s104
                                 + 1.47445 w_k1_MSFT_s104
                                 + 1.11136 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.12643 w_k1_ADM_s104
                                 + 0.952296 w_k1_AXP_s104
                                 + 1.10242 w_k1_DIS_s104
                                 + 1.05589 w_k1_FDX_s104
                                 + 1.17441 w_k1_IBM_s104
                                 + 1.19887 w_k1_INTC_s104
                                 + 1.18629 w_k1_KO_s104 + 1.27997 w_k1_MAT_s104
                                 + 1.06424 w_k1_MSFT_s104
                                 + 0.996581 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.03327 w_k1_ADM_s104
                                 + 1.08246 w_k1_AXP_s104
                                 + 1.17445 w_k1_DIS_s104
                                 + 1.05474 w_k1_FDX_s104
                                 + 1.57341 w_k1_IBM_s104
                                 + 1.29314 w_k1_INTC_s104 + 1.4278 w_k1_KO_s104
                                 + 1.64516 w_k1_MAT_s104
                                 + 1.11875 w_k1_MSFT_s104
                                 + 1.01203 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.778445 w_k1_ADM_s105
                                 + 0.720009 w_k1_AXP_s105
                                 + 0.825664 w_k1_DIS_s105
                                 + 0.777943 w_k1_FDX_s105
                                 + 0.796458 w_k1_IBM_s105
                                 + 0.824259 w_k1_INTC_s105
                                 + 0.929059 w_k1_KO_s105
                                 + 0.918404 w_k1_MAT_s105
                                 + 0.90494 w_k1_MSFT_s105
                                 + 0.976131 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.03926 w_k1_ADM_s105
                                 + 1.07661 w_k1_AXP_s105
                                 + 0.783525 w_k1_DIS_s105
                                 + 0.806394 w_k1_FDX_s105
                                 + 1.26257 w_k1_IBM_s105
                                 + 1.06615 w_k1_INTC_s105
                                 + 1.04154 w_k1_KO_s105 + 1.5858 w_k1_MAT_s105
                                 + 0.788794 w_k1_MSFT_s105
                                 + 1.07698 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.838471 w_k1_ADM_s105
                                 + 0.909644 w_k1_AXP_s105
                                 + 0.796112 w_k1_DIS_s105
                                 + 0.90038 w_k1_FDX_s105
                                 + 0.747227 w_k1_IBM_s105
                                 + 0.998249 w_k1_INTC_s105
                                 + 0.803592 w_k1_KO_s105
                                 + 0.767853 w_k1_MAT_s105
                                 + 0.919541 w_k1_MSFT_s105
                                 + 0.880295 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.912854 w_k1_ADM_s105
                                 + 0.974632 w_k1_AXP_s105
                                 + 1.15547 w_k1_DIS_s105
                                 + 0.95782 w_k1_FDX_s105
                                 + 1.35284 w_k1_IBM_s105
                                 + 0.91382 w_k1_INTC_s105
                                 + 1.19977 w_k1_KO_s105 + 1.77321 w_k1_MAT_s105
                                 + 1.01434 w_k1_MSFT_s105
                                 + 1.03103 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.08783 w_k1_ADM_s105
                                 + 1.03186 w_k1_AXP_s105
                                 + 1.36798 w_k1_DIS_s105
                                 + 1.19176 w_k1_FDX_s105
                                 + 0.982367 w_k1_IBM_s105
                                 + 1.19758 w_k1_INTC_s105
                                 + 1.06652 w_k1_KO_s105 + 1.16242 w_k1_MAT_s105
                                 + 1.31857 w_k1_MSFT_s105
                                 + 1.09225 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.764594 w_k1_ADM_s105
                                 + 0.87658 w_k1_AXP_s105
                                 + 0.585469 w_k1_DIS_s105
                                 + 0.725347 w_k1_FDX_s105
                                 + 0.938318 w_k1_IBM_s105
                                 + 0.692911 w_k1_INTC_s105
                                 + 0.729917 w_k1_KO_s105
                                 + 1.37374 w_k1_MAT_s105
                                 + 0.678602 w_k1_MSFT_s105
                                 + 0.836816 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.30219 w_k1_ADM_s105
                                 + 1.65236 w_k1_AXP_s105
                                 + 1.76911 w_k1_DIS_s105 + 1.3594 w_k1_FDX_s105
                                 + 1.53341 w_k1_IBM_s105
                                 + 1.2908 w_k1_INTC_s105 + 1.70139 w_k1_KO_s105
                                 + 1.84522 w_k1_MAT_s105
                                 + 1.40107 w_k1_MSFT_s105
                                 + 1.31849 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.99667 w_k1_ADM_s105
                                 + 0.913859 w_k1_AXP_s105
                                 + 0.735302 w_k1_DIS_s105
                                 + 0.887385 w_k1_FDX_s105
                                 + 1.02017 w_k1_IBM_s105
                                 + 0.89488 w_k1_INTC_s105
                                 + 0.844679 w_k1_KO_s105
                                 + 1.18619 w_k1_MAT_s105
                                 + 0.956419 w_k1_MSFT_s105
                                 + 1.03349 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.06853 w_k1_ADM_s106
                                 + 1.0808 w_k1_AXP_s106 + 1.20066 w_k1_DIS_s106
                                 + 1.14794 w_k1_FDX_s106
                                 + 1.18857 w_k1_IBM_s106
                                 + 1.45024 w_k1_INTC_s106
                                 + 1.11548 w_k1_KO_s106
                                 + 0.980898 w_k1_MAT_s106
                                 + 1.15413 w_k1_MSFT_s106
                                 + 1.13873 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.22855 w_k1_ADM_s106
                                 + 1.45548 w_k1_AXP_s106
                                 + 1.74143 w_k1_DIS_s106
                                 + 1.47172 w_k1_FDX_s106
                                 + 0.897222 w_k1_IBM_s106
                                 + 1.68638 w_k1_INTC_s106
                                 + 1.18284 w_k1_KO_s106
                                 + 0.764015 w_k1_MAT_s106
                                 + 1.38923 w_k1_MSFT_s106
                                 + 1.26254 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.764224 w_k1_ADM_s106
                                 + 0.809142 w_k1_AXP_s106
                                 + 0.738484 w_k1_DIS_s106
                                 + 0.879283 w_k1_FDX_s106
                                 + 0.768392 w_k1_IBM_s106
                                 + 0.985206 w_k1_INTC_s106
                                 + 0.797894 w_k1_KO_s106
                                 + 0.661047 w_k1_MAT_s106
                                 + 0.830149 w_k1_MSFT_s106
                                 + 0.996439 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.759602 w_k1_ADM_s106
                                 + 0.620782 w_k1_AXP_s106
                                 + 0.587646 w_k1_DIS_s106
                                 + 0.697098 w_k1_FDX_s106
                                 + 0.577911 w_k1_IBM_s106
                                 + 0.961422 w_k1_INTC_s106
                                 + 0.610004 w_k1_KO_s106
                                 + 0.496244 w_k1_MAT_s106
                                 + 0.804778 w_k1_MSFT_s106
                                 + 0.808936 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.991091 w_k1_ADM_s106
                                 + 0.85013 w_k1_AXP_s106
                                 + 0.849955 w_k1_DIS_s106
                                 + 0.878124 w_k1_FDX_s106
                                 + 1.12965 w_k1_IBM_s106
                                 + 1.19394 w_k1_INTC_s106
                                 + 0.958666 w_k1_KO_s106
                                 + 0.866965 w_k1_MAT_s106
                                 + 0.949579 w_k1_MSFT_s106
                                 + 0.926941 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.845902 w_k1_ADM_s106
                                 + 0.684821 w_k1_AXP_s106
                                 + 0.63238 w_k1_DIS_s106
                                 + 0.71177 w_k1_FDX_s106
                                 + 0.951658 w_k1_IBM_s106
                                 + 0.967954 w_k1_INTC_s106
                                 + 0.830379 w_k1_KO_s106
                                 + 0.635903 w_k1_MAT_s106
                                 + 0.762738 w_k1_MSFT_s106
                                 + 0.869013 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.30208 w_k1_ADM_s106
                                 + 1.55701 w_k1_AXP_s106 + 1.1343 w_k1_DIS_s106
                                 + 1.13508 w_k1_FDX_s106
                                 + 1.43115 w_k1_IBM_s106
                                 + 1.14367 w_k1_INTC_s106
                                 + 1.04582 w_k1_KO_s106 + 1.29928 w_k1_MAT_s106
                                 + 1.02011 w_k1_MSFT_s106
                                 + 1.08024 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.05965 w_k1_ADM_s106
                                 + 0.89984 w_k1_AXP_s106
                                 + 0.919699 w_k1_DIS_s106
                                 + 1.07301 w_k1_FDX_s106
                                 + 0.605134 w_k1_IBM_s106
                                 + 1.17989 w_k1_INTC_s106
                                 + 0.820708 w_k1_KO_s106
                                 + 0.51805 w_k1_MAT_s106
                                 + 0.941192 w_k1_MSFT_s106
                                 + 1.00555 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.14237 w_k1_ADM_s107
                                 + 0.733452 w_k1_AXP_s107
                                 + 0.716077 w_k1_DIS_s107
                                 + 0.852977 w_k1_FDX_s107
                                 + 0.745302 w_k1_IBM_s107
                                 + 0.843818 w_k1_INTC_s107
                                 + 0.823092 w_k1_KO_s107
                                 + 0.681022 w_k1_MAT_s107
                                 + 0.902309 w_k1_MSFT_s107
                                 + 0.819226 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.37791 w_k1_ADM_s107
                                 + 1.38221 w_k1_AXP_s107
                                 + 1.28022 w_k1_DIS_s107 + 1.1897 w_k1_FDX_s107
                                 + 1.15498 w_k1_IBM_s107
                                 + 1.13857 w_k1_INTC_s107 + 1.0915 w_k1_KO_s107
                                 + 0.941138 w_k1_MAT_s107
                                 + 1.03617 w_k1_MSFT_s107
                                 + 0.995736 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.999954 w_k1_ADM_s107
                                 + 0.730441 w_k1_AXP_s107
                                 + 0.639573 w_k1_DIS_s107
                                 + 0.854515 w_k1_FDX_s107
                                 + 0.577202 w_k1_IBM_s107
                                 + 0.925348 w_k1_INTC_s107
                                 + 0.734423 w_k1_KO_s107
                                 + 0.640756 w_k1_MAT_s107
                                 + 0.850274 w_k1_MSFT_s107
                                 + 0.792111 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.850985 w_k1_ADM_s107
                                 + 0.834563 w_k1_AXP_s107
                                 + 0.640536 w_k1_DIS_s107
                                 + 0.847264 w_k1_FDX_s107
                                 + 1.06613 w_k1_IBM_s107
                                 + 0.821805 w_k1_INTC_s107
                                 + 0.757353 w_k1_KO_s107 + 1.0722 w_k1_MAT_s107
                                 + 0.792357 w_k1_MSFT_s107
                                 + 0.76153 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.898965 w_k1_ADM_s107
                                 + 0.976565 w_k1_AXP_s107
                                 + 1.11564 w_k1_DIS_s107
                                 + 1.17042 w_k1_FDX_s107
                                 + 0.825386 w_k1_IBM_s107
                                 + 0.992845 w_k1_INTC_s107
                                 + 0.977474 w_k1_KO_s107
                                 + 0.871948 w_k1_MAT_s107
                                 + 0.971512 w_k1_MSFT_s107
                                 + 0.969411 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.32194 w_k1_ADM_s107
                                 + 1.57307 w_k1_AXP_s107 + 1.2718 w_k1_DIS_s107
                                 + 1.42737 w_k1_FDX_s107
                                 + 0.874897 w_k1_IBM_s107
                                 + 1.13197 w_k1_INTC_s107
                                 + 1.11635 w_k1_KO_s107 + 1.08134 w_k1_MAT_s107
                                 + 1.13265 w_k1_MSFT_s107
                                 + 1.07247 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.09886 w_k1_ADM_s107
                                 + 0.974534 w_k1_AXP_s107
                                 + 0.988975 w_k1_DIS_s107
                                 + 0.978973 w_k1_FDX_s107
                                 + 0.987874 w_k1_IBM_s107
                                 + 0.958114 w_k1_INTC_s107
                                 + 0.899825 w_k1_KO_s107
                                 + 0.883106 w_k1_MAT_s107
                                 + 0.904239 w_k1_MSFT_s107
                                 + 0.866199 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.30626 w_k1_ADM_s107
                                 + 1.29411 w_k1_AXP_s107
                                 + 1.22698 w_k1_DIS_s107
                                 + 1.14768 w_k1_FDX_s107
                                 + 1.55664 w_k1_IBM_s107
                                 + 1.16668 w_k1_INTC_s107
                                 + 1.38531 w_k1_KO_s107 + 1.61653 w_k1_MAT_s107
                                 + 1.17992 w_k1_MSFT_s107
                                 + 1.03318 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.13637 w_k1_ADM_s110
                                 + 1.28142 w_k1_AXP_s110
                                 + 1.41275 w_k1_DIS_s110
                                 + 1.32135 w_k1_FDX_s110
                                 + 1.01307 w_k1_IBM_s110
                                 + 1.13329 w_k1_INTC_s110
                                 + 1.03264 w_k1_KO_s110
                                 + 0.841001 w_k1_MAT_s110
                                 + 1.14976 w_k1_MSFT_s110
                                 + 1.03561 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.07512 w_k1_ADM_s110
                                 + 0.988889 w_k1_AXP_s110
                                 + 0.979244 w_k1_DIS_s110
                                 + 1.0316 w_k1_FDX_s110
                                 + 0.865288 w_k1_IBM_s110
                                 + 1.06496 w_k1_INTC_s110
                                 + 0.879809 w_k1_KO_s110
                                 + 0.66864 w_k1_MAT_s110
                                 + 0.998746 w_k1_MSFT_s110
                                 + 0.985037 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.751015 w_k1_ADM_s110
                                 + 0.785612 w_k1_AXP_s110
                                 + 0.535148 w_k1_DIS_s110
                                 + 0.765223 w_k1_FDX_s110
                                 + 0.757928 w_k1_IBM_s110
                                 + 0.79977 w_k1_INTC_s110
                                 + 0.627116 w_k1_KO_s110 + 0.7802 w_k1_MAT_s110
                                 + 0.725369 w_k1_MSFT_s110
                                 + 0.767005 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.03908 w_k1_ADM_s110
                                 + 1.04255 w_k1_AXP_s110
                                 + 0.980227 w_k1_DIS_s110
                                 + 0.990794 w_k1_FDX_s110
                                 + 1.15119 w_k1_IBM_s110
                                 + 0.865624 w_k1_INTC_s110
                                 + 0.88567 w_k1_KO_s110 + 1.04378 w_k1_MAT_s110
                                 + 1.01226 w_k1_MSFT_s110
                                 + 0.955153 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.928397 w_k1_ADM_s110
                                 + 0.840184 w_k1_AXP_s110
                                 + 0.853412 w_k1_DIS_s110
                                 + 0.834469 w_k1_FDX_s110
                                 + 1.34153 w_k1_IBM_s110
                                 + 0.846586 w_k1_INTC_s110
                                 + 1.03572 w_k1_KO_s110 + 1.11008 w_k1_MAT_s110
                                 + 1.00577 w_k1_MSFT_s110
                                 + 1.03206 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.28887 w_k1_ADM_s110
                                 + 1.3602 w_k1_AXP_s110 + 1.21066 w_k1_DIS_s110
                                 + 1.14926 w_k1_FDX_s110
                                 + 1.47098 w_k1_IBM_s110
                                 + 1.04289 w_k1_INTC_s110
                                 + 1.11933 w_k1_KO_s110 + 1.38409 w_k1_MAT_s110
                                 + 1.09631 w_k1_MSFT_s110
                                 + 1.07711 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.949345 w_k1_ADM_s110
                                 + 0.591814 w_k1_AXP_s110
                                 + 0.738033 w_k1_DIS_s110
                                 + 0.874304 w_k1_FDX_s110
                                 + 0.679696 w_k1_IBM_s110
                                 + 1.00587 w_k1_INTC_s110
                                 + 0.657124 w_k1_KO_s110
                                 + 0.298066 w_k1_MAT_s110
                                 + 0.996165 w_k1_MSFT_s110
                                 + 0.697964 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.05487 w_k1_ADM_s110
                                 + 0.860326 w_k1_AXP_s110
                                 + 0.790087 w_k1_DIS_s110
                                 + 0.99086 w_k1_FDX_s110
                                 + 0.600746 w_k1_IBM_s110
                                 + 0.780393 w_k1_INTC_s110
                                 + 0.574651 w_k1_KO_s110
                                 + 0.47055 w_k1_MAT_s110
                                 + 0.815259 w_k1_MSFT_s110
                                 + 0.800261 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.08784 w_k1_ADM_s111
                                 + 0.652784 w_k1_AXP_s111
                                 + 0.711451 w_k1_DIS_s111
                                 + 0.815591 w_k1_FDX_s111
                                 + 0.810562 w_k1_IBM_s111
                                 + 0.694934 w_k1_INTC_s111
                                 + 0.848017 w_k1_KO_s111
                                 + 0.788741 w_k1_MAT_s111
                                 + 0.865855 w_k1_MSFT_s111
                                 + 0.837563 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.881729 w_k1_ADM_s111
                                 + 0.673001 w_k1_AXP_s111
                                 + 0.698929 w_k1_DIS_s111
                                 + 0.919877 w_k1_FDX_s111
                                 + 0.621892 w_k1_IBM_s111
                                 + 0.881238 w_k1_INTC_s111
                                 + 0.685615 w_k1_KO_s111
                                 + 0.490263 w_k1_MAT_s111
                                 + 0.885965 w_k1_MSFT_s111
                                 + 0.812232 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.4168 w_k1_ADM_s111
                                 + 1.42008 w_k1_AXP_s111
                                 + 1.68386 w_k1_DIS_s111
                                 + 1.35635 w_k1_FDX_s111
                                 + 1.56668 w_k1_IBM_s111
                                 + 0.892823 w_k1_INTC_s111
                                 + 1.5139 w_k1_KO_s111 + 1.57399 w_k1_MAT_s111
                                 + 1.12763 w_k1_MSFT_s111
                                 + 1.20294 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.06885 w_k1_ADM_s111
                                 + 1.02083 w_k1_AXP_s111
                                 + 0.823936 w_k1_DIS_s111
                                 + 0.940789 w_k1_FDX_s111
                                 + 1.12945 w_k1_IBM_s111
                                 + 0.945053 w_k1_INTC_s111
                                 + 0.968284 w_k1_KO_s111
                                 + 1.67578 w_k1_MAT_s111
                                 + 0.887657 w_k1_MSFT_s111
                                 + 1.08791 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.04563 w_k1_ADM_s111
                                 + 0.940315 w_k1_AXP_s111
                                 + 0.936003 w_k1_DIS_s111
                                 + 1.07088 w_k1_FDX_s111
                                 + 0.862292 w_k1_IBM_s111
                                 + 0.936792 w_k1_INTC_s111
                                 + 0.984738 w_k1_KO_s111
                                 + 0.902375 w_k1_MAT_s111
                                 + 1.15432 w_k1_MSFT_s111
                                 + 1.04916 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.38217 w_k1_ADM_s111
                                 + 0.776372 w_k1_AXP_s111
                                 + 0.888268 w_k1_DIS_s111
                                 + 0.889373 w_k1_FDX_s111
                                 + 1.2523 w_k1_IBM_s111
                                 + 0.711713 w_k1_INTC_s111
                                 + 1.00609 w_k1_KO_s111
                                 + 0.893289 w_k1_MAT_s111
                                 + 0.835609 w_k1_MSFT_s111
                                 + 1.01262 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.60611 w_k1_ADM_s111
                                 + 1.32479 w_k1_AXP_s111 + 1.4765 w_k1_DIS_s111
                                 + 1.3263 w_k1_FDX_s111
                                 + 0.927908 w_k1_IBM_s111
                                 + 1.23283 w_k1_INTC_s111
                                 + 1.04714 w_k1_KO_s111
                                 + 0.975036 w_k1_MAT_s111
                                 + 1.14408 w_k1_MSFT_s111
                                 + 1.17918 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.1522 w_k1_ADM_s111
                                 + 1.19415 w_k1_AXP_s111
                                 + 1.09522 w_k1_DIS_s111
                                 + 1.11053 w_k1_FDX_s111 + 1.4847 w_k1_IBM_s111
                                 + 1.13091 w_k1_INTC_s111
                                 + 1.21593 w_k1_KO_s111 + 1.5702 w_k1_MAT_s111
                                 + 0.998999 w_k1_MSFT_s111
                                 + 1.20362 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.982455 w_k1_ADM_s112
                                 + 1.07386 w_k1_AXP_s112
                                 + 1.06926 w_k1_DIS_s112
                                 + 1.04886 w_k1_FDX_s112
                                 + 1.15512 w_k1_IBM_s112
                                 + 0.994708 w_k1_INTC_s112
                                 + 1.08203 w_k1_KO_s112
                                 + 0.980656 w_k1_MAT_s112
                                 + 0.970993 w_k1_MSFT_s112
                                 + 1.02955 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.08248 w_k1_ADM_s112
                                 + 1.33821 w_k1_AXP_s112
                                 + 1.34367 w_k1_DIS_s112 + 1.2805 w_k1_FDX_s112
                                 + 0.97354 w_k1_IBM_s112
                                 + 0.991192 w_k1_INTC_s112
                                 + 0.967758 w_k1_KO_s112
                                 + 1.17579 w_k1_MAT_s112
                                 + 1.15781 w_k1_MSFT_s112
                                 + 1.24941 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.25752 w_k1_ADM_s112
                                 + 1.56924 w_k1_AXP_s112 + 1.4773 w_k1_DIS_s112
                                 + 1.24025 w_k1_FDX_s112
                                 + 1.38565 w_k1_IBM_s112
                                 + 0.914707 w_k1_INTC_s112
                                 + 1.22883 w_k1_KO_s112 + 1.72658 w_k1_MAT_s112
                                 + 1.08626 w_k1_MSFT_s112
                                 + 1.34961 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.860807 w_k1_ADM_s112
                                 + 0.898989 w_k1_AXP_s112
                                 + 0.740969 w_k1_DIS_s112
                                 + 0.844035 w_k1_FDX_s112
                                 + 0.937024 w_k1_IBM_s112
                                 + 0.793449 w_k1_INTC_s112
                                 + 0.765323 w_k1_KO_s112
                                 + 0.946972 w_k1_MAT_s112
                                 + 0.777044 w_k1_MSFT_s112
                                 + 0.975994 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.03116 w_k1_ADM_s112
                                 + 0.870719 w_k1_AXP_s112
                                 + 1.08005 w_k1_DIS_s112
                                 + 0.945035 w_k1_FDX_s112
                                 + 0.763023 w_k1_IBM_s112
                                 + 0.777047 w_k1_INTC_s112
                                 + 0.847709 w_k1_KO_s112
                                 + 0.762061 w_k1_MAT_s112
                                 + 0.812243 w_k1_MSFT_s112
                                 + 0.926288 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.03578 w_k1_ADM_s112
                                 + 0.924249 w_k1_AXP_s112
                                 + 0.70032 w_k1_DIS_s112
                                 + 0.983822 w_k1_FDX_s112
                                 + 0.76315 w_k1_IBM_s112
                                 + 0.988184 w_k1_INTC_s112
                                 + 0.767814 w_k1_KO_s112
                                 + 0.85675 w_k1_MAT_s112
                                 + 0.879108 w_k1_MSFT_s112
                                 + 0.928369 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.932491 w_k1_ADM_s112
                                 + 0.855672 w_k1_AXP_s112
                                 + 0.895317 w_k1_DIS_s112
                                 + 0.947218 w_k1_FDX_s112
                                 + 1.32042 w_k1_IBM_s112
                                 + 1.10575 w_k1_INTC_s112 + 1.0145 w_k1_KO_s112
                                 + 1.3245 w_k1_MAT_s112
                                 + 0.933959 w_k1_MSFT_s112
                                 + 1.01371 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.11868 w_k1_ADM_s112
                                 + 1.20302 w_k1_AXP_s112
                                 + 1.40412 w_k1_DIS_s112
                                 + 1.12255 w_k1_FDX_s112
                                 + 1.60862 w_k1_IBM_s112
                                 + 1.09949 w_k1_INTC_s112
                                 + 1.49127 w_k1_KO_s112 + 1.338 w_k1_MAT_s112
                                 + 1.033 w_k1_MSFT_s112 + 1.25772 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.38309 w_k1_ADM_s113
                                 + 1.09983 w_k1_AXP_s113
                                 + 1.38325 w_k1_DIS_s113
                                 + 1.24968 w_k1_FDX_s113
                                 + 0.974749 w_k1_IBM_s113
                                 + 1.00506 w_k1_INTC_s113
                                 + 1.09523 w_k1_KO_s113
                                 + 0.993927 w_k1_MAT_s113
                                 + 1.08028 w_k1_MSFT_s113
                                 + 1.15882 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.861649 w_k1_ADM_s113
                                 + 0.611614 w_k1_AXP_s113
                                 + 0.850873 w_k1_DIS_s113
                                 + 0.862468 w_k1_FDX_s113
                                 + 0.784993 w_k1_IBM_s113
                                 + 0.852278 w_k1_INTC_s113
                                 + 0.851826 w_k1_KO_s113
                                 + 0.646224 w_k1_MAT_s113
                                 + 0.913709 w_k1_MSFT_s113
                                 + 0.83817 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.738003 w_k1_ADM_s113
                                 + 0.701206 w_k1_AXP_s113
                                 + 0.635605 w_k1_DIS_s113
                                 + 0.771062 w_k1_FDX_s113
                                 + 0.549554 w_k1_IBM_s113
                                 + 0.807332 w_k1_INTC_s113
                                 + 0.682374 w_k1_KO_s113
                                 + 0.737091 w_k1_MAT_s113
                                 + 0.832007 w_k1_MSFT_s113
                                 + 0.78514 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.57917 w_k1_ADM_s113
                                 + 1.65528 w_k1_AXP_s113
                                 + 2.15344 w_k1_DIS_s113
                                 + 1.69641 w_k1_FDX_s113
                                 + 1.23692 w_k1_IBM_s113
                                 + 1.39229 w_k1_INTC_s113
                                 + 1.50392 w_k1_KO_s113 + 1.18517 w_k1_MAT_s113
                                 + 1.34878 w_k1_MSFT_s113
                                 + 1.57163 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.02397 w_k1_ADM_s113
                                 + 1.03541 w_k1_AXP_s113
                                 + 0.842411 w_k1_DIS_s113
                                 + 0.832009 w_k1_FDX_s113
                                 + 1.09545 w_k1_IBM_s113
                                 + 0.848058 w_k1_INTC_s113
                                 + 1.11781 w_k1_KO_s113 + 1.25157 w_k1_MAT_s113
                                 + 0.779459 w_k1_MSFT_s113
                                 + 1.06484 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.18505 w_k1_ADM_s113
                                 + 1.05349 w_k1_AXP_s113
                                 + 0.859789 w_k1_DIS_s113
                                 + 1.08082 w_k1_FDX_s113
                                 + 0.757434 w_k1_IBM_s113
                                 + 0.887921 w_k1_INTC_s113
                                 + 0.807749 w_k1_KO_s113
                                 + 0.947984 w_k1_MAT_s113
                                 + 0.918813 w_k1_MSFT_s113
                                 + 1.0457 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.850614 w_k1_ADM_s113
                                 + 1.08489 w_k1_AXP_s113
                                 + 1.27658 w_k1_DIS_s113
                                 + 1.02977 w_k1_FDX_s113
                                 + 1.27148 w_k1_IBM_s113
                                 + 1.03702 w_k1_INTC_s113
                                 + 1.25802 w_k1_KO_s113 + 1.663 w_k1_MAT_s113
                                 + 1.10251 w_k1_MSFT_s113
                                 + 1.16643 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.607549 w_k1_ADM_s113
                                 + 0.634935 w_k1_AXP_s113
                                 + 0.662488 w_k1_DIS_s113
                                 + 0.720042 w_k1_FDX_s113
                                 + 0.827708 w_k1_IBM_s113
                                 + 0.788791 w_k1_INTC_s113
                                 + 0.757756 w_k1_KO_s113
                                 + 0.705108 w_k1_MAT_s113
                                 + 0.708511 w_k1_MSFT_s113
                                 + 0.854419 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.37098 w_k1_ADM_s114
                                 + 1.52775 w_k1_AXP_s114
                                 + 1.58586 w_k1_DIS_s114
                                 + 1.20144 w_k1_FDX_s114
                                 + 1.49453 w_k1_IBM_s114
                                 + 1.27117 w_k1_INTC_s114
                                 + 1.43628 w_k1_KO_s114 + 2.05206 w_k1_MAT_s114
                                 + 1.39778 w_k1_MSFT_s114
                                 + 1.21955 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.859451 w_k1_ADM_s114
                                 + 0.910762 w_k1_AXP_s114
                                 + 0.785023 w_k1_DIS_s114
                                 + 0.847731 w_k1_FDX_s114
                                 + 1.11001 w_k1_IBM_s114
                                 + 1.05748 w_k1_INTC_s114
                                 + 0.962715 w_k1_KO_s114
                                 + 1.07325 w_k1_MAT_s114
                                 + 0.954005 w_k1_MSFT_s114
                                 + 0.886992 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.744654 w_k1_ADM_s114
                                 + 0.847523 w_k1_AXP_s114
                                 + 0.992041 w_k1_DIS_s114
                                 + 0.80935 w_k1_FDX_s114
                                 + 1.45264 w_k1_IBM_s114
                                 + 1.01148 w_k1_INTC_s114
                                 + 1.13486 w_k1_KO_s114 + 1.26937 w_k1_MAT_s114
                                 + 1.0739 w_k1_MSFT_s114
                                 + 0.820891 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.15604 w_k1_ADM_s114
                                 + 1.42403 w_k1_AXP_s114 + 1.3857 w_k1_DIS_s114
                                 + 1.33952 w_k1_FDX_s114
                                 + 0.852097 w_k1_IBM_s114
                                 + 1.20104 w_k1_INTC_s114 + 1.1445 w_k1_KO_s114
                                 + 1.05367 w_k1_MAT_s114
                                 + 1.39506 w_k1_MSFT_s114
                                 + 0.961701 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.21222 w_k1_ADM_s114
                                 + 1.02152 w_k1_AXP_s114
                                 + 0.842535 w_k1_DIS_s114
                                 + 0.951886 w_k1_FDX_s114
                                 + 0.994703 w_k1_IBM_s114
                                 + 1.05564 w_k1_INTC_s114
                                 + 0.96817 w_k1_KO_s114 + 1.01023 w_k1_MAT_s114
                                 + 1.06987 w_k1_MSFT_s114 + 1.0567 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.965078 w_k1_ADM_s114
                                 + 0.80478 w_k1_AXP_s114
                                 + 0.921938 w_k1_DIS_s114
                                 + 0.973969 w_k1_FDX_s114
                                 + 0.695715 w_k1_IBM_s114
                                 + 1.0382 w_k1_INTC_s114
                                 + 0.860923 w_k1_KO_s114
                                 + 0.801657 w_k1_MAT_s114
                                 + 1.15355 w_k1_MSFT_s114
                                 + 0.937582 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.835611 w_k1_ADM_s114
                                 + 0.901444 w_k1_AXP_s114
                                 + 1.09302 w_k1_DIS_s114
                                 + 1.01136 w_k1_FDX_s114
                                 + 0.829754 w_k1_IBM_s114
                                 + 1.10903 w_k1_INTC_s114
                                 + 0.877079 w_k1_KO_s114
                                 + 0.777129 w_k1_MAT_s114
                                 + 1.23751 w_k1_MSFT_s114
                                 + 0.971952 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.03064 w_k1_ADM_s114
                                 + 1.31563 w_k1_AXP_s114
                                 + 0.996321 w_k1_DIS_s114
                                 + 1.15261 w_k1_FDX_s114
                                 + 1.10781 w_k1_IBM_s114
                                 + 1.42682 w_k1_INTC_s114
                                 + 1.14709 w_k1_KO_s114 + 1.40501 w_k1_MAT_s114
                                 + 1.26777 w_k1_MSFT_s114
                                 + 1.00647 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.0286 w_k1_ADM_s115
                                 + 1.169 w_k1_AXP_s115 + 0.729909 w_k1_DIS_s115
                                 + 0.983948 w_k1_FDX_s115
                                 + 0.813234 w_k1_IBM_s115
                                 + 0.847123 w_k1_INTC_s115
                                 + 0.832362 w_k1_KO_s115
                                 + 1.38773 w_k1_MAT_s115
                                 + 0.982479 w_k1_MSFT_s115
                                 + 1.12919 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.66287 w_k1_ADM_s115
                                 + 0.67693 w_k1_AXP_s115
                                 + 0.866135 w_k1_DIS_s115
                                 + 0.944865 w_k1_FDX_s115
                                 + 0.574178 w_k1_IBM_s115
                                 + 0.870552 w_k1_INTC_s115
                                 + 0.688332 w_k1_KO_s115
                                 + 0.876024 w_k1_MAT_s115
                                 + 0.970697 w_k1_MSFT_s115
                                 + 0.935449 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.707481 w_k1_ADM_s115
                                 + 0.676721 w_k1_AXP_s115
                                 + 0.624457 w_k1_DIS_s115
                                 + 0.782119 w_k1_FDX_s115
                                 + 0.799109 w_k1_IBM_s115
                                 + 0.825461 w_k1_INTC_s115
                                 + 0.751769 w_k1_KO_s115
                                 + 0.991525 w_k1_MAT_s115
                                 + 0.906139 w_k1_MSFT_s115
                                 + 0.954769 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.768141 w_k1_ADM_s115
                                 + 0.818968 w_k1_AXP_s115
                                 + 0.611119 w_k1_DIS_s115
                                 + 0.749759 w_k1_FDX_s115
                                 + 1.28455 w_k1_IBM_s115
                                 + 0.910313 w_k1_INTC_s115
                                 + 1.04005 w_k1_KO_s115 + 1.68814 w_k1_MAT_s115
                                 + 0.819869 w_k1_MSFT_s115
                                 + 0.967821 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.26085 w_k1_ADM_s115
                                 + 1.51507 w_k1_AXP_s115 + 1.5631 w_k1_DIS_s115
                                 + 1.3066 w_k1_FDX_s115 + 1.4724 w_k1_IBM_s115
                                 + 1.21195 w_k1_INTC_s115
                                 + 1.25885 w_k1_KO_s115 + 1.75594 w_k1_MAT_s115
                                 + 1.25985 w_k1_MSFT_s115
                                 + 1.28637 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.10057 w_k1_ADM_s115
                                 + 1.28826 w_k1_AXP_s115
                                 + 1.20501 w_k1_DIS_s115
                                 + 0.960661 w_k1_FDX_s115
                                 + 1.35454 w_k1_IBM_s115
                                 + 0.963449 w_k1_INTC_s115
                                 + 1.25402 w_k1_KO_s115 + 1.97946 w_k1_MAT_s115
                                 + 1.03635 w_k1_MSFT_s115
                                 + 1.13592 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.823545 w_k1_ADM_s115
                                 + 1.15171 w_k1_AXP_s115
                                 + 1.20206 w_k1_DIS_s115
                                 + 1.02498 w_k1_FDX_s115
                                 + 1.08088 w_k1_IBM_s115
                                 + 1.18388 w_k1_INTC_s115 + 1.0831 w_k1_KO_s115
                                 + 1.31318 w_k1_MAT_s115
                                 + 1.10627 w_k1_MSFT_s115
                                 + 1.05343 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.695454 w_k1_ADM_s115
                                 + 1.20094 w_k1_AXP_s115
                                 + 1.20125 w_k1_DIS_s115
                                 + 1.21992 w_k1_FDX_s115
                                 + 0.871169 w_k1_IBM_s115
                                 + 1.20871 w_k1_INTC_s115
                                 + 0.886821 w_k1_KO_s115
                                 + 1.00906 w_k1_MAT_s115
                                 + 1.06695 w_k1_MSFT_s115
                                 + 1.18962 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.92193 w_k1_ADM_s116
                                 + 0.891241 w_k1_AXP_s116
                                 + 0.856858 w_k1_DIS_s116
                                 + 0.868408 w_k1_FDX_s116
                                 + 1.27322 w_k1_IBM_s116
                                 + 0.874544 w_k1_INTC_s116
                                 + 0.986185 w_k1_KO_s116 + 1.1504 w_k1_MAT_s116
                                 + 0.8266 w_k1_MSFT_s116 + 1.00515 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.795617 w_k1_ADM_s116
                                 + 0.889632 w_k1_AXP_s116
                                 + 0.788303 w_k1_DIS_s116
                                 + 1.00519 w_k1_FDX_s116
                                 + 0.693908 w_k1_IBM_s116
                                 + 1.08602 w_k1_INTC_s116
                                 + 0.754739 w_k1_KO_s116
                                 + 0.632557 w_k1_MAT_s116
                                 + 0.892447 w_k1_MSFT_s116
                                 + 0.945508 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.13366 w_k1_ADM_s116
                                 + 1.42769 w_k1_AXP_s116
                                 + 1.24056 w_k1_DIS_s116
                                 + 1.17642 w_k1_FDX_s116
                                 + 1.30366 w_k1_IBM_s116
                                 + 1.03537 w_k1_INTC_s116
                                 + 1.20812 w_k1_KO_s116 + 1.78415 w_k1_MAT_s116
                                 + 1.09776 w_k1_MSFT_s116
                                 + 1.25149 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.749153 w_k1_ADM_s116
                                 + 0.589531 w_k1_AXP_s116
                                 + 0.550354 w_k1_DIS_s116
                                 + 0.732012 w_k1_FDX_s116
                                 + 0.670443 w_k1_IBM_s116
                                 + 0.740709 w_k1_INTC_s116
                                 + 0.731325 w_k1_KO_s116
                                 + 0.591512 w_k1_MAT_s116
                                 + 0.746108 w_k1_MSFT_s116
                                 + 0.762722 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.750945 w_k1_ADM_s116
                                 + 1.09538 w_k1_AXP_s116
                                 + 0.870096 w_k1_DIS_s116
                                 + 0.894527 w_k1_FDX_s116
                                 + 0.897271 w_k1_IBM_s116
                                 + 1.00566 w_k1_INTC_s116
                                 + 0.74592 w_k1_KO_s116
                                 + 0.701545 w_k1_MAT_s116
                                 + 0.94381 w_k1_MSFT_s116
                                 + 0.90649 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.876381 w_k1_ADM_s116
                                 + 0.877664 w_k1_AXP_s116
                                 + 1.02112 w_k1_DIS_s116
                                 + 1.07317 w_k1_FDX_s116
                                 + 0.973719 w_k1_IBM_s116
                                 + 0.96444 w_k1_INTC_s116
                                 + 0.891469 w_k1_KO_s116
                                 + 1.14909 w_k1_MAT_s116
                                 + 1.07076 w_k1_MSFT_s116
                                 + 0.983516 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.36503 w_k1_ADM_s116
                                 + 1.24068 w_k1_AXP_s116
                                 + 1.38578 w_k1_DIS_s116
                                 + 1.18865 w_k1_FDX_s116
                                 + 1.88337 w_k1_IBM_s116
                                 + 1.01377 w_k1_INTC_s116
                                 + 1.52021 w_k1_KO_s116 + 2.1832 w_k1_MAT_s116
                                 + 1.19724 w_k1_MSFT_s116
                                 + 1.14184 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.803934 w_k1_ADM_s116
                                 + 0.738164 w_k1_AXP_s116
                                 + 0.765775 w_k1_DIS_s116
                                 + 0.860977 w_k1_FDX_s116
                                 + 1.07835 w_k1_IBM_s116
                                 + 0.654424 w_k1_INTC_s116
                                 + 0.79639 w_k1_KO_s116
                                 + 0.949599 w_k1_MAT_s116
                                 + 0.734643 w_k1_MSFT_s116
                                 + 0.873698 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.784016 w_k1_ADM_s117
                                 + 0.708062 w_k1_AXP_s117
                                 + 0.510764 w_k1_DIS_s117
                                 + 0.731671 w_k1_FDX_s117
                                 + 0.701892 w_k1_IBM_s117
                                 + 0.670264 w_k1_INTC_s117
                                 + 0.669359 w_k1_KO_s117
                                 + 0.771059 w_k1_MAT_s117
                                 + 0.757775 w_k1_MSFT_s117
                                 + 0.749691 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.941905 w_k1_ADM_s117
                                 + 1.09775 w_k1_AXP_s117
                                 + 0.824927 w_k1_DIS_s117
                                 + 1.01867 w_k1_FDX_s117
                                 + 0.941676 w_k1_IBM_s117
                                 + 0.867229 w_k1_INTC_s117
                                 + 0.824286 w_k1_KO_s117
                                 + 0.924147 w_k1_MAT_s117
                                 + 0.868792 w_k1_MSFT_s117
                                 + 0.897984 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.05838 w_k1_ADM_s117
                                 + 0.864011 w_k1_AXP_s117
                                 + 1.04061 w_k1_DIS_s117
                                 + 0.960657 w_k1_FDX_s117
                                 + 0.616323 w_k1_IBM_s117
                                 + 0.86405 w_k1_INTC_s117
                                 + 0.734931 w_k1_KO_s117
                                 + 0.494646 w_k1_MAT_s117
                                 + 0.889838 w_k1_MSFT_s117
                                 + 0.95744 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.38527 w_k1_ADM_s117
                                 + 0.99666 w_k1_AXP_s117
                                 + 1.47964 w_k1_DIS_s117
                                 + 1.06703 w_k1_FDX_s117
                                 + 1.11679 w_k1_IBM_s117
                                 + 1.06838 w_k1_INTC_s117
                                 + 1.16547 w_k1_KO_s117
                                 + 0.698226 w_k1_MAT_s117
                                 + 1.08301 w_k1_MSFT_s117
                                 + 1.01585 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.17351 w_k1_ADM_s117
                                 + 1.10298 w_k1_AXP_s117
                                 + 1.10901 w_k1_DIS_s117
                                 + 1.19347 w_k1_FDX_s117
                                 + 0.98558 w_k1_IBM_s117
                                 + 1.0641 w_k1_INTC_s117 + 1.07909 w_k1_KO_s117
                                 + 1.18427 w_k1_MAT_s117
                                 + 0.975486 w_k1_MSFT_s117
                                 + 0.888525 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.66196 w_k1_ADM_s117
                                 + 1.34642 w_k1_AXP_s117
                                 + 1.59799 w_k1_DIS_s117 + 1.1405 w_k1_FDX_s117
                                 + 1.82836 w_k1_IBM_s117
                                 + 0.888106 w_k1_INTC_s117
                                 + 1.44129 w_k1_KO_s117 + 2.30988 w_k1_MAT_s117
                                 + 1.03588 w_k1_MSFT_s117
                                 + 1.26071 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.07931 w_k1_ADM_s117
                                 + 1.22795 w_k1_AXP_s117
                                 + 1.48441 w_k1_DIS_s117
                                 + 1.24214 w_k1_FDX_s117
                                 + 1.39513 w_k1_IBM_s117
                                 + 1.05467 w_k1_INTC_s117
                                 + 1.35661 w_k1_KO_s117 + 1.4771 w_k1_MAT_s117
                                 + 1.1868 w_k1_MSFT_s117 + 1.10903 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.88636 w_k1_ADM_s117
                                 + 0.675388 w_k1_AXP_s117
                                 + 0.818539 w_k1_DIS_s117
                                 + 0.817271 w_k1_FDX_s117
                                 + 0.862179 w_k1_IBM_s117
                                 + 0.710555 w_k1_INTC_s117
                                 + 0.899813 w_k1_KO_s117
                                 + 0.854525 w_k1_MAT_s117
                                 + 0.82903 w_k1_MSFT_s117
                                 + 0.731013 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0774991799158885 w_k1_ADM_s000
                                 + 0.0306750667431898 w_k1_ADM_s001
                                 + 0.0269780733173573 w_k1_ADM_s002
                                 + 0.0275276290864041 w_k1_ADM_s003
                                 + 0.111195899200796 w_k1_ADM_s004
                                 + 0.0321015703050532 w_k1_ADM_s005
                                 + 0.0273283014808121 w_k1_ADM_s006
                                 + 0.0239602524746727 w_k1_ADM_s007
                                 + 0.0443940081274655 w_k1_ADM_s010
                                 + 0.0264284964209756 w_k1_ADM_s011
                                 + 0.0297672716700504 w_k1_ADM_s012
                                 + 0.0145976934462004 w_k1_ADM_s013
                                 + 0.0514474700558517 w_k1_ADM_s014
                                 + 0.00971542896817131 w_k1_ADM_s015
                                 + 0.0253057364574822 w_k1_ADM_s016
                                 + 0.0197867037387675 w_k1_ADM_s017
                                 + 0.0850524825952699 w_k1_ADM_s100
                                 + 0.0507002056128243 w_k1_ADM_s101
                                 + 0.0342764347826263 w_k1_ADM_s102
                                 + 0.0288837355448469 w_k1_ADM_s103
                                 + 0.0210910284768955 w_k1_ADM_s104
                                 + 0.0164339027417913 w_k1_ADM_s105
                                 + 0.082996483599694 w_k1_ADM_s106
                                 + 0.0451191499506544 w_k1_ADM_s107
                                 + 0.0150731015439461 w_k1_ADM_s110
                                 + 0.0136043022111861 w_k1_ADM_s111
                                 + 0.0128728278569363 w_k1_ADM_s112
                                 + 0.0123770917537593 w_k1_ADM_s113
                                 + 0.0137287022721752 w_k1_ADM_s114
                                 + 0.00420197423616503 w_k1_ADM_s115
                                 + 0.00900337874617434 w_k1_ADM_s116
                                 + 0.00538986192307365 w_k1_ADM_s117
                                 + 0.0778678935467123 w_k1_AXP_s000
                                 + 0.0308670323132397 w_k1_AXP_s001
                                 + 0.0284438636467838 w_k1_AXP_s002
                                 + 0.0269784583675808 w_k1_AXP_s003
                                 + 0.105492502803156 w_k1_AXP_s004
                                 + 0.030372574263715 w_k1_AXP_s005
                                 + 0.0266926202131275 w_k1_AXP_s006
                                 + 0.0256345007266598 w_k1_AXP_s007
                                 + 0.0423815921184073 w_k1_AXP_s010
                                 + 0.0251354284212403 w_k1_AXP_s011
                                 + 0.0255601917476748 w_k1_AXP_s012
                                 + 0.01626278451859 w_k1_AXP_s013
                                 + 0.045053160216738 w_k1_AXP_s014
                                 + 0.0127848532552003 w_k1_AXP_s015
                                 + 0.0259574726697883 w_k1_AXP_s016
                                 + 0.0220803420845068 w_k1_AXP_s017
                                 + 0.0890796046633387 w_k1_AXP_s100
                                 + 0.0525560352580553 w_k1_AXP_s101
                                 + 0.0354544773877238 w_k1_AXP_s102
                                 + 0.0313664631836199 w_k1_AXP_s103
                                 + 0.0186324903272505 w_k1_AXP_s104
                                 + 0.0172007704757789 w_k1_AXP_s105
                                 + 0.0820857303305771 w_k1_AXP_s106
                                 + 0.0437797075007277 w_k1_AXP_s107
                                 + 0.0148004170575752 w_k1_AXP_s110
                                 + 0.0108551638774153 w_k1_AXP_s111
                                 + 0.0132290257867146 w_k1_AXP_s112
                                 + 0.0120725022471173 w_k1_AXP_s113
                                 + 0.0143282877073253 w_k1_AXP_s114
                                 + 0.00522660337742814 w_k1_AXP_s115
                                 + 0.00979183992867941 w_k1_AXP_s116
                                 + 0.00480955567539096 w_k1_AXP_s117
                                 + 0.0828436566461891 w_k1_DIS_s000
                                 + 0.0320130555863198 w_k1_DIS_s001
                                 + 0.0296959300746879 w_k1_DIS_s002
                                 + 0.024859306133886 w_k1_DIS_s003
                                 + 0.112501348180839 w_k1_DIS_s004
                                 + 0.0307240672167826 w_k1_DIS_s005
                                 + 0.0283833027780868 w_k1_DIS_s006
                                 + 0.0238838692701292 w_k1_DIS_s007
                                 + 0.0445074937260636 w_k1_DIS_s010
                                 + 0.0262788557799328 w_k1_DIS_s011
                                 + 0.0275018218881832 w_k1_DIS_s012
                                 + 0.0154506070221857 w_k1_DIS_s013
                                 + 0.0455451932081858 w_k1_DIS_s014
                                 + 0.0110850843500285 w_k1_DIS_s015
                                 + 0.0270466780634436 w_k1_DIS_s016
                                 + 0.021637701118231 w_k1_DIS_s017
                                 + 0.0933425245553652 w_k1_DIS_s100
                                 + 0.0536121097921064 w_k1_DIS_s101
                                 + 0.0342323981307744 w_k1_DIS_s102
                                 + 0.0319968363913267 w_k1_DIS_s103
                                 + 0.0194334279996928 w_k1_DIS_s104
                                 + 0.0179255964904805 w_k1_DIS_s105
                                 + 0.0818408722396169 w_k1_DIS_s106
                                 + 0.0408379811167646 w_k1_DIS_s107
                                 + 0.0143911799903546 w_k1_DIS_s110
                                 + 0.0113229578490574 w_k1_DIS_s111
                                 + 0.0131378448256446 w_k1_DIS_s112
                                 + 0.0128218877519887 w_k1_DIS_s113
                                 + 0.0140201525234271 w_k1_DIS_s114
                                 + 0.00499686154812365 w_k1_DIS_s115
                                 + 0.00934067590911476 w_k1_DIS_s116
                                 + 0.00546068267269082 w_k1_DIS_s117
                                 + 0.081456636474341 w_k1_FDX_s000
                                 + 0.0309996897503535 w_k1_FDX_s001
                                 + 0.0286199517156077 w_k1_FDX_s002
                                 + 0.0248413180954592 w_k1_FDX_s003
                                 + 0.113076822860247 w_k1_FDX_s004
                                 + 0.0291504052052603 w_k1_FDX_s005
                                 + 0.0255417545229196 w_k1_FDX_s006
                                 + 0.0237684266067565 w_k1_FDX_s007
                                 + 0.0450082684982718 w_k1_FDX_s010
                                 + 0.0267398780466477 w_k1_FDX_s011
                                 + 0.0270499467019534 w_k1_FDX_s012
                                 + 0.0144617614085368 w_k1_FDX_s013
                                 + 0.0452141907209927 w_k1_FDX_s014
                                 + 0.0111197215872941 w_k1_FDX_s015
                                 + 0.0252740984468717 w_k1_FDX_s016
                                 + 0.0222991767341046 w_k1_FDX_s017
                                 + 0.0827093953370208 w_k1_FDX_s100
                                 + 0.0492568900071404 w_k1_FDX_s101
                                 + 0.0314154563747833 w_k1_FDX_s102
                                 + 0.0313518142158654 w_k1_FDX_s103
                                 + 0.0197143684247668 w_k1_FDX_s104
                                 + 0.0168072118094728 w_k1_FDX_s105
                                 + 0.083468452741154 w_k1_FDX_s106
                                 + 0.0428758819653346 w_k1_FDX_s107
                                 + 0.0147887298235995 w_k1_FDX_s110
                                 + 0.0117048571654334 w_k1_FDX_s111
                                 + 0.0130728821894473 w_k1_FDX_s112
                                 + 0.0122987346065815 w_k1_FDX_s113
                                 + 0.0138022360768533 w_k1_FDX_s114
                                 + 0.00493020842548454 w_k1_FDX_s115
                                 + 0.00968311583375781 w_k1_FDX_s116
                                 + 0.00504048975965591 w_k1_FDX_s117
                                 + 0.0829529218252113 w_k1_IBM_s000
                                 + 0.0314532747306951 w_k1_IBM_s001
                                 + 0.0292613841968439 w_k1_IBM_s002
                                 + 0.0264261807491535 w_k1_IBM_s003
                                 + 0.109433875792577 w_k1_IBM_s004
                                 + 0.0301663763952121 w_k1_IBM_s005
                                 + 0.0252324608801081 w_k1_IBM_s006
                                 + 0.0252062960319642 w_k1_IBM_s007
                                 + 0.0480795724813428 w_k1_IBM_s010
                                 + 0.0256790875914051 w_k1_IBM_s011
                                 + 0.0284916369611211 w_k1_IBM_s012
                                 + 0.0154324805700346 w_k1_IBM_s013
                                 + 0.0474976452410763 w_k1_IBM_s014
                                 + 0.0110320361636014 w_k1_IBM_s015
                                 + 0.0246494447192475 w_k1_IBM_s016
                                 + 0.022963853873816 w_k1_IBM_s017
                                 + 0.0836317898257317 w_k1_IBM_s100
                                 + 0.0480419966346852 w_k1_IBM_s101
                                 + 0.033590329285746 w_k1_IBM_s102
                                 + 0.0320765389634951 w_k1_IBM_s103
                                 + 0.0195714063024612 w_k1_IBM_s104
                                 + 0.0174490233914107 w_k1_IBM_s105
                                 + 0.0793507512390718 w_k1_IBM_s106
                                 + 0.0424673933574337 w_k1_IBM_s107
                                 + 0.0144769984150988 w_k1_IBM_s110
                                 + 0.0116567769171275 w_k1_IBM_s111
                                 + 0.0129858772863572 w_k1_IBM_s112
                                 + 0.0117260317081955 w_k1_IBM_s113
                                 + 0.0134768284723709 w_k1_IBM_s114
                                 + 0.00494827335164605 w_k1_IBM_s115
                                 + 0.0106081964672865 w_k1_IBM_s116
                                 + 0.00514838854498981 w_k1_IBM_s117
                                 + 0.0932331755307146 w_k1_INTC_s000
                                 + 0.0289667479523229 w_k1_INTC_s001
                                 + 0.025374600465032 w_k1_INTC_s002
                                 + 0.0245045063480129 w_k1_INTC_s003
                                 + 0.124587105999722 w_k1_INTC_s004
                                 + 0.0269117390674021 w_k1_INTC_s005
                                 + 0.0269879169138388 w_k1_INTC_s006
                                 + 0.0250437840816253 w_k1_INTC_s007
                                 + 0.043293835733998 w_k1_INTC_s010
                                 + 0.0290099940877316 w_k1_INTC_s011
                                 + 0.0258045261378624 w_k1_INTC_s012
                                 + 0.0133882659211306 w_k1_INTC_s013
                                 + 0.0468859246101303 w_k1_INTC_s014
                                 + 0.0127446893883532 w_k1_INTC_s015
                                 + 0.0251014984195753 w_k1_INTC_s016
                                 + 0.0208516290954768 w_k1_INTC_s017
                                 + 0.0959085734628707 w_k1_INTC_s100
                                 + 0.0500707125115224 w_k1_INTC_s101
                                 + 0.0318499660537231 w_k1_INTC_s102
                                 + 0.0333863607043773 w_k1_INTC_s103
                                 + 0.0231171277799481 w_k1_INTC_s104
                                 + 0.0173779221463483 w_k1_INTC_s105
                                 + 0.100172854032191 w_k1_INTC_s106
                                 + 0.0406135777374762 w_k1_INTC_s107
                                 + 0.0138358572567391 w_k1_INTC_s110
                                 + 0.010268755934088 w_k1_INTC_s111
                                 + 0.0119124778497804 w_k1_INTC_s112
                                 + 0.011330300282752 w_k1_INTC_s113
                                 + 0.0151602856174711 w_k1_INTC_s114
                                 + 0.00497461211634023 w_k1_INTC_s115
                                 + 0.00919215181772892 w_k1_INTC_s116
                                 + 0.00446794194348996 w_k1_INTC_s117
                                 + 0.0805634331950238 w_k1_KO_s000
                                 + 0.0290499892134011 w_k1_KO_s001
                                 + 0.0276853583024762 w_k1_KO_s002
                                 + 0.027309239379909 w_k1_KO_s003
                                 + 0.108597878454679 w_k1_KO_s004
                                 + 0.0294625877581495 w_k1_KO_s005
                                 + 0.0273320848382073 w_k1_KO_s006
                                 + 0.023624489265711 w_k1_KO_s007
                                 + 0.0412931836260674 w_k1_KO_s010
                                 + 0.0259493204122858 w_k1_KO_s011
                                 + 0.0287792379900966 w_k1_KO_s012
                                 + 0.0155299479275573 w_k1_KO_s013
                                 + 0.0449132233789666 w_k1_KO_s014
                                 + 0.0114418323694568 w_k1_KO_s015
                                 + 0.0249909025962246 w_k1_KO_s016
                                 + 0.0210376587834217 w_k1_KO_s017
                                 + 0.0874347659473758 w_k1_KO_s100
                                 + 0.0506068321791499 w_k1_KO_s101
                                 + 0.0335708501696086 w_k1_KO_s102
                                 + 0.0316579015830317 w_k1_KO_s103
                                 + 0.0208013707089751 w_k1_KO_s104
                                 + 0.0173899445806247 w_k1_KO_s105
                                 + 0.0772933332371162 w_k1_KO_s106
                                 + 0.0401278575392205 w_k1_KO_s107
                                 + 0.0126963940975829 w_k1_KO_s110
                                 + 0.0114016987146253 w_k1_KO_s111
                                 + 0.0121831464206713 w_k1_KO_s112
                                 + 0.012318512832215 w_k1_KO_s113
                                 + 0.0137901878415133 w_k1_KO_s114
                                 + 0.00468854289456554 w_k1_KO_s115
                                 + 0.00929465595234152 w_k1_KO_s116
                                 + 0.00512565421847285 w_k1_KO_s117
                                 + 0.0819250550059781 w_k1_MAT_s000
                                 + 0.0315313171299416 w_k1_MAT_s001
                                 + 0.0278059883675877 w_k1_MAT_s002
                                 + 0.0321643779946279 w_k1_MAT_s003
                                 + 0.108791352366088 w_k1_MAT_s004
                                 + 0.0294127059928957 w_k1_MAT_s005
                                 + 0.0263094905124322 w_k1_MAT_s006
                                 + 0.0256421249654919 w_k1_MAT_s007
                                 + 0.0430523116645983 w_k1_MAT_s010
                                 + 0.0246134897523786 w_k1_MAT_s011
                                 + 0.0325919855144131 w_k1_MAT_s012
                                 + 0.0142171885157198 w_k1_MAT_s013
                                 + 0.0396683677323507 w_k1_MAT_s014
                                 + 0.0124379543300244 w_k1_MAT_s015
                                 + 0.021437291746973 w_k1_MAT_s016
                                 + 0.0269927933140392 w_k1_MAT_s017
                                 + 0.102727626685238 w_k1_MAT_s100
                                 + 0.0586157195424236 w_k1_MAT_s101
                                 + 0.0327142981891698 w_k1_MAT_s102
                                 + 0.0317028235839249 w_k1_MAT_s103
                                 + 0.0199783057523574 w_k1_MAT_s104
                                 + 0.021003291497543 w_k1_MAT_s105
                                 + 0.0653854409893781 w_k1_MAT_s106
                                 + 0.0416296834808617 w_k1_MAT_s107
                                 + 0.0123597068786077 w_k1_MAT_s110
                                 + 0.0118929338475626 w_k1_MAT_s111
                                 + 0.0131688770118957 w_k1_MAT_s112
                                 + 0.0135046073896917 w_k1_MAT_s113
                                 + 0.0149069997792016 w_k1_MAT_s114
                                 + 0.00641492344527603 w_k1_MAT_s115
                                 + 0.0113161379587583 w_k1_MAT_s116
                                 + 0.00527278303502298 w_k1_MAT_s117
                                 + 0.0868205992889427 w_k1_MSFT_s000
                                 + 0.029027891804119 w_k1_MSFT_s001
                                 + 0.027394586413479 w_k1_MSFT_s002
                                 + 0.0250657126556264 w_k1_MSFT_s003
                                 + 0.114097008693954 w_k1_MSFT_s004
                                 + 0.0296849388624823 w_k1_MSFT_s005
                                 + 0.0260252626524486 w_k1_MSFT_s006
                                 + 0.0242774227475045 w_k1_MSFT_s007
                                 + 0.044190463966646 w_k1_MSFT_s010
                                 + 0.0277630456395767 w_k1_MSFT_s011
                                 + 0.0267854549711644 w_k1_MSFT_s012
                                 + 0.0147616110538141 w_k1_MSFT_s013
                                 + 0.0454006934047419 w_k1_MSFT_s014
                                 + 0.010740124528103 w_k1_MSFT_s015
                                 + 0.026870255520582 w_k1_MSFT_s016
                                 + 0.0224519396225626 w_k1_MSFT_s017
                                 + 0.0843529260243461 w_k1_MSFT_s100
                                 + 0.0484573882895977 w_k1_MSFT_s101
                                 + 0.0297928860426422 w_k1_MSFT_s102
                                 + 0.034018883816008 w_k1_MSFT_s103
                                 + 0.0213474566439528 w_k1_MSFT_s104
                                 + 0.0177613780701489 w_k1_MSFT_s105
                                 + 0.0817747377043259 w_k1_MSFT_s106
                                 + 0.0392371223860546 w_k1_MSFT_s107
                                 + 0.0142464675420725 w_k1_MSFT_s110
                                 + 0.011221765545189 w_k1_MSFT_s111
                                 + 0.0118571754816366 w_k1_MSFT_s112
                                 + 0.0116009126888034 w_k1_MSFT_s113
                                 + 0.0158134098551922 w_k1_MSFT_s114
                                 + 0.004987939124473 w_k1_MSFT_s115
                                 + 0.0093564829738346 w_k1_MSFT_s116
                                 + 0.0047208286105747 w_k1_MSFT_s117
                                 + 0.0785201907578044 w_k1_PG_s000
                                 + 0.0281350413515735 w_k1_PG_s001
                                 + 0.0289511112673818 w_k1_PG_s002
                                 + 0.026257760973569 w_k1_PG_s003
                                 + 0.101159926221343 w_k1_PG_s004
                                 + 0.0279704566109931 w_k1_PG_s005
                                 + 0.0270482662105596 w_k1_PG_s006
                                 + 0.0255636514193878 w_k1_PG_s007
                                 + 0.0460902813231742 w_k1_PG_s010
                                 + 0.0253211570159706 w_k1_PG_s011
                                 + 0.0266188658282809 w_k1_PG_s012
                                 + 0.0152602548407539 w_k1_PG_s013
                                 + 0.0441001951134129 w_k1_PG_s014
                                 + 0.0113081868658293 w_k1_PG_s015
                                 + 0.0260488340222175 w_k1_PG_s016
                                 + 0.0230809317542499 w_k1_PG_s017
                                 + 0.083063081716101 w_k1_PG_s100
                                 + 0.0471376899423347 w_k1_PG_s101
                                 + 0.0307085280636338 w_k1_PG_s102
                                 + 0.0325311550091626 w_k1_PG_s103
                                 + 0.0178222585957699 w_k1_PG_s104
                                 + 0.0172828259384449 w_k1_PG_s105
                                 + 0.0837542526832613 w_k1_PG_s106
                                 + 0.0370471156186395 w_k1_PG_s107
                                 + 0.0135632525509023 w_k1_PG_s110
                                 + 0.011664464448601 w_k1_PG_s111
                                 + 0.013297231781108 w_k1_PG_s112
                                 + 0.0128582119407096 w_k1_PG_s113
                                 + 0.0131625721095158 w_k1_PG_s114
                                 + 0.00528023027135239 w_k1_PG_s115
                                 + 0.00983646526411605 w_k1_PG_s116
                                 + 0.00453389584218731 w_k1_PG_s117 >= 1.09
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
