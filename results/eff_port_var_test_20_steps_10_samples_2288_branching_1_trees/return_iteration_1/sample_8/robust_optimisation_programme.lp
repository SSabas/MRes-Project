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
 w_k1_ADM_s0_return_constraint:  1.15239 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.939615 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.894066 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.15707 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.998869 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.02952 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.15177 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.940537 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.75123 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.19881 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.66471 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.804049 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.25987 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.30236 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.921298 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.986862 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.28527 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.08338 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.976358 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.827408 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.09892 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.697245 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.953967 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.03719 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.978945 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.839263 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.1679 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.11993 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.22843 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.57791 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.848639 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.34536 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.816311 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.08598 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.3443 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.19567 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.22342 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.807221 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24137 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.882021 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.911554 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.42332 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.941677 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.20356 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.18178 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.903763 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.665939 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.991413 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.75333 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.606716 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.08046 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.35447 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.15245 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.08188 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.55742 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.26984 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.946263 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.940272 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.09427 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.666054 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.960791 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.976875 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.819441 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.909453 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.26023 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.20018 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.48101 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.87155 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.02039 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.3784 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.694895 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.900892 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.44044 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.34158 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.23023 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.777205 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.34086 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.858827 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.896481 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.43585 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.966322 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.1853 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.02952 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.996381 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.652141 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.1344 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.7982 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.492661 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.865677 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.60183 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.11588 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.0603 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.99947 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.53307 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.01596 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.836112 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.27232 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.626905 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.994086 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.734524 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.955327 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.790252 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.33697 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.62891 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.42514 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 2.04051 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.936657 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.57057 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.601393 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.16319 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.44528 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.66172 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.08339 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.762117 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20937 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.916226 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.956854 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.22087 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.01216 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.01852 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.981094 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.867073 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.728935 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.07057 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.51459 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.665824 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.11027 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.23249 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.985392 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.07428 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.51109 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.46444 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.972213 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.798004 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.13811 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.697948 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.835529 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.89892 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.936412 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.76746 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.1895 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.22484 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.18983 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.60611 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.924505 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.42658 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.799903 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.1362 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.12786 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.35228 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.1916 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.835288 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.13724 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.990539 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.906704 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.49086 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.819836 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.38767 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.54825 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.16945 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.822706 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.779684 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.16926 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.620201 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.655852 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.66156 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.90323 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.734169 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.3896 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.829159 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.923487 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.3558 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.37652 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.935199 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.49904 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.796103 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.676536 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.017 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.00382 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.19217 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.51317 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.37678 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.27215 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.925571 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.663029 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.751041 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.79659 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.28876 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.948059 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.931827 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.04755 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.855175 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.01487 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.28978 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.05089 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.05519 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.0821 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.06924 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.94507 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.01527 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.21434 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.80306 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.05054 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.41678 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.914502 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.04413 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.34799 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.11753 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.937088 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.920475 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.099 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.825798 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.935944 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.887449 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.01512 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.86482 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.28417 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.32843 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.21097 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.35267 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.05147 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.45244 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.967389 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.01845 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.48827 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.18183 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.14965 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.949854 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18798 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.919927 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.839337 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.35453 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.927766 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.25564 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.30793 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.14183 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.858136 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.974145 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.44657 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.654584 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.911056 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.62438 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.35598 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.813897 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.4024 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.969223 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.06808 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.01055 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.32034 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.824181 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.15653 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.750595 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.845544 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.862214 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.20882 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.4999 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.39762 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.57223 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.02358 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.114 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.689144 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.912462 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.53491 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.24795 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.993795 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.847345 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.12742 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.962723 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.981109 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.73109 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.897864 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.46631 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.50069 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.22095 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.638714 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.668253 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.29971 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.468293 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.564586 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.62853 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.95143 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.717083 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.35245 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.00556 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.847748 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.27318 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.59151 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.840767 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.26965 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.625306 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.607586 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.865469 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.00403 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.00014 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.36087 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.74457 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.37144 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.885471 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.69339 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.699127 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.60199 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.261 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.13558 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.651887 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.16758 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.900129 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.959663 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.26106 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.04321 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.01135 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.00458 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.95452 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.784913 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.0176 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.36089 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.709557 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.927237 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.38373 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.792252 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.09784 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.30374 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.32225 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.985492 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.756452 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.0715 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.837362 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.901961 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.817014 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.00511 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.794739 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.15175 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.37561 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.1631 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.40961 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.974545 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.5085 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.852814 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.185 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.22965 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.34288 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.076 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.8788 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.1333 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.93891 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.878618 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10665 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.962192 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.07159 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.19801 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.03235 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.87887 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.08808 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.37159 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.798265 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.01136 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.34946 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.18453 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.975576 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.22873 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.23047 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.10507 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.03778 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.14202 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.00341 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.03514 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.983065 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.912759 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.915461 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.1379 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.27478 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.26518 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.38814 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.969959 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.22553 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.868987 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.07339 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.33816 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.20442 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.0644 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.929045 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.331000472432318 z_k1_s0000
                                 - 0.349000452460318 z_k1_s0001
                                 - 0.743999643157745 z_k1_s0002
                                 - 0.634001132154558 z_k1_s0003
                                 - 0.17000031909824 z_k1_s0004
                                 - 0.452001295038083 z_k1_s0005
                                 - 0.498002142277129 z_k1_s0006
                                 - 0.337000231441436 z_k1_s0007
                                 - 0.926995234282829 z_k1_s0010
                                 - 1.01200159053812 z_k1_s0011
                                 - 1.29299907982143 z_k1_s0012
                                 - 0.807997060912842 z_k1_s0013
                                 - 0.937000488542869 z_k1_s0014
                                 - 0.694998975702474 z_k1_s0015
                                 - 1.47099454496521 z_k1_s0016
                                 - 0.517999439390096 z_k1_s0017
                                 - 0.290000063291781 z_k1_s0020
                                 - 0.502998803178479 z_k1_s0021
                                 - 0.371000428769836 z_k1_s0022
                                 - 0.267999879890332 z_k1_s0023
                                 - 0.258000093907861 z_k1_s0024
                                 - 0.412999911536222 z_k1_s0025
                                 - 0.187000059612287 z_k1_s0026
                                 - 0.435999173955902 z_k1_s0027
                                 - 0.444001096994791 z_k1_s0030
                                 - 0.389000714502391 z_k1_s0031
                                 - 0.550001293319096 z_k1_s0032
                                 - 1.22199936149343 z_k1_s0033
                                 - 1.48100153091836 z_k1_s0034
                                 - 0.372000987525498 z_k1_s0035
                                 - 0.776000195322691 z_k1_s0036
                                 - 1.09500140113665 z_k1_s0037
                                 - 0.180999878371143 z_k1_s0040
                                 - 0.499000514627027 z_k1_s0041
                                 - 0.144000092600788 z_k1_s0042
                                 - 0.390998784551527 z_k1_s0043
                                 - 0.427001132576596 z_k1_s0044
                                 - 0.332999134733132 z_k1_s0045
                                 - 0.239000059589508 z_k1_s0046
                                 - 0.442998928699028 z_k1_s0047
                                 - 0.0930000842310013 z_k1_s0050
                                 - 0.231000128625339 z_k1_s0051
                                 - 0.204000110887311 z_k1_s0052
                                 - 0.394000012088109 z_k1_s0053
                                 - 0.170000196506158 z_k1_s0054
                                 - 0.358000127237493 z_k1_s0055
                                 - 0.394000012088109 z_k1_s0056
                                 - 0.237999817068331 z_k1_s0057
                                 - 0.27500008634291 z_k1_s0060
                                 - 0.707001666961796 z_k1_s0061
                                 - 0.177000196601891 z_k1_s0062
                                 - 0.289000070591626 z_k1_s0063
                                 - 0.467000617840082 z_k1_s0064
                                 - 0.336000853170002 z_k1_s0065
                                 - 0.216000020808946 z_k1_s0066
                                 - 0.611001862913508 z_k1_s0067
                                 - 0.206000043807032 z_k1_s0070
                                 - 0.217999972751096 z_k1_s0071
                                 - 0.223000040344502 z_k1_s0072
                                 - 0.311000296868008 z_k1_s0073
                                 - 0.341000119228167 z_k1_s0074
                                 - 0.571999917801952 z_k1_s0075
                                 - 0.477999696806414 z_k1_s0076
                                 - 0.566998975408127 z_k1_s0077
                                 - 0.0319999388542435 z_k1_s0100
                                 - 0.0229999778014514 z_k1_s0101
                                 - 0.0529999639772324 z_k1_s0102
                                 - 0.0369999945501107 z_k1_s0103
                                 - 0.0749997914397416 z_k1_s0104
                                 - 0.0179999569055264 z_k1_s0105
                                 - 0.0709998860828167 z_k1_s0106
                                 - 0.0389999472285732 z_k1_s0107
                                 - 0.313000349657712 z_k1_s0110
                                 - 0.160000477697307 z_k1_s0111
                                 - 0.110000070691835 z_k1_s0112
                                 - 0.214000619095143 z_k1_s0113
                                 - 0.284000332462592 z_k1_s0114
                                 - 0.141000034383156 z_k1_s0115
                                 - 0.0650000057269843 z_k1_s0116
                                 - 0.299000724184301 z_k1_s0117
                                 - 0.0379999725106244 z_k1_s0120
                                 - 0.0479999456397616 z_k1_s0121
                                 - 0.0429999324252276 z_k1_s0122
                                 - 0.138999823704751 z_k1_s0123
                                 - 0.102000077696743 z_k1_s0124
                                 - 0.0740000250153246 z_k1_s0125
                                 - 0.0349999326019455 z_k1_s0126
                                 - 0.0539997589574651 z_k1_s0127
                                 - 0.105000192636567 z_k1_s0130
                                 - 0.088999571349819 z_k1_s0131
                                 - 0.146999859891698 z_k1_s0132
                                 - 0.0599999051782904 z_k1_s0133
                                 - 0.0629999106821924 z_k1_s0134
                                 - 0.0949997189574545 z_k1_s0135
                                 - 0.0409999613867996 z_k1_s0136
                                 - 0.083000106741342 z_k1_s0137
                                 - 0.358998879387524 z_k1_s0140
                                 - 0.0999997085647088 z_k1_s0141
                                 - 0.182999419353557 z_k1_s0142
                                 - 0.195000073899285 z_k1_s0143
                                 - 0.144999543618928 z_k1_s0144
                                 - 0.175999460033967 z_k1_s0145
                                 - 0.396998417123152 z_k1_s0146
                                 - 0.134999674162157 z_k1_s0147
                                 - 0.0909996081892778 z_k1_s0150
                                 - 0.103999642216207 z_k1_s0151
                                 - 0.0549999404218732 z_k1_s0152
                                 - 0.0469999582514077 z_k1_s0153
                                 - 0.115000058700055 z_k1_s0154
                                 - 0.0629999225923388 z_k1_s0155
                                 - 0.0439999649374831 z_k1_s0156
                                 - 0.110999563615442 z_k1_s0157
                                 - 0.12799968319921 z_k1_s0160
                                 - 0.05699992552455 z_k1_s0161
                                 - 0.0810001325240089 z_k1_s0162
                                 - 0.0469999502746117 z_k1_s0163
                                 - 0.0429999601746364 z_k1_s0164
                                 - 0.145999638649099 z_k1_s0165
                                 - 0.105000073123861 z_k1_s0166
                                 - 0.0589999205745377 z_k1_s0167
                                 - 0.120000367057618 z_k1_s0170
                                 - 0.0560000324934014 z_k1_s0171
                                 - 0.0890000813841887 z_k1_s0172
                                 - 0.206999956523642 z_k1_s0173
                                 - 0.172000456715843 z_k1_s0174
                                 - 0.1620005996279 z_k1_s0175
                                 - 0.138000318016146 z_k1_s0176
                                 - 0.0970000711546582 z_k1_s0177
                                 - 0.465001654273096 z_k1_s1000
                                 - 0.327000217417712 z_k1_s1001
                                 - 0.443999803099204 z_k1_s1002
                                 - 0.30600015424409 z_k1_s1003
                                 - 0.697001841333985 z_k1_s1004
                                 - 0.40199967675196 z_k1_s1005
                                 - 0.5329993045492 z_k1_s1006
                                 - 0.40199967675196 z_k1_s1007
                                 - 1.71900456664384 z_k1_s1010
                                 - 1.01400256283674 z_k1_s1011
                                 - 1.47300594246892 z_k1_s1012
                                 - 0.617001099255268 z_k1_s1013
                                 - 0.686001646530779 z_k1_s1014
                                 - 0.448000922183028 z_k1_s1015
                                 - 0.436000696534826 z_k1_s1016
                                 - 1.10900059821008 z_k1_s1017
                                 - 0.692001162304752 z_k1_s1020
                                 - 1.75400583090786 z_k1_s1021
                                 - 0.995002409652991 z_k1_s1022
                                 - 0.880003521222482 z_k1_s1023
                                 - 0.844003936009537 z_k1_s1024
                                 - 0.475000887767681 z_k1_s1025
                                 - 0.59600098773529 z_k1_s1026
                                 - 1.44800551459301 z_k1_s1027
                                 - 0.67700119999939 z_k1_s1030
                                 - 2.77300998200156 z_k1_s1031
                                 - 0.629001654023568 z_k1_s1032
                                 - 0.650001229537533 z_k1_s1033
                                 - 0.601001617737062 z_k1_s1034
                                 - 1.03300385533227 z_k1_s1035
                                 - 1.9180033890275 z_k1_s1036
                                 - 0.753001685663548 z_k1_s1037
                                 - 0.342001089072416 z_k1_s1040
                                 - 0.323001102585279 z_k1_s1041
                                 - 1.05200495500996 z_k1_s1042
                                 - 0.41100111012065 z_k1_s1043
                                 - 0.2800009929245 z_k1_s1044
                                 - 0.825002568713989 z_k1_s1045
                                 - 0.952002221279886 z_k1_s1046
                                 - 0.256000869750859 z_k1_s1047
                                 - 0.386000889207684 z_k1_s1050
                                 - 0.31000007629647 z_k1_s1051
                                 - 0.202999060355653 z_k1_s1052
                                 - 0.182000013277289 z_k1_s1053
                                 - 0.1879999643282 z_k1_s1054
                                 - 0.337000442225436 z_k1_s1055
                                 - 0.215999019432611 z_k1_s1056
                                 - 0.131999900119817 z_k1_s1057
                                 - 0.460000627571738 z_k1_s1060
                                 - 0.610999014309324 z_k1_s1061
                                 - 0.745998914227102 z_k1_s1062
                                 - 0.414999571265756 z_k1_s1063
                                 - 0.244000133903259 z_k1_s1064
                                 - 0.275000062491784 z_k1_s1065
                                 - 0.327001355756478 z_k1_s1066
                                 - 0.19100016108739 z_k1_s1067
                                 - 0.447999536286596 z_k1_s1070
                                 - 0.27299987152465 z_k1_s1071
                                 - 0.843999566682799 z_k1_s1072
                                 - 0.227000095252372 z_k1_s1073
                                 - 0.278999909342775 z_k1_s1074
                                 - 0.402999252650653 z_k1_s1075
                                 - 0.434999659813997 z_k1_s1076
                                 - 0.173000063089252 z_k1_s1077
                                 - 0.296001387362605 z_k1_s1100
                                 - 0.134999943138322 z_k1_s1101
                                 - 0.389999516288471 z_k1_s1102
                                 - 0.31599988819785 z_k1_s1103
                                 - 0.62399865106153 z_k1_s1104
                                 - 0.144000131014218 z_k1_s1105
                                 - 0.363000390160843 z_k1_s1106
                                 - 0.606998775351524 z_k1_s1107
                                 - 0.200999796225428 z_k1_s1110
                                 - 0.176000159779364 z_k1_s1111
                                 - 0.352999934216697 z_k1_s1112
                                 - 0.176000159779364 z_k1_s1113
                                 - 0.153999985306996 z_k1_s1114
                                 - 0.295000654061869 z_k1_s1115
                                 - 0.113999948993504 z_k1_s1116
                                 - 0.0759999144956871 z_k1_s1117
                                 - 0.444000534038509 z_k1_s1120
                                 - 0.315000423440902 z_k1_s1121
                                 - 0.484000483122908 z_k1_s1122
                                 - 0.512000813882547 z_k1_s1123
                                 - 0.588999891468756 z_k1_s1124
                                 - 0.619000769283454 z_k1_s1125
                                 - 0.410002226119287 z_k1_s1126
                                 - 0.291000453990262 z_k1_s1127
                                 - 0.171999886053319 z_k1_s1130
                                 - 0.175000348519127 z_k1_s1131
                                 - 0.229000548907839 z_k1_s1132
                                 - 0.104999938311615 z_k1_s1133
                                 - 0.240000439283395 z_k1_s1134
                                 - 0.195000272292801 z_k1_s1135
                                 - 0.135999752460844 z_k1_s1136
                                 - 0.101999882045598 z_k1_s1137
                                 - 0.508999794227053 z_k1_s1140
                                 - 0.118999886794974 z_k1_s1141
                                 - 0.152999899964936 z_k1_s1142
                                 - 0.141999923815811 z_k1_s1143
                                 - 0.117999961372277 z_k1_s1144
                                 - 0.125999842653527 z_k1_s1145
                                 - 0.320000030246762 z_k1_s1146
                                 - 0.105999900500564 z_k1_s1147
                                 - 0.41499959524525 z_k1_s1150
                                 - 0.321999886782937 z_k1_s1151
                                 - 0.269999774754717 z_k1_s1152
                                 - 0.39399929023786 z_k1_s1153
                                 - 0.177999930273802 z_k1_s1154
                                 - 0.677999172530897 z_k1_s1155
                                 - 0.43499974887142 z_k1_s1156
                                 - 0.181999960999036 z_k1_s1157
                                 - 0.277000403193488 z_k1_s1160
                                 - 0.336999680297322 z_k1_s1161
                                 - 0.146999832003302 z_k1_s1162
                                 - 0.459998858459734 z_k1_s1163
                                 - 0.2050005162694 z_k1_s1164
                                 - 0.0999999896389104 z_k1_s1165
                                 - 0.10999993202284 z_k1_s1166
                                 - 0.250000445596955 z_k1_s1167
                                 - 0.841002151159579 z_k1_s1170
                                 - 0.559999454605169 z_k1_s1171
                                 - 0.725000330033737 z_k1_s1172
                                 - 0.366999987946688 z_k1_s1173
                                 - 0.661998388215153 z_k1_s1174
                                 - 0.320999894596943 z_k1_s1175
                                 - 0.945001497271829 z_k1_s1176
                                 - 1.10500134144473 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.938462 w_k1_ADM_s000
                                 + 0.978066 w_k1_AXP_s000
                                 + 0.912601 w_k1_DIS_s000
                                 + 0.828704 w_k1_FDX_s000
                                 + 1.10896 w_k1_IBM_s000
                                 + 0.801483 w_k1_INTC_s000
                                 + 1.01572 w_k1_KO_s000 + 1.12194 w_k1_MAT_s000
                                 + 0.911627 w_k1_MSFT_s000
                                 + 0.95025 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.07225 w_k1_ADM_s000
                                 + 1.09074 w_k1_AXP_s000
                                 + 1.18524 w_k1_DIS_s000
                                 + 0.962692 w_k1_FDX_s000
                                 + 1.47508 w_k1_IBM_s000
                                 + 0.897902 w_k1_INTC_s000
                                 + 1.29778 w_k1_KO_s000 + 1.36009 w_k1_MAT_s000
                                 + 0.860764 w_k1_MSFT_s000
                                 + 1.03581 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.809593 w_k1_ADM_s000
                                 + 1.00785 w_k1_AXP_s000
                                 + 1.11436 w_k1_DIS_s000
                                 + 0.988855 w_k1_FDX_s000
                                 + 0.919108 w_k1_IBM_s000
                                 + 0.993846 w_k1_INTC_s000
                                 + 0.983314 w_k1_KO_s000
                                 + 0.897517 w_k1_MAT_s000
                                 + 0.989723 w_k1_MSFT_s000
                                 + 0.902629 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.96991 w_k1_ADM_s000
                                 + 1.29871 w_k1_AXP_s000 + 1.2089 w_k1_DIS_s000
                                 + 1.23913 w_k1_FDX_s000
                                 + 0.742091 w_k1_IBM_s000
                                 + 1.28766 w_k1_INTC_s000
                                 + 0.938226 w_k1_KO_s000
                                 + 0.727565 w_k1_MAT_s000
                                 + 1.10638 w_k1_MSFT_s000
                                 + 1.00292 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.661014 w_k1_ADM_s000
                                 + 0.668961 w_k1_AXP_s000
                                 + 0.655837 w_k1_DIS_s000
                                 + 0.669283 w_k1_FDX_s000
                                 + 1.17293 w_k1_IBM_s000
                                 + 0.782245 w_k1_INTC_s000
                                 + 0.951588 w_k1_KO_s000
                                 + 0.835254 w_k1_MAT_s000
                                 + 0.717738 w_k1_MSFT_s000
                                 + 0.914494 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.03332 w_k1_ADM_s000
                                 + 1.22077 w_k1_AXP_s000
                                 + 1.23273 w_k1_DIS_s000
                                 + 1.07333 w_k1_FDX_s000
                                 + 1.31189 w_k1_IBM_s000
                                 + 0.838784 w_k1_INTC_s000
                                 + 1.21106 w_k1_KO_s000 + 1.67301 w_k1_MAT_s000
                                 + 1.02923 w_k1_MSFT_s000
                                 + 1.13926 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.856141 w_k1_ADM_s000
                                 + 0.849232 w_k1_AXP_s000
                                 + 0.750658 w_k1_DIS_s000
                                 + 0.787511 w_k1_FDX_s000
                                 + 0.743519 w_k1_IBM_s000
                                 + 0.919206 w_k1_INTC_s000
                                 + 0.80049 w_k1_KO_s000
                                 + 0.851168 w_k1_MAT_s000
                                 + 0.773633 w_k1_MSFT_s000
                                 + 0.735537 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.02193 w_k1_ADM_s000
                                 + 1.69505 w_k1_AXP_s000
                                 + 1.80953 w_k1_DIS_s000
                                 + 1.37648 w_k1_FDX_s000
                                 + 1.26792 w_k1_IBM_s000
                                 + 1.44239 w_k1_INTC_s000
                                 + 1.35929 w_k1_KO_s000 + 1.90626 w_k1_MAT_s000
                                 + 1.41417 w_k1_MSFT_s000
                                 + 1.32388 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.982983 w_k1_ADM_s001
                                 + 0.988857 w_k1_AXP_s001
                                 + 0.736169 w_k1_DIS_s001
                                 + 0.880522 w_k1_FDX_s001
                                 + 0.940492 w_k1_IBM_s001
                                 + 0.737962 w_k1_INTC_s001
                                 + 0.81083 w_k1_KO_s001
                                 + 0.831797 w_k1_MAT_s001
                                 + 0.818181 w_k1_MSFT_s001
                                 + 0.965264 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.23682 w_k1_ADM_s001
                                 + 1.27133 w_k1_AXP_s001
                                 + 1.61449 w_k1_DIS_s001
                                 + 1.35912 w_k1_FDX_s001 + 1.1926 w_k1_IBM_s001
                                 + 1.10056 w_k1_INTC_s001
                                 + 1.18217 w_k1_KO_s001 + 1.09342 w_k1_MAT_s001
                                 + 1.29011 w_k1_MSFT_s001
                                 + 1.20859 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.1076 w_k1_ADM_s001
                                 + 1.08843 w_k1_AXP_s001
                                 + 0.967426 w_k1_DIS_s001
                                 + 1.07367 w_k1_FDX_s001
                                 + 0.761549 w_k1_IBM_s001
                                 + 0.882457 w_k1_INTC_s001
                                 + 0.851522 w_k1_KO_s001
                                 + 0.794144 w_k1_MAT_s001
                                 + 0.985582 w_k1_MSFT_s001
                                 + 1.05932 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.91736 w_k1_ADM_s001
                                 + 0.865626 w_k1_AXP_s001
                                 + 0.734828 w_k1_DIS_s001
                                 + 0.831479 w_k1_FDX_s001
                                 + 1.17728 w_k1_IBM_s001
                                 + 0.734822 w_k1_INTC_s001
                                 + 0.888273 w_k1_KO_s001
                                 + 1.12424 w_k1_MAT_s001
                                 + 0.837937 w_k1_MSFT_s001
                                 + 0.970559 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.879254 w_k1_ADM_s001
                                 + 0.615256 w_k1_AXP_s001
                                 + 0.588783 w_k1_DIS_s001
                                 + 0.758348 w_k1_FDX_s001
                                 + 0.692992 w_k1_IBM_s001
                                 + 0.690118 w_k1_INTC_s001
                                 + 0.726722 w_k1_KO_s001
                                 + 0.52231 w_k1_MAT_s001
                                 + 0.675717 w_k1_MSFT_s001
                                 + 0.838919 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.845564 w_k1_ADM_s001
                                 + 0.875101 w_k1_AXP_s001
                                 + 0.930122 w_k1_DIS_s001
                                 + 0.922369 w_k1_FDX_s001
                                 + 0.948272 w_k1_IBM_s001
                                 + 0.924858 w_k1_INTC_s001
                                 + 0.93301 w_k1_KO_s001
                                 + 0.717256 w_k1_MAT_s001
                                 + 1.06225 w_k1_MSFT_s001
                                 + 0.936497 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.2893 w_k1_ADM_s001
                                 + 1.40248 w_k1_AXP_s001
                                 + 1.18565 w_k1_DIS_s001
                                 + 1.25865 w_k1_FDX_s001
                                 + 1.08175 w_k1_IBM_s001 + 1.136 w_k1_INTC_s001
                                 + 0.950111 w_k1_KO_s001
                                 + 1.13528 w_k1_MAT_s001
                                 + 1.08684 w_k1_MSFT_s001
                                 + 1.09038 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.87756 w_k1_ADM_s001
                                 + 0.990405 w_k1_AXP_s001
                                 + 0.792994 w_k1_DIS_s001
                                 + 0.870983 w_k1_FDX_s001
                                 + 1.53801 w_k1_IBM_s001
                                 + 0.833416 w_k1_INTC_s001
                                 + 1.19359 w_k1_KO_s001 + 1.2068 w_k1_MAT_s001
                                 + 0.785178 w_k1_MSFT_s001
                                 + 1.02878 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.16129 w_k1_ADM_s002
                                 + 1.03133 w_k1_AXP_s002
                                 + 1.27856 w_k1_DIS_s002
                                 + 1.10552 w_k1_FDX_s002
                                 + 0.82622 w_k1_IBM_s002
                                 + 0.942901 w_k1_INTC_s002
                                 + 1.08692 w_k1_KO_s002 + 1.01466 w_k1_MAT_s002
                                 + 1.2025 w_k1_MSFT_s002 + 1.06578 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.29152 w_k1_ADM_s002
                                 + 1.48615 w_k1_AXP_s002
                                 + 1.31232 w_k1_DIS_s002
                                 + 1.19409 w_k1_FDX_s002 + 1.3885 w_k1_IBM_s002
                                 + 1.05133 w_k1_INTC_s002 + 1.1098 w_k1_KO_s002
                                 + 1.69564 w_k1_MAT_s002
                                 + 1.21467 w_k1_MSFT_s002 + 1.156 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.61217 w_k1_ADM_s002
                                 + 1.49388 w_k1_AXP_s002
                                 + 1.30531 w_k1_DIS_s002
                                 + 1.31618 w_k1_FDX_s002
                                 + 1.06036 w_k1_IBM_s002
                                 + 0.990688 w_k1_INTC_s002
                                 + 1.13263 w_k1_KO_s002 + 1.39108 w_k1_MAT_s002
                                 + 1.15107 w_k1_MSFT_s002 + 1.1394 w_k1_PG_s002
                                 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.10272 w_k1_ADM_s002
                                 + 1.04358 w_k1_AXP_s002
                                 + 0.919855 w_k1_DIS_s002
                                 + 0.951911 w_k1_FDX_s002
                                 + 1.37974 w_k1_IBM_s002
                                 + 0.984145 w_k1_INTC_s002
                                 + 1.26759 w_k1_KO_s002 + 1.72764 w_k1_MAT_s002
                                 + 1.07527 w_k1_MSFT_s002
                                 + 1.09793 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.1907 w_k1_ADM_s002
                                 + 0.906129 w_k1_AXP_s002
                                 + 1.03277 w_k1_DIS_s002
                                 + 1.05003 w_k1_FDX_s002
                                 + 0.584665 w_k1_IBM_s002
                                 + 0.977031 w_k1_INTC_s002
                                 + 0.808413 w_k1_KO_s002
                                 + 0.649326 w_k1_MAT_s002
                                 + 1.16768 w_k1_MSFT_s002
                                 + 0.931343 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.920941 w_k1_ADM_s002
                                 + 1.00558 w_k1_AXP_s002
                                 + 0.958369 w_k1_DIS_s002
                                 + 0.977631 w_k1_FDX_s002
                                 + 0.912265 w_k1_IBM_s002
                                 + 0.952484 w_k1_INTC_s002
                                 + 0.940904 w_k1_KO_s002
                                 + 1.25413 w_k1_MAT_s002
                                 + 1.0242 w_k1_MSFT_s002 + 1.01839 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.08827 w_k1_ADM_s002
                                 + 0.968578 w_k1_AXP_s002
                                 + 0.683154 w_k1_DIS_s002
                                 + 0.8098 w_k1_FDX_s002 + 1.219 w_k1_IBM_s002
                                 + 0.591629 w_k1_INTC_s002
                                 + 1.00385 w_k1_KO_s002 + 1.26789 w_k1_MAT_s002
                                 + 0.795941 w_k1_MSFT_s002
                                 + 0.960297 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.846442 w_k1_ADM_s002
                                 + 0.853254 w_k1_AXP_s002
                                 + 0.686996 w_k1_DIS_s002
                                 + 0.783357 w_k1_FDX_s002
                                 + 0.773164 w_k1_IBM_s002
                                 + 0.773354 w_k1_INTC_s002
                                 + 0.687669 w_k1_KO_s002
                                 + 0.691599 w_k1_MAT_s002
                                 + 0.825104 w_k1_MSFT_s002
                                 + 0.760541 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.900694 w_k1_ADM_s003
                                 + 1.09823 w_k1_AXP_s003
                                 + 0.739157 w_k1_DIS_s003
                                 + 0.897914 w_k1_FDX_s003
                                 + 1.06483 w_k1_IBM_s003
                                 + 1.09299 w_k1_INTC_s003
                                 + 0.95302 w_k1_KO_s003 + 1.35724 w_k1_MAT_s003
                                 + 0.863144 w_k1_MSFT_s003
                                 + 1.05532 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.981081 w_k1_ADM_s003
                                 + 0.653549 w_k1_AXP_s003
                                 + 0.769747 w_k1_DIS_s003
                                 + 0.893648 w_k1_FDX_s003
                                 + 0.650564 w_k1_IBM_s003
                                 + 0.911104 w_k1_INTC_s003
                                 + 0.916504 w_k1_KO_s003
                                 + 0.663707 w_k1_MAT_s003
                                 + 1.05225 w_k1_MSFT_s003
                                 + 0.906091 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.795903 w_k1_ADM_s003
                                 + 0.69919 w_k1_AXP_s003
                                 + 0.600106 w_k1_DIS_s003
                                 + 0.728139 w_k1_FDX_s003
                                 + 0.704339 w_k1_IBM_s003
                                 + 0.909837 w_k1_INTC_s003
                                 + 0.761645 w_k1_KO_s003
                                 + 0.591587 w_k1_MAT_s003
                                 + 0.776396 w_k1_MSFT_s003
                                 + 0.788084 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.914571 w_k1_ADM_s003
                                 + 1.08294 w_k1_AXP_s003
                                 + 1.12503 w_k1_DIS_s003 + 1.1167 w_k1_FDX_s003
                                 + 0.838501 w_k1_IBM_s003
                                 + 1.03909 w_k1_INTC_s003
                                 + 1.05158 w_k1_KO_s003 + 1.27563 w_k1_MAT_s003
                                 + 1.12231 w_k1_MSFT_s003
                                 + 1.01641 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.01289 w_k1_ADM_s003
                                 + 1.18744 w_k1_AXP_s003
                                 + 1.39278 w_k1_DIS_s003
                                 + 1.30787 w_k1_FDX_s003
                                 + 1.35106 w_k1_IBM_s003
                                 + 1.29717 w_k1_INTC_s003
                                 + 1.44854 w_k1_KO_s003 + 1.69084 w_k1_MAT_s003
                                 + 1.31421 w_k1_MSFT_s003
                                 + 1.31401 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.662367 w_k1_ADM_s003
                                 + 0.6169 w_k1_AXP_s003
                                 + 0.555257 w_k1_DIS_s003
                                 + 0.687376 w_k1_FDX_s003
                                 + 0.962461 w_k1_IBM_s003
                                 + 0.928048 w_k1_INTC_s003
                                 + 0.891227 w_k1_KO_s003
                                 + 0.844762 w_k1_MAT_s003
                                 + 0.79561 w_k1_MSFT_s003
                                 + 0.805612 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.827995 w_k1_ADM_s003
                                 + 0.691232 w_k1_AXP_s003
                                 + 0.946132 w_k1_DIS_s003
                                 + 0.844445 w_k1_FDX_s003
                                 + 1.17943 w_k1_IBM_s003
                                 + 1.11853 w_k1_INTC_s003
                                 + 1.17306 w_k1_KO_s003 + 1.03334 w_k1_MAT_s003
                                 + 1.04387 w_k1_MSFT_s003
                                 + 0.930417 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.952805 w_k1_ADM_s003
                                 + 0.935239 w_k1_AXP_s003
                                 + 0.817095 w_k1_DIS_s003
                                 + 0.968018 w_k1_FDX_s003
                                 + 1.01071 w_k1_IBM_s003
                                 + 1.0646 w_k1_INTC_s003 + 1.16133 w_k1_KO_s003
                                 + 1.07501 w_k1_MAT_s003
                                 + 1.03736 w_k1_MSFT_s003
                                 + 0.927208 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.09396 w_k1_ADM_s004
                                 + 1.38157 w_k1_AXP_s004 + 1.2425 w_k1_DIS_s004
                                 + 1.22191 w_k1_FDX_s004
                                 + 0.872426 w_k1_IBM_s004
                                 + 1.26679 w_k1_INTC_s004
                                 + 1.00608 w_k1_KO_s004
                                 + 0.812622 w_k1_MAT_s004
                                 + 1.06299 w_k1_MSFT_s004
                                 + 0.989516 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.913191 w_k1_ADM_s004
                                 + 1.0553 w_k1_AXP_s004
                                 + 0.849663 w_k1_DIS_s004
                                 + 0.918851 w_k1_FDX_s004
                                 + 0.992927 w_k1_IBM_s004
                                 + 1.04805 w_k1_INTC_s004
                                 + 0.856919 w_k1_KO_s004
                                 + 1.17543 w_k1_MAT_s004
                                 + 0.908567 w_k1_MSFT_s004
                                 + 0.89107 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.24958 w_k1_ADM_s004
                                 + 1.69544 w_k1_AXP_s004
                                 + 1.80455 w_k1_DIS_s004 + 1.4386 w_k1_FDX_s004
                                 + 1.80085 w_k1_IBM_s004
                                 + 1.72216 w_k1_INTC_s004
                                 + 1.54559 w_k1_KO_s004 + 1.38773 w_k1_MAT_s004
                                 + 1.37039 w_k1_MSFT_s004
                                 + 1.44759 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.05594 w_k1_ADM_s004
                                 + 0.831008 w_k1_AXP_s004
                                 + 0.956699 w_k1_DIS_s004
                                 + 1.05837 w_k1_FDX_s004
                                 + 0.671194 w_k1_IBM_s004
                                 + 1.29821 w_k1_INTC_s004
                                 + 0.905299 w_k1_KO_s004
                                 + 0.512688 w_k1_MAT_s004
                                 + 0.994829 w_k1_MSFT_s004
                                 + 0.886161 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.01005 w_k1_ADM_s004
                                 + 0.984832 w_k1_AXP_s004
                                 + 1.23615 w_k1_DIS_s004
                                 + 1.03575 w_k1_FDX_s004
                                 + 1.25387 w_k1_IBM_s004
                                 + 1.20749 w_k1_INTC_s004
                                 + 1.22721 w_k1_KO_s004 + 1.32675 w_k1_MAT_s004
                                 + 0.987754 w_k1_MSFT_s004
                                 + 1.18136 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.839112 w_k1_ADM_s004
                                 + 0.841615 w_k1_AXP_s004
                                 + 0.627916 w_k1_DIS_s004
                                 + 0.800883 w_k1_FDX_s004
                                 + 0.968952 w_k1_IBM_s004
                                 + 0.912849 w_k1_INTC_s004
                                 + 0.795475 w_k1_KO_s004
                                 + 1.02205 w_k1_MAT_s004
                                 + 0.77052 w_k1_MSFT_s004
                                 + 0.842649 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.05904 w_k1_ADM_s004
                                 + 1.21357 w_k1_AXP_s004
                                 + 1.45664 w_k1_DIS_s004
                                 + 1.35788 w_k1_FDX_s004
                                 + 1.11711 w_k1_IBM_s004
                                 + 1.52272 w_k1_INTC_s004
                                 + 1.19537 w_k1_KO_s004
                                 + 0.999869 w_k1_MAT_s004
                                 + 1.32976 w_k1_MSFT_s004
                                 + 1.18022 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.711796 w_k1_ADM_s004
                                 + 0.61103 w_k1_AXP_s004
                                 + 0.782026 w_k1_DIS_s004
                                 + 0.832933 w_k1_FDX_s004
                                 + 0.760752 w_k1_IBM_s004
                                 + 0.97871 w_k1_INTC_s004
                                 + 0.757279 w_k1_KO_s004 + 0.7018 w_k1_MAT_s004
                                 + 0.85934 w_k1_MSFT_s004
                                 + 0.809281 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.926688 w_k1_ADM_s005
                                 + 1.4005 w_k1_AXP_s005
                                 + 0.990959 w_k1_DIS_s005
                                 + 1.08685 w_k1_FDX_s005
                                 + 0.765117 w_k1_IBM_s005
                                 + 0.783741 w_k1_INTC_s005
                                 + 0.804721 w_k1_KO_s005
                                 + 1.65698 w_k1_MAT_s005
                                 + 0.849226 w_k1_MSFT_s005
                                 + 0.959266 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.42352 w_k1_ADM_s005
                                 + 1.67984 w_k1_AXP_s005
                                 + 1.61097 w_k1_DIS_s005 + 1.4012 w_k1_FDX_s005
                                 + 1.29308 w_k1_IBM_s005
                                 + 1.2989 w_k1_INTC_s005 + 1.34791 w_k1_KO_s005
                                 + 1.44016 w_k1_MAT_s005 + 1.228 w_k1_MSFT_s005
                                 + 1.18907 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.18157 w_k1_ADM_s005
                                 + 1.33872 w_k1_AXP_s005
                                 + 1.32476 w_k1_DIS_s005
                                 + 1.34874 w_k1_FDX_s005
                                 + 0.840691 w_k1_IBM_s005
                                 + 0.935261 w_k1_INTC_s005
                                 + 0.941326 w_k1_KO_s005
                                 + 1.31148 w_k1_MAT_s005
                                 + 1.01644 w_k1_MSFT_s005 + 1.0012 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.03688 w_k1_ADM_s005
                                 + 1.22398 w_k1_AXP_s005
                                 + 1.02585 w_k1_DIS_s005
                                 + 0.985786 w_k1_FDX_s005
                                 + 1.2726 w_k1_IBM_s005
                                 + 0.888816 w_k1_INTC_s005
                                 + 1.11017 w_k1_KO_s005 + 1.87149 w_k1_MAT_s005
                                 + 0.917649 w_k1_MSFT_s005
                                 + 1.16972 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.991474 w_k1_ADM_s005
                                 + 1.02522 w_k1_AXP_s005
                                 + 1.39158 w_k1_DIS_s005
                                 + 1.15794 w_k1_FDX_s005
                                 + 0.993133 w_k1_IBM_s005
                                 + 1.33488 w_k1_INTC_s005
                                 + 1.16223 w_k1_KO_s005
                                 + 0.953907 w_k1_MAT_s005
                                 + 1.22473 w_k1_MSFT_s005
                                 + 1.16182 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.757373 w_k1_ADM_s005
                                 + 0.708479 w_k1_AXP_s005
                                 + 0.616141 w_k1_DIS_s005
                                 + 0.713217 w_k1_FDX_s005
                                 + 0.84095 w_k1_IBM_s005
                                 + 0.753737 w_k1_INTC_s005
                                 + 0.73659 w_k1_KO_s005
                                 + 0.974965 w_k1_MAT_s005
                                 + 0.752916 w_k1_MSFT_s005
                                 + 0.832581 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.06391 w_k1_ADM_s005
                                 + 0.994283 w_k1_AXP_s005
                                 + 0.981401 w_k1_DIS_s005
                                 + 1.04405 w_k1_FDX_s005
                                 + 0.646616 w_k1_IBM_s005
                                 + 0.895455 w_k1_INTC_s005
                                 + 0.817563 w_k1_KO_s005
                                 + 0.867291 w_k1_MAT_s005
                                 + 1.07118 w_k1_MSFT_s005
                                 + 0.892427 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.877626 w_k1_ADM_s005
                                 + 0.970315 w_k1_AXP_s005
                                 + 1.04155 w_k1_DIS_s005
                                 + 0.837527 w_k1_FDX_s005
                                 + 1.07851 w_k1_IBM_s005
                                 + 0.955161 w_k1_INTC_s005
                                 + 1.13946 w_k1_KO_s005 + 1.10545 w_k1_MAT_s005
                                 + 0.919814 w_k1_MSFT_s005
                                 + 0.900994 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.08312 w_k1_ADM_s006
                                 + 1.61721 w_k1_AXP_s006
                                 + 1.67311 w_k1_DIS_s006 + 1.4032 w_k1_FDX_s006
                                 + 1.23285 w_k1_IBM_s006
                                 + 1.20986 w_k1_INTC_s006
                                 + 1.21102 w_k1_KO_s006 + 1.53571 w_k1_MAT_s006
                                 + 1.49365 w_k1_MSFT_s006
                                 + 1.32858 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.842646 w_k1_ADM_s006
                                 + 0.772877 w_k1_AXP_s006
                                 + 0.860881 w_k1_DIS_s006
                                 + 0.897416 w_k1_FDX_s006
                                 + 0.715335 w_k1_IBM_s006
                                 + 0.894163 w_k1_INTC_s006
                                 + 0.865216 w_k1_KO_s006
                                 + 0.997808 w_k1_MAT_s006
                                 + 0.998605 w_k1_MSFT_s006
                                 + 0.997002 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.49854 w_k1_ADM_s006
                                 + 1.27024 w_k1_AXP_s006
                                 + 1.62675 w_k1_DIS_s006
                                 + 1.47022 w_k1_FDX_s006
                                 + 0.85043 w_k1_IBM_s006
                                 + 1.10461 w_k1_INTC_s006
                                 + 1.08957 w_k1_KO_s006 + 1.05273 w_k1_MAT_s006
                                 + 1.53186 w_k1_MSFT_s006
                                 + 1.13964 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.06777 w_k1_ADM_s006
                                 + 1.02695 w_k1_AXP_s006
                                 + 1.11792 w_k1_DIS_s006
                                 + 1.08124 w_k1_FDX_s006
                                 + 0.77297 w_k1_IBM_s006
                                 + 1.07772 w_k1_INTC_s006
                                 + 0.993643 w_k1_KO_s006
                                 + 1.27118 w_k1_MAT_s006
                                 + 1.15363 w_k1_MSFT_s006
                                 + 0.988427 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.872616 w_k1_ADM_s006
                                 + 1.12734 w_k1_AXP_s006 + 1.0033 w_k1_DIS_s006
                                 + 1.02712 w_k1_FDX_s006
                                 + 1.14247 w_k1_IBM_s006
                                 + 0.991134 w_k1_INTC_s006
                                 + 1.0029 w_k1_KO_s006 + 1.71222 w_k1_MAT_s006
                                 + 1.06675 w_k1_MSFT_s006
                                 + 1.01402 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.980713 w_k1_ADM_s006
                                 + 1.26958 w_k1_AXP_s006
                                 + 1.23998 w_k1_DIS_s006
                                 + 0.97781 w_k1_FDX_s006
                                 + 1.50216 w_k1_IBM_s006
                                 + 0.961401 w_k1_INTC_s006
                                 + 1.24742 w_k1_KO_s006 + 1.66416 w_k1_MAT_s006
                                 + 1.14097 w_k1_MSFT_s006
                                 + 1.00186 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.21975 w_k1_ADM_s006
                                 + 1.32762 w_k1_AXP_s006
                                 + 1.16433 w_k1_DIS_s006 + 1.1817 w_k1_FDX_s006
                                 + 1.04267 w_k1_IBM_s006
                                 + 1.08772 w_k1_INTC_s006
                                 + 1.04902 w_k1_KO_s006 + 1.12409 w_k1_MAT_s006
                                 + 1.06053 w_k1_MSFT_s006 + 1.117 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.707616 w_k1_ADM_s006
                                 + 0.723165 w_k1_AXP_s006
                                 + 0.696667 w_k1_DIS_s006
                                 + 0.727247 w_k1_FDX_s006
                                 + 0.965916 w_k1_IBM_s006
                                 + 0.739387 w_k1_INTC_s006
                                 + 0.8694 w_k1_KO_s006 + 1.26277 w_k1_MAT_s006
                                 + 0.849696 w_k1_MSFT_s006
                                 + 0.803538 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.34546 w_k1_ADM_s007
                                 + 1.33963 w_k1_AXP_s007
                                 + 1.48099 w_k1_DIS_s007
                                 + 1.29048 w_k1_FDX_s007
                                 + 1.49605 w_k1_IBM_s007
                                 + 1.11158 w_k1_INTC_s007
                                 + 1.31155 w_k1_KO_s007 + 1.08598 w_k1_MAT_s007
                                 + 1.22912 w_k1_MSFT_s007 + 1.2313 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.626615 w_k1_ADM_s007
                                 + 0.672882 w_k1_AXP_s007
                                 + 0.659669 w_k1_DIS_s007
                                 + 0.818473 w_k1_FDX_s007
                                 + 0.674092 w_k1_IBM_s007
                                 + 0.736269 w_k1_INTC_s007
                                 + 0.740058 w_k1_KO_s007
                                 + 0.594736 w_k1_MAT_s007
                                 + 0.686682 w_k1_MSFT_s007
                                 + 0.796228 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.887985 w_k1_ADM_s007
                                 + 0.741215 w_k1_AXP_s007
                                 + 0.658104 w_k1_DIS_s007
                                 + 0.700835 w_k1_FDX_s007
                                 + 0.945752 w_k1_IBM_s007
                                 + 0.688819 w_k1_INTC_s007
                                 + 0.824071 w_k1_KO_s007
                                 + 0.855905 w_k1_MAT_s007
                                 + 0.666164 w_k1_MSFT_s007
                                 + 0.940193 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.928761 w_k1_ADM_s007
                                 + 0.871826 w_k1_AXP_s007
                                 + 0.9624 w_k1_DIS_s007 + 1.01343 w_k1_FDX_s007
                                 + 0.60714 w_k1_IBM_s007
                                 + 0.84682 w_k1_INTC_s007
                                 + 0.726884 w_k1_KO_s007
                                 + 0.538338 w_k1_MAT_s007
                                 + 0.922464 w_k1_MSFT_s007
                                 + 0.869251 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.00335 w_k1_ADM_s007
                                 + 1.09817 w_k1_AXP_s007
                                 + 0.966333 w_k1_DIS_s007
                                 + 0.95697 w_k1_FDX_s007
                                 + 1.59954 w_k1_IBM_s007
                                 + 0.890837 w_k1_INTC_s007
                                 + 1.31279 w_k1_KO_s007 + 1.58665 w_k1_MAT_s007
                                 + 0.722888 w_k1_MSFT_s007
                                 + 1.20265 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.18507 w_k1_ADM_s007
                                 + 1.29063 w_k1_AXP_s007 + 1.3674 w_k1_DIS_s007
                                 + 1.24826 w_k1_FDX_s007
                                 + 1.06253 w_k1_IBM_s007
                                 + 1.00571 w_k1_INTC_s007
                                 + 1.10223 w_k1_KO_s007 + 1.06727 w_k1_MAT_s007
                                 + 1.14291 w_k1_MSFT_s007
                                 + 1.02827 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.859195 w_k1_ADM_s007
                                 + 0.756338 w_k1_AXP_s007
                                 + 0.955415 w_k1_DIS_s007
                                 + 0.93771 w_k1_FDX_s007
                                 + 0.804663 w_k1_IBM_s007
                                 + 0.877819 w_k1_INTC_s007
                                 + 0.879232 w_k1_KO_s007
                                 + 0.804777 w_k1_MAT_s007
                                 + 0.856292 w_k1_MSFT_s007
                                 + 1.04032 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.92218 w_k1_ADM_s007
                                 + 1.04351 w_k1_AXP_s007
                                 + 0.928945 w_k1_DIS_s007
                                 + 0.982486 w_k1_FDX_s007
                                 + 0.983284 w_k1_IBM_s007
                                 + 0.805525 w_k1_INTC_s007
                                 + 0.990012 w_k1_KO_s007
                                 + 1.12837 w_k1_MAT_s007
                                 + 0.886503 w_k1_MSFT_s007
                                 + 1.00807 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.66891 w_k1_ADM_s010
                                 + 1.4797 w_k1_AXP_s010 + 1.46153 w_k1_DIS_s010
                                 + 1.29486 w_k1_FDX_s010
                                 + 1.22546 w_k1_IBM_s010
                                 + 1.19986 w_k1_INTC_s010
                                 + 1.18908 w_k1_KO_s010 + 1.66467 w_k1_MAT_s010
                                 + 1.29651 w_k1_MSFT_s010 + 1.0824 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.80496 w_k1_ADM_s010
                                 + 1.59833 w_k1_AXP_s010
                                 + 1.39461 w_k1_DIS_s010
                                 + 1.43369 w_k1_FDX_s010
                                 + 0.930919 w_k1_IBM_s010
                                 + 1.12368 w_k1_INTC_s010
                                 + 1.13867 w_k1_KO_s010 + 1.23347 w_k1_MAT_s010
                                 + 1.61936 w_k1_MSFT_s010
                                 + 1.38585 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.46732 w_k1_ADM_s010
                                 + 1.18982 w_k1_AXP_s010
                                 + 1.14927 w_k1_DIS_s010
                                 + 1.09928 w_k1_FDX_s010
                                 + 1.10058 w_k1_IBM_s010
                                 + 1.08066 w_k1_INTC_s010
                                 + 1.07329 w_k1_KO_s010 + 1.06859 w_k1_MAT_s010
                                 + 1.22251 w_k1_MSFT_s010
                                 + 1.12468 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.88677 w_k1_ADM_s010
                                 + 1.25043 w_k1_AXP_s010
                                 + 1.33922 w_k1_DIS_s010
                                 + 1.07248 w_k1_FDX_s010
                                 + 1.62461 w_k1_IBM_s010
                                 + 1.02293 w_k1_INTC_s010
                                 + 1.31538 w_k1_KO_s010 + 1.70371 w_k1_MAT_s010
                                 + 1.41774 w_k1_MSFT_s010
                                 + 1.33881 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.01528 w_k1_ADM_s010
                                 + 0.708962 w_k1_AXP_s010
                                 + 0.752994 w_k1_DIS_s010
                                 + 0.810786 w_k1_FDX_s010
                                 + 0.967049 w_k1_IBM_s010
                                 + 1.05389 w_k1_INTC_s010
                                 + 0.839463 w_k1_KO_s010
                                 + 0.798629 w_k1_MAT_s010
                                 + 1.10265 w_k1_MSFT_s010
                                 + 0.930723 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.895812 w_k1_ADM_s010
                                 + 0.892447 w_k1_AXP_s010
                                 + 0.753968 w_k1_DIS_s010
                                 + 0.754632 w_k1_FDX_s010
                                 + 1.49927 w_k1_IBM_s010
                                 + 0.948701 w_k1_INTC_s010
                                 + 1.15931 w_k1_KO_s010 + 1.64014 w_k1_MAT_s010
                                 + 1.01489 w_k1_MSFT_s010
                                 + 1.11919 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.29665 w_k1_ADM_s010
                                 + 1.1786 w_k1_AXP_s010 + 1.00286 w_k1_DIS_s010
                                 + 1.17047 w_k1_FDX_s010
                                 + 0.729829 w_k1_IBM_s010
                                 + 1.11266 w_k1_INTC_s010
                                 + 0.859124 w_k1_KO_s010
                                 + 0.98688 w_k1_MAT_s010
                                 + 1.24648 w_k1_MSFT_s010
                                 + 0.951968 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.765732 w_k1_ADM_s010
                                 + 0.731531 w_k1_AXP_s010
                                 + 0.760924 w_k1_DIS_s010
                                 + 0.796425 w_k1_FDX_s010
                                 + 0.655252 w_k1_IBM_s010
                                 + 0.917618 w_k1_INTC_s010
                                 + 0.717713 w_k1_KO_s010
                                 + 0.479828 w_k1_MAT_s010
                                 + 0.952694 w_k1_MSFT_s010
                                 + 0.821528 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.956549 w_k1_ADM_s011
                                 + 0.838503 w_k1_AXP_s011
                                 + 0.774928 w_k1_DIS_s011
                                 + 0.863834 w_k1_FDX_s011
                                 + 0.966441 w_k1_IBM_s011
                                 + 0.83503 w_k1_INTC_s011
                                 + 1.01498 w_k1_KO_s011 + 1.04089 w_k1_MAT_s011
                                 + 0.767174 w_k1_MSFT_s011
                                 + 1.03659 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.71016 w_k1_ADM_s011
                                 + 1.53624 w_k1_AXP_s011
                                 + 1.67921 w_k1_DIS_s011
                                 + 1.36583 w_k1_FDX_s011
                                 + 1.37607 w_k1_IBM_s011
                                 + 1.34453 w_k1_INTC_s011
                                 + 1.62799 w_k1_KO_s011 + 1.67244 w_k1_MAT_s011
                                 + 1.33871 w_k1_MSFT_s011
                                 + 1.30924 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.19299 w_k1_ADM_s011
                                 + 1.1955 w_k1_AXP_s011 + 1.17709 w_k1_DIS_s011
                                 + 1.16584 w_k1_FDX_s011
                                 + 1.40391 w_k1_IBM_s011
                                 + 1.08674 w_k1_INTC_s011
                                 + 1.36338 w_k1_KO_s011 + 1.60538 w_k1_MAT_s011
                                 + 0.983935 w_k1_MSFT_s011
                                 + 1.23545 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.33782 w_k1_ADM_s011
                                 + 1.25477 w_k1_AXP_s011
                                 + 1.06653 w_k1_DIS_s011
                                 + 1.19123 w_k1_FDX_s011
                                 + 1.07865 w_k1_IBM_s011
                                 + 1.01167 w_k1_INTC_s011
                                 + 1.27044 w_k1_KO_s011 + 1.66331 w_k1_MAT_s011
                                 + 0.9738 w_k1_MSFT_s011 + 1.10654 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.10965 w_k1_ADM_s011
                                 + 0.860376 w_k1_AXP_s011
                                 + 1.12154 w_k1_DIS_s011
                                 + 1.03195 w_k1_FDX_s011
                                 + 0.800026 w_k1_IBM_s011
                                 + 1.18543 w_k1_INTC_s011
                                 + 0.899617 w_k1_KO_s011
                                 + 0.585631 w_k1_MAT_s011
                                 + 0.971228 w_k1_MSFT_s011
                                 + 0.99488 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.36278 w_k1_ADM_s011
                                 + 1.0413 w_k1_AXP_s011
                                 + 0.905587 w_k1_DIS_s011
                                 + 0.89168 w_k1_FDX_s011
                                 + 1.33774 w_k1_IBM_s011
                                 + 1.06097 w_k1_INTC_s011
                                 + 1.33313 w_k1_KO_s011 + 1.22238 w_k1_MAT_s011
                                 + 0.871464 w_k1_MSFT_s011
                                 + 1.12309 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.39614 w_k1_ADM_s011
                                 + 1.28075 w_k1_AXP_s011
                                 + 1.36649 w_k1_DIS_s011
                                 + 1.13503 w_k1_FDX_s011
                                 + 1.97765 w_k1_IBM_s011
                                 + 0.849005 w_k1_INTC_s011
                                 + 1.50883 w_k1_KO_s011 + 2.32917 w_k1_MAT_s011
                                 + 0.837994 w_k1_MSFT_s011
                                 + 1.24391 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.894383 w_k1_ADM_s011
                                 + 0.713709 w_k1_AXP_s011
                                 + 0.613188 w_k1_DIS_s011
                                 + 0.85631 w_k1_FDX_s011
                                 + 0.683902 w_k1_IBM_s011
                                 + 0.924324 w_k1_INTC_s011
                                 + 0.83472 w_k1_KO_s011
                                 + 0.820427 w_k1_MAT_s011
                                 + 0.776073 w_k1_MSFT_s011
                                 + 0.919796 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.979005 w_k1_ADM_s012
                                 + 0.735046 w_k1_AXP_s012
                                 + 1.00613 w_k1_DIS_s012
                                 + 0.898499 w_k1_FDX_s012
                                 + 0.869569 w_k1_IBM_s012
                                 + 0.932943 w_k1_INTC_s012
                                 + 1.21456 w_k1_KO_s012 + 1.10179 w_k1_MAT_s012
                                 + 1.13806 w_k1_MSFT_s012
                                 + 1.13575 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.5368 w_k1_ADM_s012
                                 + 1.52592 w_k1_AXP_s012
                                 + 1.51422 w_k1_DIS_s012
                                 + 1.37625 w_k1_FDX_s012
                                 + 1.39821 w_k1_IBM_s012
                                 + 1.17158 w_k1_INTC_s012
                                 + 1.32435 w_k1_KO_s012 + 1.60101 w_k1_MAT_s012
                                 + 1.23744 w_k1_MSFT_s012
                                 + 1.21457 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.987144 w_k1_ADM_s012
                                 + 1.2896 w_k1_AXP_s012 + 1.35022 w_k1_DIS_s012
                                 + 1.07245 w_k1_FDX_s012
                                 + 1.38007 w_k1_IBM_s012
                                 + 1.3497 w_k1_INTC_s012 + 1.61534 w_k1_KO_s012
                                 + 1.71093 w_k1_MAT_s012
                                 + 1.27559 w_k1_MSFT_s012
                                 + 1.36167 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.999615 w_k1_ADM_s012
                                 + 1.07533 w_k1_AXP_s012
                                 + 1.04262 w_k1_DIS_s012
                                 + 1.03284 w_k1_FDX_s012
                                 + 1.19669 w_k1_IBM_s012
                                 + 1.0085 w_k1_INTC_s012 + 1.19949 w_k1_KO_s012
                                 + 1.43848 w_k1_MAT_s012
                                 + 1.15572 w_k1_MSFT_s012
                                 + 1.19854 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.986613 w_k1_ADM_s012
                                 + 0.891806 w_k1_AXP_s012
                                 + 0.685974 w_k1_DIS_s012
                                 + 0.89354 w_k1_FDX_s012
                                 + 0.769917 w_k1_IBM_s012
                                 + 0.959351 w_k1_INTC_s012
                                 + 0.782587 w_k1_KO_s012
                                 + 0.774024 w_k1_MAT_s012
                                 + 0.947202 w_k1_MSFT_s012
                                 + 0.921906 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.643766 w_k1_ADM_s012
                                 + 0.872273 w_k1_AXP_s012
                                 + 0.57231 w_k1_DIS_s012
                                 + 0.747235 w_k1_FDX_s012
                                 + 0.821137 w_k1_IBM_s012
                                 + 0.849114 w_k1_INTC_s012
                                 + 0.716948 w_k1_KO_s012
                                 + 1.26478 w_k1_MAT_s012
                                 + 0.778964 w_k1_MSFT_s012
                                 + 0.919978 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.958518 w_k1_ADM_s012
                                 + 1.06442 w_k1_AXP_s012 + 1.3632 w_k1_DIS_s012
                                 + 1.30179 w_k1_FDX_s012
                                 + 0.757006 w_k1_IBM_s012
                                 + 1.15647 w_k1_INTC_s012
                                 + 1.12942 w_k1_KO_s012
                                 + 0.872936 w_k1_MAT_s012
                                 + 1.33519 w_k1_MSFT_s012
                                 + 1.34452 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.09276 w_k1_ADM_s012
                                 + 0.828191 w_k1_AXP_s012
                                 + 0.876969 w_k1_DIS_s012
                                 + 0.988464 w_k1_FDX_s012
                                 + 0.730934 w_k1_IBM_s012
                                 + 1.24528 w_k1_INTC_s012 + 1.022 w_k1_KO_s012
                                 + 0.794564 w_k1_MAT_s012
                                 + 1.18068 w_k1_MSFT_s012
                                 + 1.02143 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.81504 w_k1_ADM_s013
                                 + 1.62093 w_k1_AXP_s013
                                 + 1.52549 w_k1_DIS_s013
                                 + 1.34911 w_k1_FDX_s013
                                 + 1.29655 w_k1_IBM_s013
                                 + 1.60018 w_k1_INTC_s013
                                 + 1.36205 w_k1_KO_s013 + 1.13134 w_k1_MAT_s013
                                 + 1.38607 w_k1_MSFT_s013
                                 + 1.16092 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.25893 w_k1_ADM_s013
                                 + 0.919386 w_k1_AXP_s013
                                 + 1.21257 w_k1_DIS_s013
                                 + 0.988233 w_k1_FDX_s013
                                 + 1.10227 w_k1_IBM_s013
                                 + 1.10526 w_k1_INTC_s013
                                 + 1.38794 w_k1_KO_s013
                                 + 0.859419 w_k1_MAT_s013
                                 + 1.01164 w_k1_MSFT_s013 + 1.0471 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.12594 w_k1_ADM_s013
                                 + 0.741314 w_k1_AXP_s013
                                 + 0.730893 w_k1_DIS_s013
                                 + 0.823775 w_k1_FDX_s013
                                 + 0.723967 w_k1_IBM_s013
                                 + 0.839516 w_k1_INTC_s013
                                 + 0.762015 w_k1_KO_s013
                                 + 0.540705 w_k1_MAT_s013
                                 + 0.687375 w_k1_MSFT_s013
                                 + 0.848299 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.39891 w_k1_ADM_s013
                                 + 1.23539 w_k1_AXP_s013
                                 + 0.851858 w_k1_DIS_s013
                                 + 0.932674 w_k1_FDX_s013
                                 + 1.07662 w_k1_IBM_s013
                                 + 1.13297 w_k1_INTC_s013
                                 + 1.16015 w_k1_KO_s013
                                 + 0.854546 w_k1_MAT_s013
                                 + 0.866349 w_k1_MSFT_s013
                                 + 1.04661 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.08555 w_k1_ADM_s013
                                 + 0.917156 w_k1_AXP_s013
                                 + 0.866706 w_k1_DIS_s013
                                 + 0.840316 w_k1_FDX_s013
                                 + 1.14696 w_k1_IBM_s013
                                 + 1.18264 w_k1_INTC_s013
                                 + 1.12375 w_k1_KO_s013
                                 + 0.941455 w_k1_MAT_s013
                                 + 0.845961 w_k1_MSFT_s013
                                 + 0.966442 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.63754 w_k1_ADM_s013
                                 + 1.27679 w_k1_AXP_s013
                                 + 1.20359 w_k1_DIS_s013
                                 + 1.25135 w_k1_FDX_s013
                                 + 0.873313 w_k1_IBM_s013
                                 + 1.48464 w_k1_INTC_s013 + 1.1743 w_k1_KO_s013
                                 + 0.857134 w_k1_MAT_s013
                                 + 1.26232 w_k1_MSFT_s013
                                 + 1.02408 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.945863 w_k1_ADM_s013
                                 + 0.691794 w_k1_AXP_s013
                                 + 0.792103 w_k1_DIS_s013
                                 + 0.705034 w_k1_FDX_s013
                                 + 0.910021 w_k1_IBM_s013
                                 + 1.03841 w_k1_INTC_s013
                                 + 0.956953 w_k1_KO_s013
                                 + 0.636287 w_k1_MAT_s013
                                 + 0.844387 w_k1_MSFT_s013
                                 + 0.841079 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.51239 w_k1_ADM_s013
                                 + 0.981326 w_k1_AXP_s013
                                 + 0.974062 w_k1_DIS_s013
                                 + 1.06491 w_k1_FDX_s013
                                 + 0.597518 w_k1_IBM_s013
                                 + 1.11741 w_k1_INTC_s013
                                 + 0.984479 w_k1_KO_s013
                                 + 0.508106 w_k1_MAT_s013
                                 + 0.98616 w_k1_MSFT_s013
                                 + 1.02637 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.05392 w_k1_ADM_s014
                                 + 0.850464 w_k1_AXP_s014
                                 + 1.01717 w_k1_DIS_s014
                                 + 0.986656 w_k1_FDX_s014
                                 + 1.13082 w_k1_IBM_s014
                                 + 0.981086 w_k1_INTC_s014
                                 + 0.90757 w_k1_KO_s014
                                 + 0.873559 w_k1_MAT_s014
                                 + 0.97649 w_k1_MSFT_s014
                                 + 0.911295 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.00511 w_k1_ADM_s014
                                 + 0.608562 w_k1_AXP_s014
                                 + 1.06814 w_k1_DIS_s014
                                 + 1.08065 w_k1_FDX_s014
                                 + 0.699433 w_k1_IBM_s014
                                 + 1.26336 w_k1_INTC_s014
                                 + 0.936007 w_k1_KO_s014
                                 + 0.582621 w_k1_MAT_s014
                                 + 1.26027 w_k1_MSFT_s014
                                 + 0.912054 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.940668 w_k1_ADM_s014
                                 + 0.654953 w_k1_AXP_s014
                                 + 0.600197 w_k1_DIS_s014
                                 + 0.850435 w_k1_FDX_s014
                                 + 0.686818 w_k1_IBM_s014
                                 + 0.819766 w_k1_INTC_s014
                                 + 0.603291 w_k1_KO_s014
                                 + 0.539332 w_k1_MAT_s014
                                 + 0.756636 w_k1_MSFT_s014
                                 + 0.779239 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.868537 w_k1_ADM_s014
                                 + 0.626159 w_k1_AXP_s014
                                 + 0.889307 w_k1_DIS_s014
                                 + 0.805456 w_k1_FDX_s014
                                 + 0.98191 w_k1_IBM_s014
                                 + 1.09359 w_k1_INTC_s014
                                 + 0.954317 w_k1_KO_s014
                                 + 0.764506 w_k1_MAT_s014
                                 + 0.863186 w_k1_MSFT_s014
                                 + 0.942035 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.845536 w_k1_ADM_s014
                                 + 0.519367 w_k1_AXP_s014
                                 + 0.636824 w_k1_DIS_s014
                                 + 0.683348 w_k1_FDX_s014
                                 + 0.869611 w_k1_IBM_s014
                                 + 0.775961 w_k1_INTC_s014
                                 + 0.717877 w_k1_KO_s014
                                 + 0.64816 w_k1_MAT_s014
                                 + 0.711628 w_k1_MSFT_s014
                                 + 0.838788 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.07961 w_k1_ADM_s014
                                 + 1.02528 w_k1_AXP_s014
                                 + 0.750422 w_k1_DIS_s014
                                 + 0.970973 w_k1_FDX_s014
                                 + 0.833841 w_k1_IBM_s014
                                 + 0.865386 w_k1_INTC_s014
                                 + 0.757529 w_k1_KO_s014
                                 + 1.02812 w_k1_MAT_s014
                                 + 0.812859 w_k1_MSFT_s014
                                 + 0.916031 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.09856 w_k1_ADM_s014
                                 + 1.27884 w_k1_AXP_s014 + 1.2456 w_k1_DIS_s014
                                 + 1.21085 w_k1_FDX_s014 + 1.2592 w_k1_IBM_s014
                                 + 1.34624 w_k1_INTC_s014
                                 + 1.11507 w_k1_KO_s014 + 1.22043 w_k1_MAT_s014
                                 + 1.06844 w_k1_MSFT_s014
                                 + 0.999174 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.3664 w_k1_ADM_s014
                                 + 1.5247 w_k1_AXP_s014 + 1.78169 w_k1_DIS_s014
                                 + 1.46034 w_k1_FDX_s014
                                 + 1.37684 w_k1_IBM_s014
                                 + 1.47729 w_k1_INTC_s014
                                 + 1.20058 w_k1_KO_s014 + 1.54039 w_k1_MAT_s014
                                 + 1.32809 w_k1_MSFT_s014
                                 + 1.08342 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.00195 w_k1_ADM_s015
                                 + 0.735044 w_k1_AXP_s015
                                 + 0.713396 w_k1_DIS_s015
                                 + 0.839349 w_k1_FDX_s015
                                 + 0.750905 w_k1_IBM_s015
                                 + 0.722453 w_k1_INTC_s015
                                 + 0.733706 w_k1_KO_s015
                                 + 0.549479 w_k1_MAT_s015
                                 + 0.784894 w_k1_MSFT_s015
                                 + 0.893674 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.204 w_k1_ADM_s015
                                 + 0.952927 w_k1_AXP_s015
                                 + 1.02006 w_k1_DIS_s015
                                 + 1.05193 w_k1_FDX_s015
                                 + 1.01454 w_k1_IBM_s015
                                 + 1.1541 w_k1_INTC_s015 + 1.0504 w_k1_KO_s015
                                 + 1.13308 w_k1_MAT_s015
                                 + 1.17118 w_k1_MSFT_s015
                                 + 0.965028 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.34097 w_k1_ADM_s015
                                 + 0.857987 w_k1_AXP_s015
                                 + 0.822716 w_k1_DIS_s015
                                 + 1.11147 w_k1_FDX_s015
                                 + 0.687093 w_k1_IBM_s015
                                 + 0.947712 w_k1_INTC_s015
                                 + 0.929108 w_k1_KO_s015
                                 + 0.537864 w_k1_MAT_s015
                                 + 1.11763 w_k1_MSFT_s015
                                 + 1.06627 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.69962 w_k1_ADM_s015
                                 + 1.11783 w_k1_AXP_s015
                                 + 1.35036 w_k1_DIS_s015
                                 + 0.987454 w_k1_FDX_s015
                                 + 1.7437 w_k1_IBM_s015
                                 + 1.28646 w_k1_INTC_s015
                                 + 1.42436 w_k1_KO_s015 + 1.14733 w_k1_MAT_s015
                                 + 1.26083 w_k1_MSFT_s015
                                 + 1.07162 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.24038 w_k1_ADM_s015
                                 + 1.15569 w_k1_AXP_s015
                                 + 1.17684 w_k1_DIS_s015
                                 + 1.03595 w_k1_FDX_s015 + 1.3215 w_k1_IBM_s015
                                 + 0.796106 w_k1_INTC_s015
                                 + 1.07972 w_k1_KO_s015 + 1.36889 w_k1_MAT_s015
                                 + 1.00506 w_k1_MSFT_s015
                                 + 0.968445 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.13209 w_k1_ADM_s015
                                 + 1.47354 w_k1_AXP_s015
                                 + 0.880447 w_k1_DIS_s015
                                 + 1.08563 w_k1_FDX_s015
                                 + 0.914882 w_k1_IBM_s015
                                 + 1.02158 w_k1_INTC_s015
                                 + 0.894351 w_k1_KO_s015
                                 + 1.14638 w_k1_MAT_s015
                                 + 0.9613 w_k1_MSFT_s015
                                 + 0.879256 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.966945 w_k1_ADM_s015
                                 + 1.12932 w_k1_AXP_s015
                                 + 1.15109 w_k1_DIS_s015 + 1.1535 w_k1_FDX_s015
                                 + 0.725081 w_k1_IBM_s015
                                 + 0.903797 w_k1_INTC_s015
                                 + 0.873875 w_k1_KO_s015
                                 + 0.834002 w_k1_MAT_s015
                                 + 1.11751 w_k1_MSFT_s015
                                 + 0.905798 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.15918 w_k1_ADM_s015
                                 + 1.31862 w_k1_AXP_s015
                                 + 1.31507 w_k1_DIS_s015
                                 + 1.29539 w_k1_FDX_s015
                                 + 0.975868 w_k1_IBM_s015
                                 + 1.27862 w_k1_INTC_s015
                                 + 1.34682 w_k1_KO_s015 + 1.05844 w_k1_MAT_s015
                                 + 1.46867 w_k1_MSFT_s015
                                 + 1.12835 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.2421 w_k1_ADM_s016
                                 + 1.04759 w_k1_AXP_s016
                                 + 1.31215 w_k1_DIS_s016 + 1.0401 w_k1_FDX_s016
                                 + 1.50631 w_k1_IBM_s016
                                 + 1.17621 w_k1_INTC_s016
                                 + 1.25362 w_k1_KO_s016 + 1.38416 w_k1_MAT_s016
                                 + 1.19375 w_k1_MSFT_s016
                                 + 1.14912 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.801215 w_k1_ADM_s016
                                 + 1.4783 w_k1_AXP_s016 + 1.02721 w_k1_DIS_s016
                                 + 1.05278 w_k1_FDX_s016
                                 + 1.14199 w_k1_IBM_s016
                                 + 1.01483 w_k1_INTC_s016
                                 + 0.966935 w_k1_KO_s016
                                 + 1.42574 w_k1_MAT_s016
                                 + 1.05906 w_k1_MSFT_s016
                                 + 1.28905 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.914341 w_k1_ADM_s016
                                 + 0.770858 w_k1_AXP_s016
                                 + 0.72343 w_k1_DIS_s016
                                 + 0.864804 w_k1_FDX_s016
                                 + 0.687252 w_k1_IBM_s016
                                 + 0.752223 w_k1_INTC_s016
                                 + 0.691767 w_k1_KO_s016
                                 + 0.482316 w_k1_MAT_s016
                                 + 0.769556 w_k1_MSFT_s016
                                 + 0.881275 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.741899 w_k1_ADM_s016
                                 + 0.885155 w_k1_AXP_s016
                                 + 0.832731 w_k1_DIS_s016
                                 + 0.705465 w_k1_FDX_s016
                                 + 1.15595 w_k1_IBM_s016
                                 + 1.04757 w_k1_INTC_s016
                                 + 0.932109 w_k1_KO_s016
                                 + 0.809926 w_k1_MAT_s016
                                 + 0.835345 w_k1_MSFT_s016
                                 + 0.976876 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.641408 w_k1_ADM_s016
                                 + 1.02866 w_k1_AXP_s016
                                 + 0.766881 w_k1_DIS_s016
                                 + 0.791438 w_k1_FDX_s016
                                 + 0.838223 w_k1_IBM_s016
                                 + 1.07042 w_k1_INTC_s016
                                 + 0.69129 w_k1_KO_s016
                                 + 0.723625 w_k1_MAT_s016
                                 + 0.839259 w_k1_MSFT_s016
                                 + 0.79567 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.07913 w_k1_ADM_s016
                                 + 1.12888 w_k1_AXP_s016
                                 + 1.08478 w_k1_DIS_s016
                                 + 1.00485 w_k1_FDX_s016
                                 + 0.867288 w_k1_IBM_s016
                                 + 0.953315 w_k1_INTC_s016
                                 + 0.870027 w_k1_KO_s016
                                 + 0.806363 w_k1_MAT_s016
                                 + 1.07052 w_k1_MSFT_s016
                                 + 1.03186 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.70495 w_k1_ADM_s016
                                 + 1.13854 w_k1_AXP_s016
                                 + 1.50152 w_k1_DIS_s016 + 1.3053 w_k1_FDX_s016
                                 + 0.9869 w_k1_IBM_s016
                                 + 1.16904 w_k1_INTC_s016
                                 + 1.08122 w_k1_KO_s016 + 0.88627 w_k1_MAT_s016
                                 + 1.18858 w_k1_MSFT_s016
                                 + 1.15818 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.845339 w_k1_ADM_s016
                                 + 0.836993 w_k1_AXP_s016
                                 + 0.910612 w_k1_DIS_s016
                                 + 0.853054 w_k1_FDX_s016
                                 + 0.904078 w_k1_IBM_s016
                                 + 0.950912 w_k1_INTC_s016
                                 + 0.851898 w_k1_KO_s016
                                 + 0.751769 w_k1_MAT_s016
                                 + 0.89079 w_k1_MSFT_s016
                                 + 0.870891 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.91498 w_k1_ADM_s017
                                 + 1.43838 w_k1_AXP_s017
                                 + 1.52995 w_k1_DIS_s017
                                 + 1.41057 w_k1_FDX_s017
                                 + 1.32926 w_k1_IBM_s017
                                 + 1.0262 w_k1_INTC_s017 + 1.30739 w_k1_KO_s017
                                 + 1.44616 w_k1_MAT_s017
                                 + 1.05992 w_k1_MSFT_s017
                                 + 1.23198 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.44112 w_k1_ADM_s017
                                 + 1.21369 w_k1_AXP_s017 + 1.394 w_k1_DIS_s017
                                 + 1.30894 w_k1_FDX_s017
                                 + 0.817947 w_k1_IBM_s017
                                 + 1.3331 w_k1_INTC_s017 + 1.02308 w_k1_KO_s017
                                 + 0.673353 w_k1_MAT_s017
                                 + 1.02853 w_k1_MSFT_s017
                                 + 0.953317 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.13336 w_k1_ADM_s017
                                 + 0.955197 w_k1_AXP_s017
                                 + 0.771049 w_k1_DIS_s017
                                 + 0.913541 w_k1_FDX_s017
                                 + 0.581648 w_k1_IBM_s017
                                 + 0.888658 w_k1_INTC_s017
                                 + 0.702443 w_k1_KO_s017
                                 + 0.725986 w_k1_MAT_s017
                                 + 0.703223 w_k1_MSFT_s017
                                 + 0.701755 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.21344 w_k1_ADM_s017
                                 + 1.06686 w_k1_AXP_s017
                                 + 1.06417 w_k1_DIS_s017
                                 + 0.982666 w_k1_FDX_s017
                                 + 1.46526 w_k1_IBM_s017
                                 + 0.913651 w_k1_INTC_s017
                                 + 1.29154 w_k1_KO_s017 + 1.45897 w_k1_MAT_s017
                                 + 0.824729 w_k1_MSFT_s017
                                 + 0.895077 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.14748 w_k1_ADM_s017
                                 + 0.75888 w_k1_AXP_s017
                                 + 1.17461 w_k1_DIS_s017
                                 + 1.09519 w_k1_FDX_s017
                                 + 0.745064 w_k1_IBM_s017
                                 + 1.05477 w_k1_INTC_s017
                                 + 0.951595 w_k1_KO_s017
                                 + 0.484759 w_k1_MAT_s017
                                 + 0.887658 w_k1_MSFT_s017
                                 + 0.876497 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.879882 w_k1_ADM_s017
                                 + 0.897235 w_k1_AXP_s017
                                 + 1.10599 w_k1_DIS_s017
                                 + 0.985243 w_k1_FDX_s017
                                 + 1.12727 w_k1_IBM_s017
                                 + 0.86851 w_k1_INTC_s017
                                 + 0.985463 w_k1_KO_s017
                                 + 1.25717 w_k1_MAT_s017
                                 + 0.977489 w_k1_MSFT_s017
                                 + 0.874335 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.03499 w_k1_ADM_s017
                                 + 1.10098 w_k1_AXP_s017
                                 + 0.73043 w_k1_DIS_s017
                                 + 0.885773 w_k1_FDX_s017
                                 + 0.897251 w_k1_IBM_s017
                                 + 0.819269 w_k1_INTC_s017
                                 + 0.8501 w_k1_KO_s017 + 0.792654 w_k1_MAT_s017
                                 + 0.845807 w_k1_MSFT_s017
                                 + 0.795846 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.05681 w_k1_ADM_s017
                                 + 0.712717 w_k1_AXP_s017 + 0.714 w_k1_DIS_s017
                                 + 0.754943 w_k1_FDX_s017
                                 + 0.767727 w_k1_IBM_s017
                                 + 0.843675 w_k1_INTC_s017
                                 + 0.739668 w_k1_KO_s017
                                 + 0.64441 w_k1_MAT_s017
                                 + 0.760433 w_k1_MSFT_s017
                                 + 0.704806 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.620429 w_k1_ADM_s100
                                 + 0.823053 w_k1_AXP_s100
                                 + 0.731878 w_k1_DIS_s100
                                 + 0.728847 w_k1_FDX_s100
                                 + 0.978103 w_k1_IBM_s100
                                 + 0.880772 w_k1_INTC_s100
                                 + 0.881869 w_k1_KO_s100
                                 + 1.05189 w_k1_MAT_s100
                                 + 0.769257 w_k1_MSFT_s100
                                 + 0.871092 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.09264 w_k1_ADM_s100
                                 + 1.41127 w_k1_AXP_s100
                                 + 0.856614 w_k1_DIS_s100
                                 + 1.11522 w_k1_FDX_s100
                                 + 0.882045 w_k1_IBM_s100
                                 + 1.0883 w_k1_INTC_s100
                                 + 0.890162 w_k1_KO_s100
                                 + 1.01766 w_k1_MAT_s100
                                 + 0.959162 w_k1_MSFT_s100 + 1.03 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.01753 w_k1_ADM_s100
                                 + 1.58846 w_k1_AXP_s100
                                 + 1.65687 w_k1_DIS_s100
                                 + 1.30038 w_k1_FDX_s100
                                 + 1.40309 w_k1_IBM_s100
                                 + 1.19023 w_k1_INTC_s100
                                 + 1.21982 w_k1_KO_s100 + 1.64787 w_k1_MAT_s100
                                 + 1.31728 w_k1_MSFT_s100
                                 + 1.15649 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.810341 w_k1_ADM_s100
                                 + 1.0929 w_k1_AXP_s100 + 1.00315 w_k1_DIS_s100
                                 + 0.961734 w_k1_FDX_s100
                                 + 1.02029 w_k1_IBM_s100
                                 + 1.12444 w_k1_INTC_s100
                                 + 0.940945 w_k1_KO_s100
                                 + 1.34751 w_k1_MAT_s100
                                 + 0.968309 w_k1_MSFT_s100
                                 + 1.00366 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.802271 w_k1_ADM_s100
                                 + 0.870451 w_k1_AXP_s100
                                 + 0.954777 w_k1_DIS_s100
                                 + 0.995204 w_k1_FDX_s100
                                 + 0.766934 w_k1_IBM_s100
                                 + 1.13801 w_k1_INTC_s100
                                 + 0.857506 w_k1_KO_s100
                                 + 0.876208 w_k1_MAT_s100
                                 + 1.08289 w_k1_MSFT_s100
                                 + 0.848047 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.653438 w_k1_ADM_s100
                                 + 0.696144 w_k1_AXP_s100
                                 + 0.53607 w_k1_DIS_s100
                                 + 0.732963 w_k1_FDX_s100
                                 + 0.669205 w_k1_IBM_s100
                                 + 0.925315 w_k1_INTC_s100
                                 + 0.666466 w_k1_KO_s100
                                 + 0.574575 w_k1_MAT_s100
                                 + 0.759277 w_k1_MSFT_s100
                                 + 0.763964 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.16597 w_k1_ADM_s100
                                 + 1.11736 w_k1_AXP_s100
                                 + 1.09054 w_k1_DIS_s100 + 1.1869 w_k1_FDX_s100
                                 + 1.12853 w_k1_IBM_s100
                                 + 1.00604 w_k1_INTC_s100
                                 + 1.08228 w_k1_KO_s100
                                 + 0.984895 w_k1_MAT_s100
                                 + 1.17898 w_k1_MSFT_s100
                                 + 1.09683 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.969716 w_k1_ADM_s100
                                 + 0.851437 w_k1_AXP_s100
                                 + 0.901368 w_k1_DIS_s100
                                 + 0.831202 w_k1_FDX_s100
                                 + 1.29527 w_k1_IBM_s100
                                 + 0.912596 w_k1_INTC_s100
                                 + 1.13405 w_k1_KO_s100 + 1.18655 w_k1_MAT_s100
                                 + 0.970344 w_k1_MSFT_s100
                                 + 1.05753 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.897351 w_k1_ADM_s101
                                 + 1.01199 w_k1_AXP_s101
                                 + 0.918582 w_k1_DIS_s101
                                 + 1.0561 w_k1_FDX_s101 + 0.91269 w_k1_IBM_s101
                                 + 1.07345 w_k1_INTC_s101
                                 + 0.923995 w_k1_KO_s101
                                 + 0.725709 w_k1_MAT_s101
                                 + 1.14252 w_k1_MSFT_s101
                                 + 0.927193 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.772002 w_k1_ADM_s101
                                 + 0.900164 w_k1_AXP_s101
                                 + 0.745672 w_k1_DIS_s101
                                 + 0.823749 w_k1_FDX_s101
                                 + 0.804817 w_k1_IBM_s101
                                 + 0.79749 w_k1_INTC_s101
                                 + 0.964149 w_k1_KO_s101
                                 + 0.896319 w_k1_MAT_s101
                                 + 0.858657 w_k1_MSFT_s101
                                 + 0.814176 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.26677 w_k1_ADM_s101
                                 + 1.26555 w_k1_AXP_s101
                                 + 1.42501 w_k1_DIS_s101
                                 + 1.27529 w_k1_FDX_s101
                                 + 0.994378 w_k1_IBM_s101
                                 + 1.35729 w_k1_INTC_s101
                                 + 1.16383 w_k1_KO_s101 + 1.41769 w_k1_MAT_s101
                                 + 1.41297 w_k1_MSFT_s101
                                 + 1.09033 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.839707 w_k1_ADM_s101
                                 + 0.867588 w_k1_AXP_s101
                                 + 0.837042 w_k1_DIS_s101
                                 + 0.810457 w_k1_FDX_s101
                                 + 1.16375 w_k1_IBM_s101
                                 + 0.906074 w_k1_INTC_s101
                                 + 1.1292 w_k1_KO_s101 + 1.39668 w_k1_MAT_s101
                                 + 0.871008 w_k1_MSFT_s101
                                 + 0.983674 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.978597 w_k1_ADM_s101
                                 + 1.23738 w_k1_AXP_s101
                                 + 1.27223 w_k1_DIS_s101
                                 + 1.11487 w_k1_FDX_s101
                                 + 1.47965 w_k1_IBM_s101
                                 + 1.16027 w_k1_INTC_s101
                                 + 1.43507 w_k1_KO_s101 + 1.98336 w_k1_MAT_s101
                                 + 1.1658 w_k1_MSFT_s101 + 1.10091 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.27718 w_k1_ADM_s101
                                 + 1.77112 w_k1_AXP_s101
                                 + 2.03064 w_k1_DIS_s101
                                 + 1.62829 w_k1_FDX_s101
                                 + 1.38139 w_k1_IBM_s101
                                 + 1.76318 w_k1_INTC_s101
                                 + 1.71337 w_k1_KO_s101 + 1.82997 w_k1_MAT_s101
                                 + 1.67006 w_k1_MSFT_s101
                                 + 1.20149 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.09525 w_k1_ADM_s101
                                 + 1.2221 w_k1_AXP_s101 + 1.02993 w_k1_DIS_s101
                                 + 0.932194 w_k1_FDX_s101
                                 + 1.33712 w_k1_IBM_s101
                                 + 1.16846 w_k1_INTC_s101
                                 + 1.30453 w_k1_KO_s101 + 1.90922 w_k1_MAT_s101
                                 + 1.07351 w_k1_MSFT_s101
                                 + 1.00577 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.684541 w_k1_ADM_s101
                                 + 0.586287 w_k1_AXP_s101
                                 + 0.667582 w_k1_DIS_s101
                                 + 0.774007 w_k1_FDX_s101
                                 + 0.666244 w_k1_IBM_s101
                                 + 0.857095 w_k1_INTC_s101
                                 + 0.784393 w_k1_KO_s101
                                 + 0.763187 w_k1_MAT_s101
                                 + 0.865941 w_k1_MSFT_s101
                                 + 0.759411 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.918572 w_k1_ADM_s102
                                 + 0.911418 w_k1_AXP_s102
                                 + 0.937854 w_k1_DIS_s102
                                 + 1.00474 w_k1_FDX_s102
                                 + 0.710486 w_k1_IBM_s102
                                 + 0.929817 w_k1_INTC_s102
                                 + 0.775566 w_k1_KO_s102
                                 + 0.537528 w_k1_MAT_s102
                                 + 0.883116 w_k1_MSFT_s102
                                 + 0.890142 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.642681 w_k1_ADM_s102
                                 + 0.727054 w_k1_AXP_s102
                                 + 0.515767 w_k1_DIS_s102
                                 + 0.676132 w_k1_FDX_s102
                                 + 0.756208 w_k1_IBM_s102
                                 + 0.68115 w_k1_INTC_s102
                                 + 0.668306 w_k1_KO_s102
                                 + 0.556446 w_k1_MAT_s102
                                 + 0.684735 w_k1_MSFT_s102
                                 + 0.760209 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.896201 w_k1_ADM_s102
                                 + 0.973456 w_k1_AXP_s102
                                 + 1.14103 w_k1_DIS_s102
                                 + 1.09086 w_k1_FDX_s102
                                 + 1.19071 w_k1_IBM_s102
                                 + 0.999963 w_k1_INTC_s102
                                 + 1.14245 w_k1_KO_s102 + 1.21224 w_k1_MAT_s102
                                 + 1.01344 w_k1_MSFT_s102
                                 + 1.03919 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.967468 w_k1_ADM_s102
                                 + 1.09361 w_k1_AXP_s102
                                 + 0.986764 w_k1_DIS_s102
                                 + 0.888968 w_k1_FDX_s102
                                 + 1.38756 w_k1_IBM_s102
                                 + 0.960993 w_k1_INTC_s102
                                 + 1.04997 w_k1_KO_s102 + 1.37239 w_k1_MAT_s102
                                 + 0.88095 w_k1_MSFT_s102
                                 + 0.947093 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.833758 w_k1_ADM_s102
                                 + 0.815774 w_k1_AXP_s102
                                 + 0.852991 w_k1_DIS_s102
                                 + 0.79671 w_k1_FDX_s102
                                 + 1.04309 w_k1_IBM_s102
                                 + 0.804044 w_k1_INTC_s102
                                 + 0.951643 w_k1_KO_s102
                                 + 0.873399 w_k1_MAT_s102
                                 + 0.830961 w_k1_MSFT_s102
                                 + 0.891999 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.09799 w_k1_ADM_s102
                                 + 1.01579 w_k1_AXP_s102
                                 + 0.873567 w_k1_DIS_s102
                                 + 1.10094 w_k1_FDX_s102
                                 + 0.959936 w_k1_IBM_s102
                                 + 0.939507 w_k1_INTC_s102
                                 + 0.890414 w_k1_KO_s102
                                 + 1.03897 w_k1_MAT_s102
                                 + 0.968844 w_k1_MSFT_s102
                                 + 1.07509 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.17894 w_k1_ADM_s102
                                 + 1.21543 w_k1_AXP_s102
                                 + 1.05083 w_k1_DIS_s102
                                 + 1.26453 w_k1_FDX_s102
                                 + 0.689611 w_k1_IBM_s102
                                 + 1.16416 w_k1_INTC_s102
                                 + 0.890146 w_k1_KO_s102
                                 + 0.585962 w_k1_MAT_s102
                                 + 1.0553 w_k1_MSFT_s102 + 1.02294 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.27317 w_k1_ADM_s102
                                 + 1.55321 w_k1_AXP_s102
                                 + 1.67335 w_k1_DIS_s102
                                 + 1.39049 w_k1_FDX_s102
                                 + 1.31138 w_k1_IBM_s102
                                 + 1.31821 w_k1_INTC_s102
                                 + 1.37101 w_k1_KO_s102 + 1.35581 w_k1_MAT_s102
                                 + 1.1863 w_k1_MSFT_s102 + 1.30742 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.813719 w_k1_ADM_s103
                                 + 0.767791 w_k1_AXP_s103
                                 + 0.689028 w_k1_DIS_s103
                                 + 0.743442 w_k1_FDX_s103
                                 + 1.22598 w_k1_IBM_s103
                                 + 0.878754 w_k1_INTC_s103
                                 + 0.917724 w_k1_KO_s103
                                 + 0.851753 w_k1_MAT_s103
                                 + 0.807131 w_k1_MSFT_s103
                                 + 0.911268 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.17646 w_k1_ADM_s103
                                 + 1.01253 w_k1_AXP_s103 + 1.4239 w_k1_DIS_s103
                                 + 1.18433 w_k1_FDX_s103
                                 + 1.10923 w_k1_IBM_s103
                                 + 1.11714 w_k1_INTC_s103
                                 + 1.33817 w_k1_KO_s103 + 1.05017 w_k1_MAT_s103
                                 + 1.1741 w_k1_MSFT_s103 + 1.13776 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.665152 w_k1_ADM_s103
                                 + 0.591243 w_k1_AXP_s103
                                 + 0.511327 w_k1_DIS_s103
                                 + 0.668471 w_k1_FDX_s103
                                 + 0.8613 w_k1_IBM_s103
                                 + 0.730907 w_k1_INTC_s103
                                 + 0.789001 w_k1_KO_s103
                                 + 0.905544 w_k1_MAT_s103
                                 + 0.75063 w_k1_MSFT_s103
                                 + 0.798058 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.10498 w_k1_ADM_s103
                                 + 1.34827 w_k1_AXP_s103
                                 + 1.60852 w_k1_DIS_s103
                                 + 1.25604 w_k1_FDX_s103
                                 + 1.66516 w_k1_IBM_s103
                                 + 1.35849 w_k1_INTC_s103
                                 + 1.67133 w_k1_KO_s103 + 1.84794 w_k1_MAT_s103
                                 + 1.27483 w_k1_MSFT_s103
                                 + 1.24879 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.01949 w_k1_ADM_s103
                                 + 1.39937 w_k1_AXP_s103
                                 + 0.977302 w_k1_DIS_s103
                                 + 1.20137 w_k1_FDX_s103
                                 + 1.34616 w_k1_IBM_s103
                                 + 1.05917 w_k1_INTC_s103
                                 + 1.08906 w_k1_KO_s103 + 1.81076 w_k1_MAT_s103
                                 + 1.09341 w_k1_MSFT_s103
                                 + 1.07392 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.88089 w_k1_ADM_s103
                                 + 0.772836 w_k1_AXP_s103
                                 + 0.827855 w_k1_DIS_s103
                                 + 0.956927 w_k1_FDX_s103
                                 + 0.784397 w_k1_IBM_s103
                                 + 0.978091 w_k1_INTC_s103
                                 + 0.818553 w_k1_KO_s103
                                 + 0.846713 w_k1_MAT_s103
                                 + 0.898624 w_k1_MSFT_s103
                                 + 0.863764 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.933086 w_k1_ADM_s103
                                 + 0.806898 w_k1_AXP_s103
                                 + 0.834264 w_k1_DIS_s103
                                 + 0.96327 w_k1_FDX_s103
                                 + 0.941922 w_k1_IBM_s103
                                 + 1.0385 w_k1_INTC_s103 + 1.02232 w_k1_KO_s103
                                 + 0.818754 w_k1_MAT_s103
                                 + 0.886695 w_k1_MSFT_s103
                                 + 0.923816 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.903574 w_k1_ADM_s103
                                 + 0.542124 w_k1_AXP_s103
                                 + 0.544886 w_k1_DIS_s103
                                 + 0.705972 w_k1_FDX_s103
                                 + 0.585439 w_k1_IBM_s103
                                 + 0.715857 w_k1_INTC_s103
                                 + 0.633442 w_k1_KO_s103
                                 + 0.472114 w_k1_MAT_s103
                                 + 0.834904 w_k1_MSFT_s103
                                 + 0.764012 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.09359 w_k1_ADM_s104
                                 + 1.05673 w_k1_AXP_s104
                                 + 1.19851 w_k1_DIS_s104
                                 + 1.03549 w_k1_FDX_s104
                                 + 1.40483 w_k1_IBM_s104
                                 + 0.860129 w_k1_INTC_s104
                                 + 1.04052 w_k1_KO_s104 + 1.50136 w_k1_MAT_s104
                                 + 1.01572 w_k1_MSFT_s104
                                 + 1.05836 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.11269 w_k1_ADM_s104
                                 + 1.34189 w_k1_AXP_s104
                                 + 1.59605 w_k1_DIS_s104
                                 + 1.31966 w_k1_FDX_s104 + 1.4203 w_k1_IBM_s104
                                 + 1.2244 w_k1_INTC_s104 + 1.30233 w_k1_KO_s104
                                 + 1.56864 w_k1_MAT_s104
                                 + 1.18322 w_k1_MSFT_s104
                                 + 1.16381 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.969973 w_k1_ADM_s104
                                 + 0.833387 w_k1_AXP_s104
                                 + 0.87752 w_k1_DIS_s104
                                 + 0.979879 w_k1_FDX_s104
                                 + 0.834019 w_k1_IBM_s104
                                 + 0.793745 w_k1_INTC_s104
                                 + 0.820061 w_k1_KO_s104
                                 + 0.847581 w_k1_MAT_s104
                                 + 0.936553 w_k1_MSFT_s104
                                 + 0.833943 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.755123 w_k1_ADM_s104
                                 + 0.746042 w_k1_AXP_s104
                                 + 0.598275 w_k1_DIS_s104
                                 + 0.763352 w_k1_FDX_s104
                                 + 0.870587 w_k1_IBM_s104
                                 + 0.839565 w_k1_INTC_s104
                                 + 0.774833 w_k1_KO_s104
                                 + 0.880434 w_k1_MAT_s104
                                 + 0.835117 w_k1_MSFT_s104
                                 + 0.850739 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.931869 w_k1_ADM_s104
                                 + 1.37933 w_k1_AXP_s104 + 1.1481 w_k1_DIS_s104
                                 + 1.0016 w_k1_FDX_s104 + 1.66799 w_k1_IBM_s104
                                 + 1.05487 w_k1_INTC_s104
                                 + 1.24198 w_k1_KO_s104 + 2.20739 w_k1_MAT_s104
                                 + 1.12113 w_k1_MSFT_s104
                                 + 1.03366 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.16059 w_k1_ADM_s104
                                 + 1.30661 w_k1_AXP_s104
                                 + 1.33225 w_k1_DIS_s104
                                 + 1.36684 w_k1_FDX_s104
                                 + 0.955888 w_k1_IBM_s104
                                 + 1.12888 w_k1_INTC_s104
                                 + 1.00065 w_k1_KO_s104
                                 + 0.784087 w_k1_MAT_s104
                                 + 1.09429 w_k1_MSFT_s104
                                 + 1.07972 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.09517 w_k1_ADM_s104
                                 + 1.01611 w_k1_AXP_s104
                                 + 1.00217 w_k1_DIS_s104
                                 + 0.979491 w_k1_FDX_s104
                                 + 1.14834 w_k1_IBM_s104
                                 + 0.826533 w_k1_INTC_s104
                                 + 0.979886 w_k1_KO_s104
                                 + 1.01725 w_k1_MAT_s104
                                 + 0.93815 w_k1_MSFT_s104
                                 + 0.959873 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.804862 w_k1_ADM_s104
                                 + 0.7595 w_k1_AXP_s104
                                 + 0.638673 w_k1_DIS_s104
                                 + 0.764799 w_k1_FDX_s104
                                 + 1.22152 w_k1_IBM_s104
                                 + 0.719438 w_k1_INTC_s104
                                 + 0.768113 w_k1_KO_s104
                                 + 1.04455 w_k1_MAT_s104
                                 + 0.785754 w_k1_MSFT_s104
                                 + 0.780271 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.1972 w_k1_ADM_s105
                                 + 1.12688 w_k1_AXP_s105
                                 + 1.32011 w_k1_DIS_s105
                                 + 1.10972 w_k1_FDX_s105
                                 + 1.33244 w_k1_IBM_s105
                                 + 0.833795 w_k1_INTC_s105
                                 + 1.13198 w_k1_KO_s105 + 1.87052 w_k1_MAT_s105
                                 + 1.04015 w_k1_MSFT_s105
                                 + 1.07993 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.794206 w_k1_ADM_s105
                                 + 0.801347 w_k1_AXP_s105
                                 + 0.751257 w_k1_DIS_s105
                                 + 0.811336 w_k1_FDX_s105
                                 + 0.715977 w_k1_IBM_s105
                                 + 0.78699 w_k1_INTC_s105
                                 + 0.743944 w_k1_KO_s105
                                 + 0.724819 w_k1_MAT_s105
                                 + 0.768803 w_k1_MSFT_s105
                                 + 0.810233 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.24279 w_k1_ADM_s105
                                 + 1.63493 w_k1_AXP_s105
                                 + 1.52563 w_k1_DIS_s105
                                 + 1.46191 w_k1_FDX_s105
                                 + 1.11792 w_k1_IBM_s105
                                 + 1.23674 w_k1_INTC_s105
                                 + 0.979246 w_k1_KO_s105
                                 + 1.23842 w_k1_MAT_s105
                                 + 1.09203 w_k1_MSFT_s105
                                 + 1.11639 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.01355 w_k1_ADM_s105
                                 + 0.848207 w_k1_AXP_s105
                                 + 0.984659 w_k1_DIS_s105
                                 + 1.0171 w_k1_FDX_s105
                                 + 0.654151 w_k1_IBM_s105
                                 + 0.861428 w_k1_INTC_s105
                                 + 0.762346 w_k1_KO_s105
                                 + 0.644845 w_k1_MAT_s105
                                 + 0.93068 w_k1_MSFT_s105
                                 + 0.840208 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.19842 w_k1_ADM_s105
                                 + 0.941403 w_k1_AXP_s105
                                 + 1.04028 w_k1_DIS_s105
                                 + 0.968977 w_k1_FDX_s105
                                 + 1.00226 w_k1_IBM_s105
                                 + 0.944091 w_k1_INTC_s105
                                 + 0.922833 w_k1_KO_s105
                                 + 0.988975 w_k1_MAT_s105
                                 + 0.918871 w_k1_MSFT_s105
                                 + 0.919932 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.16743 w_k1_ADM_s105
                                 + 1.13325 w_k1_AXP_s105
                                 + 1.14331 w_k1_DIS_s105
                                 + 1.11337 w_k1_FDX_s105
                                 + 0.946366 w_k1_IBM_s105
                                 + 0.968856 w_k1_INTC_s105
                                 + 0.999453 w_k1_KO_s105
                                 + 1.28534 w_k1_MAT_s105
                                 + 1.03616 w_k1_MSFT_s105
                                 + 1.07004 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.888859 w_k1_ADM_s105
                                 + 0.898726 w_k1_AXP_s105
                                 + 0.845595 w_k1_DIS_s105
                                 + 0.783343 w_k1_FDX_s105
                                 + 1.18614 w_k1_IBM_s105
                                 + 0.827026 w_k1_INTC_s105
                                 + 0.943765 w_k1_KO_s105
                                 + 1.16173 w_k1_MAT_s105
                                 + 0.821736 w_k1_MSFT_s105
                                 + 1.01032 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.772655 w_k1_ADM_s105
                                 + 0.908771 w_k1_AXP_s105
                                 + 0.842888 w_k1_DIS_s105
                                 + 0.907748 w_k1_FDX_s105
                                 + 1.04564 w_k1_IBM_s105
                                 + 0.873062 w_k1_INTC_s105
                                 + 0.88039 w_k1_KO_s105 + 1.08049 w_k1_MAT_s105
                                 + 0.904966 w_k1_MSFT_s105
                                 + 0.931686 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.24748 w_k1_ADM_s106
                                 + 1.46052 w_k1_AXP_s106
                                 + 1.33088 w_k1_DIS_s106
                                 + 1.31248 w_k1_FDX_s106
                                 + 1.27296 w_k1_IBM_s106
                                 + 1.27717 w_k1_INTC_s106
                                 + 1.36484 w_k1_KO_s106 + 1.92561 w_k1_MAT_s106
                                 + 1.22585 w_k1_MSFT_s106
                                 + 1.11798 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.869715 w_k1_ADM_s106
                                 + 0.902363 w_k1_AXP_s106
                                 + 0.876684 w_k1_DIS_s106
                                 + 0.892872 w_k1_FDX_s106
                                 + 1.34455 w_k1_IBM_s106
                                 + 0.983988 w_k1_INTC_s106
                                 + 1.11933 w_k1_KO_s106 + 1.34707 w_k1_MAT_s106
                                 + 0.977366 w_k1_MSFT_s106
                                 + 0.955303 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.18389 w_k1_ADM_s106
                                 + 1.08715 w_k1_AXP_s106
                                 + 0.987557 w_k1_DIS_s106
                                 + 1.05101 w_k1_FDX_s106
                                 + 1.01203 w_k1_IBM_s106
                                 + 1.07665 w_k1_INTC_s106
                                 + 0.884536 w_k1_KO_s106
                                 + 1.07023 w_k1_MAT_s106
                                 + 1.07099 w_k1_MSFT_s106
                                 + 0.974223 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.962718 w_k1_ADM_s106
                                 + 0.902411 w_k1_AXP_s106
                                 + 0.904239 w_k1_DIS_s106
                                 + 0.928819 w_k1_FDX_s106
                                 + 0.752525 w_k1_IBM_s106
                                 + 0.809056 w_k1_INTC_s106
                                 + 0.79324 w_k1_KO_s106
                                 + 0.711849 w_k1_MAT_s106
                                 + 0.840335 w_k1_MSFT_s106
                                 + 0.861729 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.00818 w_k1_ADM_s106
                                 + 0.737181 w_k1_AXP_s106
                                 + 0.584233 w_k1_DIS_s106
                                 + 0.772273 w_k1_FDX_s106
                                 + 0.662037 w_k1_IBM_s106
                                 + 0.638224 w_k1_INTC_s106
                                 + 0.6239 w_k1_KO_s106 + 0.583229 w_k1_MAT_s106
                                 + 0.705301 w_k1_MSFT_s106
                                 + 0.752868 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.12394 w_k1_ADM_s106
                                 + 1.21069 w_k1_AXP_s106
                                 + 1.26113 w_k1_DIS_s106
                                 + 1.24987 w_k1_FDX_s106
                                 + 0.947994 w_k1_IBM_s106
                                 + 1.16807 w_k1_INTC_s106
                                 + 1.10167 w_k1_KO_s106 + 1.25438 w_k1_MAT_s106
                                 + 1.15306 w_k1_MSFT_s106
                                 + 1.14038 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.759559 w_k1_ADM_s106
                                 + 0.727342 w_k1_AXP_s106
                                 + 0.71589 w_k1_DIS_s106
                                 + 0.76269 w_k1_FDX_s106
                                 + 0.924431 w_k1_IBM_s106
                                 + 0.833885 w_k1_INTC_s106
                                 + 0.789105 w_k1_KO_s106
                                 + 0.983062 w_k1_MAT_s106
                                 + 0.910325 w_k1_MSFT_s106
                                 + 0.888926 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.827231 w_k1_ADM_s106
                                 + 1.501 w_k1_AXP_s106 + 0.718532 w_k1_DIS_s106
                                 + 0.969612 w_k1_FDX_s106
                                 + 0.824214 w_k1_IBM_s106
                                 + 0.847348 w_k1_INTC_s106
                                 + 0.774054 w_k1_KO_s106
                                 + 1.43723 w_k1_MAT_s106
                                 + 0.867051 w_k1_MSFT_s106
                                 + 0.934554 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.823695 w_k1_ADM_s107
                                 + 0.74262 w_k1_AXP_s107
                                 + 0.914299 w_k1_DIS_s107
                                 + 0.877105 w_k1_FDX_s107
                                 + 0.909047 w_k1_IBM_s107
                                 + 1.14182 w_k1_INTC_s107 + 1.0018 w_k1_KO_s107
                                 + 0.769109 w_k1_MAT_s107
                                 + 1.03413 w_k1_MSFT_s107
                                 + 0.945605 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.03617 w_k1_ADM_s107
                                 + 0.94331 w_k1_AXP_s107
                                 + 1.12587 w_k1_DIS_s107
                                 + 0.844244 w_k1_FDX_s107
                                 + 1.63301 w_k1_IBM_s107
                                 + 1.01852 w_k1_INTC_s107
                                 + 1.26258 w_k1_KO_s107 + 1.12045 w_k1_MAT_s107
                                 + 1.05509 w_k1_MSFT_s107
                                 + 1.01498 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.981184 w_k1_ADM_s107
                                 + 0.832776 w_k1_AXP_s107
                                 + 1.16609 w_k1_DIS_s107 + 1.1301 w_k1_FDX_s107
                                 + 0.880742 w_k1_IBM_s107
                                 + 1.10107 w_k1_INTC_s107
                                 + 1.03108 w_k1_KO_s107
                                 + 0.635821 w_k1_MAT_s107
                                 + 1.16664 w_k1_MSFT_s107
                                 + 1.04758 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.744726 w_k1_ADM_s107
                                 + 1.22187 w_k1_AXP_s107
                                 + 0.864974 w_k1_DIS_s107
                                 + 0.961506 w_k1_FDX_s107
                                 + 1.09779 w_k1_IBM_s107
                                 + 1.24683 w_k1_INTC_s107
                                 + 0.938283 w_k1_KO_s107
                                 + 0.802788 w_k1_MAT_s107
                                 + 1.04162 w_k1_MSFT_s107
                                 + 0.896351 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.09885 w_k1_ADM_s107
                                 + 1.21549 w_k1_AXP_s107
                                 + 1.40798 w_k1_DIS_s107
                                 + 1.05124 w_k1_FDX_s107
                                 + 1.28322 w_k1_IBM_s107
                                 + 1.1949 w_k1_INTC_s107 + 1.24678 w_k1_KO_s107
                                 + 1.0507 w_k1_MAT_s107
                                 + 1.13396 w_k1_MSFT_s107 + 1.1125 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.758017 w_k1_ADM_s107
                                 + 0.730124 w_k1_AXP_s107
                                 + 0.73027 w_k1_DIS_s107
                                 + 0.867918 w_k1_FDX_s107
                                 + 0.597126 w_k1_IBM_s107
                                 + 0.912534 w_k1_INTC_s107
                                 + 0.652734 w_k1_KO_s107
                                 + 0.329966 w_k1_MAT_s107
                                 + 0.810428 w_k1_MSFT_s107
                                 + 0.739177 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.30799 w_k1_ADM_s107
                                 + 1.77516 w_k1_AXP_s107 + 1.5186 w_k1_DIS_s107
                                 + 1.34356 w_k1_FDX_s107
                                 + 1.09724 w_k1_IBM_s107
                                 + 1.22034 w_k1_INTC_s107
                                 + 1.16198 w_k1_KO_s107 + 1.21601 w_k1_MAT_s107
                                 + 1.12423 w_k1_MSFT_s107
                                 + 1.24196 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.841885 w_k1_ADM_s107
                                 + 0.638141 w_k1_AXP_s107
                                 + 0.659393 w_k1_DIS_s107
                                 + 0.703821 w_k1_FDX_s107
                                 + 0.884132 w_k1_IBM_s107
                                 + 0.731402 w_k1_INTC_s107
                                 + 0.722269 w_k1_KO_s107
                                 + 0.741328 w_k1_MAT_s107
                                 + 0.760712 w_k1_MSFT_s107
                                 + 0.780081 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.09807 w_k1_ADM_s110
                                 + 0.812559 w_k1_AXP_s110
                                 + 0.797454 w_k1_DIS_s110
                                 + 0.898402 w_k1_FDX_s110
                                 + 0.644519 w_k1_IBM_s110
                                 + 0.924667 w_k1_INTC_s110
                                 + 0.819928 w_k1_KO_s110
                                 + 0.489576 w_k1_MAT_s110
                                 + 0.851237 w_k1_MSFT_s110
                                 + 0.915566 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.983677 w_k1_ADM_s110
                                 + 1.03249 w_k1_AXP_s110
                                 + 1.58361 w_k1_DIS_s110
                                 + 1.37508 w_k1_FDX_s110
                                 + 0.75532 w_k1_IBM_s110
                                 + 1.39766 w_k1_INTC_s110
                                 + 1.18978 w_k1_KO_s110
                                 + 0.808812 w_k1_MAT_s110
                                 + 1.4594 w_k1_MSFT_s110 + 1.20368 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.952335 w_k1_ADM_s110
                                 + 1.15318 w_k1_AXP_s110
                                 + 0.933452 w_k1_DIS_s110
                                 + 1.01322 w_k1_FDX_s110
                                 + 1.00598 w_k1_IBM_s110
                                 + 1.05028 w_k1_INTC_s110 + 1.0818 w_k1_KO_s110
                                 + 1.02521 w_k1_MAT_s110
                                 + 0.995516 w_k1_MSFT_s110
                                 + 1.06556 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.950162 w_k1_ADM_s110
                                 + 0.752466 w_k1_AXP_s110
                                 + 0.642496 w_k1_DIS_s110
                                 + 0.79901 w_k1_FDX_s110
                                 + 0.862007 w_k1_IBM_s110
                                 + 0.842807 w_k1_INTC_s110
                                 + 0.819827 w_k1_KO_s110
                                 + 0.782684 w_k1_MAT_s110
                                 + 0.803065 w_k1_MSFT_s110
                                 + 0.907916 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.12299 w_k1_ADM_s110
                                 + 0.706647 w_k1_AXP_s110
                                 + 0.939378 w_k1_DIS_s110
                                 + 0.866946 w_k1_FDX_s110
                                 + 1.2426 w_k1_IBM_s110
                                 + 0.772584 w_k1_INTC_s110
                                 + 1.20205 w_k1_KO_s110 + 1.07258 w_k1_MAT_s110
                                 + 0.903898 w_k1_MSFT_s110
                                 + 1.0976 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.639569 w_k1_ADM_s110
                                 + 0.51044 w_k1_AXP_s110
                                 + 0.563248 w_k1_DIS_s110
                                 + 0.62652 w_k1_FDX_s110
                                 + 0.780846 w_k1_IBM_s110
                                 + 0.758916 w_k1_INTC_s110
                                 + 0.817662 w_k1_KO_s110
                                 + 0.633409 w_k1_MAT_s110
                                 + 0.88349 w_k1_MSFT_s110
                                 + 0.84598 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.30485 w_k1_ADM_s110
                                 + 0.90598 w_k1_AXP_s110
                                 + 0.961119 w_k1_DIS_s110
                                 + 1.03343 w_k1_FDX_s110
                                 + 0.857164 w_k1_IBM_s110
                                 + 0.988261 w_k1_INTC_s110
                                 + 0.974928 w_k1_KO_s110
                                 + 0.790331 w_k1_MAT_s110
                                 + 1.04313 w_k1_MSFT_s110
                                 + 1.08753 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.41903 w_k1_ADM_s110
                                 + 1.25834 w_k1_AXP_s110
                                 + 1.56163 w_k1_DIS_s110 + 1.1785 w_k1_FDX_s110
                                 + 1.26852 w_k1_IBM_s110
                                 + 1.0967 w_k1_INTC_s110 + 1.38804 w_k1_KO_s110
                                 + 0.957406 w_k1_MAT_s110
                                 + 1.04712 w_k1_MSFT_s110
                                 + 1.30995 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.991725 w_k1_ADM_s111
                                 + 1.27898 w_k1_AXP_s111
                                 + 0.975636 w_k1_DIS_s111
                                 + 1.06209 w_k1_FDX_s111
                                 + 1.28798 w_k1_IBM_s111
                                 + 1.11201 w_k1_INTC_s111
                                 + 1.25481 w_k1_KO_s111 + 1.48418 w_k1_MAT_s111
                                 + 1.04503 w_k1_MSFT_s111
                                 + 1.11753 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.892648 w_k1_ADM_s111
                                 + 0.917339 w_k1_AXP_s111
                                 + 0.786852 w_k1_DIS_s111
                                 + 0.827062 w_k1_FDX_s111
                                 + 1.14735 w_k1_IBM_s111
                                 + 0.871076 w_k1_INTC_s111
                                 + 0.945327 w_k1_KO_s111
                                 + 1.12336 w_k1_MAT_s111
                                 + 0.858895 w_k1_MSFT_s111
                                 + 0.965861 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.31804 w_k1_ADM_s111
                                 + 1.02322 w_k1_AXP_s111
                                 + 1.19258 w_k1_DIS_s111 + 1.1611 w_k1_FDX_s111
                                 + 1.0942 w_k1_IBM_s111
                                 + 0.958463 w_k1_INTC_s111
                                 + 1.16055 w_k1_KO_s111 + 1.00371 w_k1_MAT_s111
                                 + 1.02232 w_k1_MSFT_s111
                                 + 1.10037 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.62222 w_k1_ADM_s111
                                 + 1.4319 w_k1_AXP_s111 + 1.72461 w_k1_DIS_s111
                                 + 1.49561 w_k1_FDX_s111
                                 + 1.15845 w_k1_IBM_s111
                                 + 0.971002 w_k1_INTC_s111
                                 + 1.31502 w_k1_KO_s111 + 1.59335 w_k1_MAT_s111
                                 + 1.149 w_k1_MSFT_s111 + 1.30634 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.779804 w_k1_ADM_s111
                                 + 0.604213 w_k1_AXP_s111
                                 + 0.686683 w_k1_DIS_s111
                                 + 0.776395 w_k1_FDX_s111
                                 + 0.747514 w_k1_IBM_s111
                                 + 0.593744 w_k1_INTC_s111
                                 + 0.742225 w_k1_KO_s111
                                 + 0.957248 w_k1_MAT_s111
                                 + 0.775021 w_k1_MSFT_s111
                                 + 0.859114 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.859998 w_k1_ADM_s111
                                 + 0.768678 w_k1_AXP_s111
                                 + 0.881135 w_k1_DIS_s111
                                 + 0.937547 w_k1_FDX_s111
                                 + 0.858011 w_k1_IBM_s111
                                 + 0.972103 w_k1_INTC_s111
                                 + 0.808723 w_k1_KO_s111
                                 + 0.716138 w_k1_MAT_s111
                                 + 0.864046 w_k1_MSFT_s111
                                 + 0.906078 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.899226 w_k1_ADM_s111
                                 + 0.860054 w_k1_AXP_s111
                                 + 0.918806 w_k1_DIS_s111
                                 + 1.00562 w_k1_FDX_s111
                                 + 0.682256 w_k1_IBM_s111
                                 + 0.940911 w_k1_INTC_s111
                                 + 0.802504 w_k1_KO_s111
                                 + 0.909147 w_k1_MAT_s111
                                 + 1.02223 w_k1_MSFT_s111
                                 + 0.934309 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.05771 w_k1_ADM_s111
                                 + 1.26436 w_k1_AXP_s111
                                 + 1.32789 w_k1_DIS_s111
                                 + 1.12068 w_k1_FDX_s111
                                 + 1.70477 w_k1_IBM_s111
                                 + 1.01786 w_k1_INTC_s111
                                 + 1.41053 w_k1_KO_s111 + 2.0396 w_k1_MAT_s111
                                 + 0.933644 w_k1_MSFT_s111
                                 + 1.14085 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.12351 w_k1_ADM_s112
                                 + 1.17727 w_k1_AXP_s112
                                 + 1.37072 w_k1_DIS_s112
                                 + 1.12875 w_k1_FDX_s112
                                 + 0.803633 w_k1_IBM_s112
                                 + 1.04753 w_k1_INTC_s112
                                 + 0.985906 w_k1_KO_s112
                                 + 1.00174 w_k1_MAT_s112
                                 + 1.14356 w_k1_MSFT_s112
                                 + 1.03975 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.34608 w_k1_ADM_s112
                                 + 1.09473 w_k1_AXP_s112
                                 + 1.36627 w_k1_DIS_s112
                                 + 1.18415 w_k1_FDX_s112
                                 + 1.04467 w_k1_IBM_s112
                                 + 1.08992 w_k1_INTC_s112
                                 + 1.09758 w_k1_KO_s112
                                 + 0.955373 w_k1_MAT_s112
                                 + 1.122 w_k1_MSFT_s112 + 1.05935 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.06315 w_k1_ADM_s112
                                 + 0.968648 w_k1_AXP_s112
                                 + 0.975823 w_k1_DIS_s112
                                 + 1.01734 w_k1_FDX_s112
                                 + 0.965366 w_k1_IBM_s112
                                 + 0.979671 w_k1_INTC_s112
                                 + 0.95569 w_k1_KO_s112
                                 + 0.795399 w_k1_MAT_s112
                                 + 0.904118 w_k1_MSFT_s112
                                 + 0.90487 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.19133 w_k1_ADM_s112
                                 + 1.20649 w_k1_AXP_s112
                                 + 1.21395 w_k1_DIS_s112
                                 + 0.985352 w_k1_FDX_s112
                                 + 1.43433 w_k1_IBM_s112
                                 + 0.831728 w_k1_INTC_s112
                                 + 1.13734 w_k1_KO_s112 + 1.3158 w_k1_MAT_s112
                                 + 0.910184 w_k1_MSFT_s112
                                 + 1.08651 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.1959 w_k1_ADM_s112
                                 + 1.67148 w_k1_AXP_s112
                                 + 1.72901 w_k1_DIS_s112
                                 + 1.51387 w_k1_FDX_s112
                                 + 1.23307 w_k1_IBM_s112
                                 + 1.26087 w_k1_INTC_s112
                                 + 1.28644 w_k1_KO_s112 + 1.3502 w_k1_MAT_s112
                                 + 1.25807 w_k1_MSFT_s112
                                 + 1.11399 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.892574 w_k1_ADM_s112
                                 + 0.665382 w_k1_AXP_s112
                                 + 0.700389 w_k1_DIS_s112
                                 + 0.759598 w_k1_FDX_s112
                                 + 0.675616 w_k1_IBM_s112
                                 + 0.521581 w_k1_INTC_s112
                                 + 0.771997 w_k1_KO_s112
                                 + 0.59044 w_k1_MAT_s112
                                 + 0.793867 w_k1_MSFT_s112
                                 + 0.837535 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.938258 w_k1_ADM_s112
                                 + 0.730341 w_k1_AXP_s112
                                 + 0.827652 w_k1_DIS_s112
                                 + 0.862828 w_k1_FDX_s112
                                 + 0.943515 w_k1_IBM_s112
                                 + 0.87027 w_k1_INTC_s112
                                 + 0.860896 w_k1_KO_s112
                                 + 0.641935 w_k1_MAT_s112
                                 + 0.912821 w_k1_MSFT_s112
                                 + 0.945201 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.730542 w_k1_ADM_s112
                                 + 0.6548 w_k1_AXP_s112
                                 + 0.626085 w_k1_DIS_s112
                                 + 0.653792 w_k1_FDX_s112
                                 + 1.08515 w_k1_IBM_s112
                                 + 0.699366 w_k1_INTC_s112
                                 + 0.801717 w_k1_KO_s112
                                 + 0.657252 w_k1_MAT_s112
                                 + 0.669117 w_k1_MSFT_s112
                                 + 0.783858 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.15756 w_k1_ADM_s113
                                 + 1.39803 w_k1_AXP_s113
                                 + 1.61338 w_k1_DIS_s113
                                 + 1.32001 w_k1_FDX_s113
                                 + 1.41367 w_k1_IBM_s113
                                 + 1.36884 w_k1_INTC_s113
                                 + 1.58883 w_k1_KO_s113 + 1.11065 w_k1_MAT_s113
                                 + 1.28355 w_k1_MSFT_s113
                                 + 1.19072 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.707711 w_k1_ADM_s113
                                 + 1.11243 w_k1_AXP_s113
                                 + 1.13936 w_k1_DIS_s113
                                 + 1.02904 w_k1_FDX_s113
                                 + 1.13761 w_k1_IBM_s113
                                 + 1.38622 w_k1_INTC_s113
                                 + 1.17183 w_k1_KO_s113 + 1.13505 w_k1_MAT_s113
                                 + 1.07323 w_k1_MSFT_s113
                                 + 1.07063 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.764246 w_k1_ADM_s113
                                 + 0.670284 w_k1_AXP_s113
                                 + 0.661753 w_k1_DIS_s113
                                 + 0.778836 w_k1_FDX_s113
                                 + 0.790607 w_k1_IBM_s113
                                 + 1.04797 w_k1_INTC_s113
                                 + 0.796098 w_k1_KO_s113
                                 + 0.684944 w_k1_MAT_s113
                                 + 0.875336 w_k1_MSFT_s113
                                 + 0.854283 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.12558 w_k1_ADM_s113
                                 + 1.05392 w_k1_AXP_s113 + 1.3402 w_k1_DIS_s113
                                 + 1.20309 w_k1_FDX_s113
                                 + 0.823131 w_k1_IBM_s113
                                 + 1.54639 w_k1_INTC_s113 + 1.1381 w_k1_KO_s113
                                 + 0.682841 w_k1_MAT_s113
                                 + 1.28446 w_k1_MSFT_s113
                                 + 1.05555 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.959659 w_k1_ADM_s113
                                 + 1.02942 w_k1_AXP_s113
                                 + 0.82225 w_k1_DIS_s113
                                 + 0.96527 w_k1_FDX_s113
                                 + 0.866868 w_k1_IBM_s113
                                 + 1.1589 w_k1_INTC_s113
                                 + 0.953251 w_k1_KO_s113
                                 + 0.907337 w_k1_MAT_s113
                                 + 0.997117 w_k1_MSFT_s113
                                 + 0.834168 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.816327 w_k1_ADM_s113
                                 + 0.895942 w_k1_AXP_s113
                                 + 0.858962 w_k1_DIS_s113
                                 + 0.826776 w_k1_FDX_s113
                                 + 1.1481 w_k1_IBM_s113
                                 + 1.19337 w_k1_INTC_s113
                                 + 0.907337 w_k1_KO_s113
                                 + 0.907516 w_k1_MAT_s113
                                 + 0.926291 w_k1_MSFT_s113
                                 + 0.854962 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.21768 w_k1_ADM_s113
                                 + 1.04567 w_k1_AXP_s113
                                 + 0.809764 w_k1_DIS_s113
                                 + 1.03469 w_k1_FDX_s113
                                 + 0.690091 w_k1_IBM_s113
                                 + 1.17146 w_k1_INTC_s113
                                 + 0.838725 w_k1_KO_s113
                                 + 0.96234 w_k1_MAT_s113
                                 + 1.01677 w_k1_MSFT_s113
                                 + 0.911581 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.09356 w_k1_ADM_s113
                                 + 1.43167 w_k1_AXP_s113
                                 + 1.06554 w_k1_DIS_s113
                                 + 1.13508 w_k1_FDX_s113
                                 + 1.01057 w_k1_IBM_s113
                                 + 1.06257 w_k1_INTC_s113
                                 + 1.00043 w_k1_KO_s113 + 1.51767 w_k1_MAT_s113
                                 + 1.07435 w_k1_MSFT_s113
                                 + 1.03653 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.76605 w_k1_ADM_s114
                                 + 0.897876 w_k1_AXP_s114
                                 + 1.00158 w_k1_DIS_s114
                                 + 0.96923 w_k1_FDX_s114
                                 + 1.17535 w_k1_IBM_s114
                                 + 0.763071 w_k1_INTC_s114
                                 + 0.964489 w_k1_KO_s114
                                 + 1.38882 w_k1_MAT_s114
                                 + 1.03809 w_k1_MSFT_s114
                                 + 0.809747 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.484364 w_k1_ADM_s114
                                 + 0.61901 w_k1_AXP_s114
                                 + 0.52699 w_k1_DIS_s114
                                 + 0.678076 w_k1_FDX_s114
                                 + 0.837904 w_k1_IBM_s114
                                 + 0.609332 w_k1_INTC_s114
                                 + 0.633276 w_k1_KO_s114
                                 + 0.690447 w_k1_MAT_s114
                                 + 0.649133 w_k1_MSFT_s114
                                 + 0.673898 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.969051 w_k1_ADM_s114
                                 + 1.27562 w_k1_AXP_s114
                                 + 1.28591 w_k1_DIS_s114
                                 + 1.27368 w_k1_FDX_s114
                                 + 0.970765 w_k1_IBM_s114
                                 + 0.901091 w_k1_INTC_s114
                                 + 0.9944 w_k1_KO_s114 + 1.20096 w_k1_MAT_s114
                                 + 1.24008 w_k1_MSFT_s114
                                 + 1.13698 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.832265 w_k1_ADM_s114
                                 + 1.40536 w_k1_AXP_s114
                                 + 1.58138 w_k1_DIS_s114
                                 + 1.45798 w_k1_FDX_s114
                                 + 1.23674 w_k1_IBM_s114
                                 + 1.16684 w_k1_INTC_s114
                                 + 1.15038 w_k1_KO_s114 + 1.46812 w_k1_MAT_s114
                                 + 1.29591 w_k1_MSFT_s114
                                 + 1.07341 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.919899 w_k1_ADM_s114
                                 + 1.44317 w_k1_AXP_s114
                                 + 1.49567 w_k1_DIS_s114
                                 + 1.01527 w_k1_FDX_s114
                                 + 1.86915 w_k1_IBM_s114
                                 + 1.03041 w_k1_INTC_s114
                                 + 1.58113 w_k1_KO_s114 + 1.7438 w_k1_MAT_s114
                                 + 1.06931 w_k1_MSFT_s114 + 1.0486 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.665242 w_k1_ADM_s114
                                 + 0.677263 w_k1_AXP_s114
                                 + 0.626806 w_k1_DIS_s114
                                 + 0.80778 w_k1_FDX_s114
                                 + 0.597079 w_k1_IBM_s114
                                 + 0.626794 w_k1_INTC_s114
                                 + 0.639679 w_k1_KO_s114
                                 + 0.677984 w_k1_MAT_s114
                                 + 0.724644 w_k1_MSFT_s114
                                 + 0.660558 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.892132 w_k1_ADM_s114
                                 + 0.779072 w_k1_AXP_s114
                                 + 0.720032 w_k1_DIS_s114
                                 + 0.882401 w_k1_FDX_s114
                                 + 0.834017 w_k1_IBM_s114
                                 + 0.634192 w_k1_INTC_s114
                                 + 0.81199 w_k1_KO_s114
                                 + 0.791266 w_k1_MAT_s114
                                 + 0.929425 w_k1_MSFT_s114
                                 + 0.894723 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.17711 w_k1_ADM_s114
                                 + 0.955806 w_k1_AXP_s114
                                 + 1.06677 w_k1_DIS_s114
                                 + 1.13607 w_k1_FDX_s114
                                 + 0.755863 w_k1_IBM_s114
                                 + 0.985803 w_k1_INTC_s114
                                 + 0.902825 w_k1_KO_s114
                                 + 0.664284 w_k1_MAT_s114
                                 + 1.10465 w_k1_MSFT_s114
                                 + 0.960588 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.17528 w_k1_ADM_s115
                                 + 0.861005 w_k1_AXP_s115
                                 + 0.751416 w_k1_DIS_s115
                                 + 0.854757 w_k1_FDX_s115
                                 + 0.968511 w_k1_IBM_s115
                                 + 1.04489 w_k1_INTC_s115
                                 + 0.949907 w_k1_KO_s115
                                 + 1.07945 w_k1_MAT_s115
                                 + 0.836897 w_k1_MSFT_s115
                                 + 0.925611 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.852003 w_k1_ADM_s115
                                 + 0.79651 w_k1_AXP_s115
                                 + 0.684916 w_k1_DIS_s115
                                 + 0.741599 w_k1_FDX_s115
                                 + 0.881668 w_k1_IBM_s115
                                 + 1.01249 w_k1_INTC_s115
                                 + 0.866239 w_k1_KO_s115
                                 + 0.671578 w_k1_MAT_s115
                                 + 0.837097 w_k1_MSFT_s115
                                 + 0.949693 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.51479 w_k1_ADM_s115
                                 + 1.79034 w_k1_AXP_s115
                                 + 1.51872 w_k1_DIS_s115
                                 + 1.32128 w_k1_FDX_s115
                                 + 1.70921 w_k1_IBM_s115
                                 + 1.55993 w_k1_INTC_s115 + 1.8005 w_k1_KO_s115
                                 + 2.48722 w_k1_MAT_s115
                                 + 1.54848 w_k1_MSFT_s115
                                 + 1.39239 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.887171 w_k1_ADM_s115
                                 + 0.883155 w_k1_AXP_s115
                                 + 0.980703 w_k1_DIS_s115
                                 + 1.02981 w_k1_FDX_s115
                                 + 0.758242 w_k1_IBM_s115
                                 + 1.40171 w_k1_INTC_s115
                                 + 0.91551 w_k1_KO_s115
                                 + 0.667617 w_k1_MAT_s115
                                 + 1.05942 w_k1_MSFT_s115
                                 + 0.946462 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.09435 w_k1_ADM_s115
                                 + 0.921118 w_k1_AXP_s115
                                 + 0.875119 w_k1_DIS_s115
                                 + 0.916561 w_k1_FDX_s115
                                 + 1.45036 w_k1_IBM_s115
                                 + 1.16014 w_k1_INTC_s115 + 1.1647 w_k1_KO_s115
                                 + 1.29127 w_k1_MAT_s115
                                 + 1.03557 w_k1_MSFT_s115
                                 + 0.951217 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.37858 w_k1_ADM_s115
                                 + 1.19018 w_k1_AXP_s115
                                 + 1.34901 w_k1_DIS_s115
                                 + 1.19761 w_k1_FDX_s115
                                 + 1.06566 w_k1_IBM_s115
                                 + 1.51691 w_k1_INTC_s115
                                 + 1.26346 w_k1_KO_s115 + 1.17595 w_k1_MAT_s115
                                 + 1.25044 w_k1_MSFT_s115
                                 + 1.06181 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.10615 w_k1_ADM_s115
                                 + 1.49069 w_k1_AXP_s115
                                 + 0.980627 w_k1_DIS_s115
                                 + 1.11656 w_k1_FDX_s115 + 1.0254 w_k1_IBM_s115
                                 + 1.28054 w_k1_INTC_s115
                                 + 1.13715 w_k1_KO_s115 + 1.7852 w_k1_MAT_s115
                                 + 1.05402 w_k1_MSFT_s115
                                 + 1.24954 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.729741 w_k1_ADM_s115
                                 + 0.746136 w_k1_AXP_s115
                                 + 0.533046 w_k1_DIS_s115
                                 + 0.762596 w_k1_FDX_s115
                                 + 0.668897 w_k1_IBM_s115
                                 + 1.16276 w_k1_INTC_s115
                                 + 0.820762 w_k1_KO_s115
                                 + 0.568569 w_k1_MAT_s115
                                 + 0.884738 w_k1_MSFT_s115
                                 + 0.856165 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.00317 w_k1_ADM_s116
                                 + 1.28561 w_k1_AXP_s116 + 1.0045 w_k1_DIS_s116
                                 + 1.04224 w_k1_FDX_s116
                                 + 1.14536 w_k1_IBM_s116
                                 + 1.18947 w_k1_INTC_s116
                                 + 0.963794 w_k1_KO_s116
                                 + 1.29811 w_k1_MAT_s116
                                 + 1.08357 w_k1_MSFT_s116
                                 + 1.10674 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.09007 w_k1_ADM_s116
                                 + 1.34342 w_k1_AXP_s116
                                 + 1.17996 w_k1_DIS_s116
                                 + 1.19161 w_k1_FDX_s116 + 1.0462 w_k1_IBM_s116
                                 + 1.15494 w_k1_INTC_s116
                                 + 1.13094 w_k1_KO_s116 + 1.03379 w_k1_MAT_s116
                                 + 1.1089 w_k1_MSFT_s116 + 1.08392 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.43445 w_k1_ADM_s116
                                 + 1.59144 w_k1_AXP_s116
                                 + 1.64872 w_k1_DIS_s116
                                 + 1.34934 w_k1_FDX_s116
                                 + 1.29531 w_k1_IBM_s116
                                 + 0.959055 w_k1_INTC_s116
                                 + 1.25491 w_k1_KO_s116 + 1.50225 w_k1_MAT_s116
                                 + 1.27686 w_k1_MSFT_s116
                                 + 1.12088 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.913344 w_k1_ADM_s116
                                 + 0.8853 w_k1_AXP_s116
                                 + 0.975686 w_k1_DIS_s116
                                 + 0.969153 w_k1_FDX_s116
                                 + 0.874011 w_k1_IBM_s116
                                 + 1.07577 w_k1_INTC_s116
                                 + 0.852557 w_k1_KO_s116
                                 + 0.886333 w_k1_MAT_s116
                                 + 1.09148 w_k1_MSFT_s116
                                 + 0.963344 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.940935 w_k1_ADM_s116
                                 + 0.906073 w_k1_AXP_s116
                                 + 0.715924 w_k1_DIS_s116
                                 + 0.856829 w_k1_FDX_s116
                                 + 0.850519 w_k1_IBM_s116
                                 + 0.797353 w_k1_INTC_s116
                                 + 0.661291 w_k1_KO_s116
                                 + 0.85653 w_k1_MAT_s116
                                 + 0.798331 w_k1_MSFT_s116
                                 + 0.841671 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.01993 w_k1_ADM_s116
                                 + 1.26438 w_k1_AXP_s116
                                 + 1.23323 w_k1_DIS_s116
                                 + 1.07842 w_k1_FDX_s116
                                 + 1.49753 w_k1_IBM_s116
                                 + 1.00682 w_k1_INTC_s116
                                 + 1.25415 w_k1_KO_s116 + 1.50469 w_k1_MAT_s116
                                 + 1.10449 w_k1_MSFT_s116
                                 + 1.20153 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.826612 w_k1_ADM_s116
                                 + 0.903903 w_k1_AXP_s116
                                 + 0.707617 w_k1_DIS_s116
                                 + 0.767123 w_k1_FDX_s116
                                 + 1.22183 w_k1_IBM_s116
                                 + 1.02811 w_k1_INTC_s116
                                 + 1.04076 w_k1_KO_s116 + 1.11133 w_k1_MAT_s116
                                 + 1.00937 w_k1_MSFT_s116 + 1.0488 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.727433 w_k1_ADM_s116
                                 + 0.765972 w_k1_AXP_s116
                                 + 0.795945 w_k1_DIS_s116
                                 + 0.844174 w_k1_FDX_s116
                                 + 0.619369 w_k1_IBM_s116
                                 + 0.830084 w_k1_INTC_s116
                                 + 0.773143 w_k1_KO_s116
                                 + 0.573328 w_k1_MAT_s116
                                 + 0.897465 w_k1_MSFT_s116
                                 + 0.830302 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.997966 w_k1_ADM_s117
                                 + 0.700564 w_k1_AXP_s117
                                 + 0.689139 w_k1_DIS_s117
                                 + 0.773323 w_k1_FDX_s117
                                 + 0.877352 w_k1_IBM_s117
                                 + 0.850882 w_k1_INTC_s117
                                 + 0.885216 w_k1_KO_s117
                                 + 1.03705 w_k1_MAT_s117
                                 + 0.993751 w_k1_MSFT_s117
                                 + 0.85097 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.954701 w_k1_ADM_s117
                                 + 0.779464 w_k1_AXP_s117
                                 + 0.796671 w_k1_DIS_s117
                                 + 0.799135 w_k1_FDX_s117
                                 + 1.2342 w_k1_IBM_s117
                                 + 0.729191 w_k1_INTC_s117
                                 + 0.877412 w_k1_KO_s117
                                 + 1.39285 w_k1_MAT_s117
                                 + 0.792582 w_k1_MSFT_s117
                                 + 0.86568 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.24292 w_k1_ADM_s117
                                 + 1.01237 w_k1_AXP_s117
                                 + 1.05194 w_k1_DIS_s117
                                 + 1.04106 w_k1_FDX_s117
                                 + 0.710307 w_k1_IBM_s117
                                 + 0.860931 w_k1_INTC_s117
                                 + 0.880009 w_k1_KO_s117
                                 + 0.970841 w_k1_MAT_s117
                                 + 1.10323 w_k1_MSFT_s117
                                 + 0.960022 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.11698 w_k1_ADM_s117
                                 + 1.1621 w_k1_AXP_s117 + 1.43521 w_k1_DIS_s117
                                 + 1.04998 w_k1_FDX_s117
                                 + 1.66521 w_k1_IBM_s117
                                 + 1.00232 w_k1_INTC_s117
                                 + 1.48628 w_k1_KO_s117 + 2.39785 w_k1_MAT_s117
                                 + 1.2237 w_k1_MSFT_s117 + 1.21609 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.48657 w_k1_ADM_s117
                                 + 1.2793 w_k1_AXP_s117 + 1.48325 w_k1_DIS_s117
                                 + 1.35284 w_k1_FDX_s117
                                 + 1.02792 w_k1_IBM_s117
                                 + 1.23276 w_k1_INTC_s117
                                 + 1.30815 w_k1_KO_s117 + 1.53821 w_k1_MAT_s117
                                 + 1.29978 w_k1_MSFT_s117
                                 + 1.18543 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.801605 w_k1_ADM_s117
                                 + 0.561581 w_k1_AXP_s117
                                 + 0.570653 w_k1_DIS_s117
                                 + 0.706672 w_k1_FDX_s117
                                 + 0.630647 w_k1_IBM_s117
                                 + 0.768807 w_k1_INTC_s117
                                 + 0.615051 w_k1_KO_s117
                                 + 0.620524 w_k1_MAT_s117
                                 + 0.692268 w_k1_MSFT_s117
                                 + 0.765323 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.41617 w_k1_ADM_s117
                                 + 1.37608 w_k1_AXP_s117
                                 + 1.45327 w_k1_DIS_s117
                                 + 1.19969 w_k1_FDX_s117
                                 + 1.26033 w_k1_IBM_s117
                                 + 1.12493 w_k1_INTC_s117
                                 + 1.21488 w_k1_KO_s117 + 1.75489 w_k1_MAT_s117
                                 + 1.1713 w_k1_MSFT_s117 + 1.17467 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.1175 w_k1_ADM_s117
                                 + 1.05887 w_k1_AXP_s117
                                 + 1.11134 w_k1_DIS_s117
                                 + 1.02781 w_k1_FDX_s117
                                 + 0.971222 w_k1_IBM_s117
                                 + 1.0377 w_k1_INTC_s117 + 1.14672 w_k1_KO_s117
                                 + 1.32037 w_k1_MAT_s117
                                 + 1.18296 w_k1_MSFT_s117
                                 + 1.02513 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0325229330942678 w_k1_ADM_s000
                                 + 0.080988907244986 w_k1_ADM_s001
                                 + 0.0314015560760573 w_k1_ADM_s002
                                 + 0.0576924310193073 w_k1_ADM_s003
                                 + 0.0252565185063021 w_k1_ADM_s004
                                 + 0.021323322942237 w_k1_ADM_s005
                                 + 0.0290029103563101 w_k1_ADM_s006
                                 + 0.0285420498626738 w_k1_ADM_s007
                                 + 0.00456693320275037 w_k1_ADM_s010
                                 + 0.0185601314944367 w_k1_ADM_s011
                                 + 0.00533192058370286 w_k1_ADM_s012
                                 + 0.00940335540336307 w_k1_ADM_s013
                                 + 0.0175357653007894 w_k1_ADM_s014
                                 + 0.00755207341230495 w_k1_ADM_s015
                                 + 0.00727616477153992 w_k1_ADM_s016
                                 + 0.012478003192311 w_k1_ADM_s017
                                 + 0.0317869456624888 w_k1_ADM_s100
                                 + 0.071896054656541 w_k1_ADM_s101
                                 + 0.0727746435821229 w_k1_ADM_s102
                                 + 0.0894257689044559 w_k1_ADM_s103
                                 + 0.0453124548254567 w_k1_ADM_s104
                                 + 0.0205778589958952 w_k1_ADM_s105
                                 + 0.033494025046965 w_k1_ADM_s106
                                 + 0.029757416967241 w_k1_ADM_s107
                                 + 0.0325734065383883 w_k1_ADM_s110
                                 + 0.0166390909075975 w_k1_ADM_s111
                                 + 0.0390154673840083 w_k1_ADM_s112
                                 + 0.0128279818224464 w_k1_ADM_s113
                                 + 0.0131662906984592 w_k1_ADM_s114
                                 + 0.0326408142567517 w_k1_ADM_s115
                                 + 0.0184390383053105 w_k1_ADM_s116
                                 + 0.0649951648277068 w_k1_ADM_s117
                                 + 0.0393729751428655 w_k1_AXP_s000
                                 + 0.0807077509619292 w_k1_AXP_s001
                                 + 0.03082754872854 w_k1_AXP_s002
                                 + 0.0599832579501551 w_k1_AXP_s003
                                 + 0.0260723626389391 w_k1_AXP_s004
                                 + 0.0232424251740297 w_k1_AXP_s005
                                 + 0.0319444911872034 w_k1_AXP_s006
                                 + 0.0292499600642795 w_k1_AXP_s007
                                 + 0.00374884330179145 w_k1_AXP_s010
                                 + 0.0159609617841856 w_k1_AXP_s011
                                 + 0.00543588445866191 w_k1_AXP_s012
                                 + 0.00724008386434376 w_k1_AXP_s013
                                 + 0.0157741687888273 w_k1_AXP_s014
                                 + 0.0068751358902853 w_k1_AXP_s015
                                 + 0.00700373734055441 w_k1_AXP_s016
                                 + 0.0104337489620221 w_k1_AXP_s017
                                 + 0.0370829521366344 w_k1_AXP_s100
                                 + 0.0787718665020256 w_k1_AXP_s101
                                 + 0.0798140273178953 w_k1_AXP_s102
                                 + 0.0817104079389366 w_k1_AXP_s103
                                 + 0.0460412811755959 w_k1_AXP_s104
                                 + 0.0204262834972028 w_k1_AXP_s105
                                 + 0.034460410680977 w_k1_AXP_s106
                                 + 0.0308639756949953 w_k1_AXP_s107
                                 + 0.0267455563077641 w_k1_AXP_s110
                                 + 0.0154568442133421 w_k1_AXP_s111
                                 + 0.0384046092573674 w_k1_AXP_s112
                                 + 0.014093047957614 w_k1_AXP_s113
                                 + 0.0153165900464436 w_k1_AXP_s114
                                 + 0.0320029318971187 w_k1_AXP_s115
                                 + 0.0205313103043979 w_k1_AXP_s116
                                 + 0.0568374692665112 w_k1_AXP_s117
                                 + 0.039635774460536 w_k1_DIS_s000
                                 + 0.0751389032020822 w_k1_DIS_s001
                                 + 0.0285121006594326 w_k1_DIS_s002
                                 + 0.0623064792085269 w_k1_DIS_s003
                                 + 0.02714304798155 w_k1_DIS_s004
                                 + 0.0223043779216177 w_k1_DIS_s005
                                 + 0.0324209992210638 w_k1_DIS_s006
                                 + 0.0299002703251048 w_k1_DIS_s007
                                 + 0.00360231997042329 w_k1_DIS_s010
                                 + 0.0158731624550187 w_k1_DIS_s011
                                 + 0.005212871333296 w_k1_DIS_s012
                                 + 0.00708914076632773 w_k1_DIS_s013
                                 + 0.0171466906605958 w_k1_DIS_s014
                                 + 0.00667145703822231 w_k1_DIS_s015
                                 + 0.00726981005215504 w_k1_DIS_s016
                                 + 0.0110182729709898 w_k1_DIS_s017
                                 + 0.0348763945345749 w_k1_DIS_s100
                                 + 0.0792254350619689 w_k1_DIS_s101
                                 + 0.077415263852553 w_k1_DIS_s102
                                 + 0.0923508594819969 w_k1_DIS_s103
                                 + 0.0463261243405547 w_k1_DIS_s104
                                 + 0.0210614016983563 w_k1_DIS_s105
                                 + 0.031205344295025 w_k1_DIS_s106
                                 + 0.0335928659906851 w_k1_DIS_s107
                                 + 0.0298098170313079 w_k1_DIS_s110
                                 + 0.0163044894884011 w_k1_DIS_s111
                                 + 0.0410627644584167 w_k1_DIS_s112
                                 + 0.013528031337726 w_k1_DIS_s113
                                 + 0.0159276994036736 w_k1_DIS_s114
                                 + 0.0292281618883066 w_k1_DIS_s115
                                 + 0.0191398126040804 w_k1_DIS_s116
                                 + 0.0608154478573735 w_k1_DIS_s117
                                 + 0.0358658006631948 w_k1_FDX_s000
                                 + 0.0790601215609997 w_k1_FDX_s001
                                 + 0.0283228709717003 w_k1_FDX_s002
                                 + 0.0641932116452471 w_k1_FDX_s003
                                 + 0.0270313419674949 w_k1_FDX_s004
                                 + 0.0215116611695139 w_k1_FDX_s005
                                 + 0.0310086955971134 w_k1_FDX_s006
                                 + 0.0296135444489781 w_k1_FDX_s007
                                 + 0.00360909727571628 w_k1_FDX_s010
                                 + 0.0162069555676971 w_k1_FDX_s011
                                 + 0.00535258656049688 w_k1_FDX_s012
                                 + 0.00695776026134736 w_k1_FDX_s013
                                 + 0.01722793003502 w_k1_FDX_s014
                                 + 0.00675392455367879 w_k1_FDX_s015
                                 + 0.00664473163688615 w_k1_FDX_s016
                                 + 0.0107073626029507 w_k1_FDX_s017
                                 + 0.0353032117893562 w_k1_FDX_s100
                                 + 0.0778837646481063 w_k1_FDX_s101
                                 + 0.0771139618737009 w_k1_FDX_s102
                                 + 0.0911405477148107 w_k1_FDX_s103
                                 + 0.0466132874792676 w_k1_FDX_s104
                                 + 0.0200814348751123 w_k1_FDX_s105
                                 + 0.0328554309693572 w_k1_FDX_s106
                                 + 0.0314476239110686 w_k1_FDX_s107
                                 + 0.0282088198736466 w_k1_FDX_s110
                                 + 0.0162809274999717 w_k1_FDX_s111
                                 + 0.037769436132918 w_k1_FDX_s112
                                 + 0.0136118455466014 w_k1_FDX_s113
                                 + 0.0160030842664194 w_k1_FDX_s114
                                 + 0.0295563045950346 w_k1_FDX_s115
                                 + 0.0191335484644715 w_k1_FDX_s116
                                 + 0.0562985277366622 w_k1_FDX_s117
                                 + 0.0362611027100923 w_k1_IBM_s000
                                 + 0.0771098421290341 w_k1_IBM_s001
                                 + 0.0279384278663857 w_k1_IBM_s002
                                 + 0.065188157438188 w_k1_IBM_s003
                                 + 0.0263720445410874 w_k1_IBM_s004
                                 + 0.0202410797105457 w_k1_IBM_s005
                                 + 0.0307234601022177 w_k1_IBM_s006
                                 + 0.0295022195150446 w_k1_IBM_s007
                                 + 0.00355954860886501 w_k1_IBM_s010
                                 + 0.0165679368587194 w_k1_IBM_s011
                                 + 0.0053110138901452 w_k1_IBM_s012
                                 + 0.00647387453512959 w_k1_IBM_s013
                                 + 0.0175168814880864 w_k1_IBM_s014
                                 + 0.00643422292545022 w_k1_IBM_s015
                                 + 0.006875300359282 w_k1_IBM_s016
                                 + 0.0106945248331841 w_k1_IBM_s017
                                 + 0.0360420743125771 w_k1_IBM_s100
                                 + 0.0752352108695163 w_k1_IBM_s101
                                 + 0.0787010421097826 w_k1_IBM_s102
                                 + 0.0939678657039479 w_k1_IBM_s103
                                 + 0.0483599752478616 w_k1_IBM_s104
                                 + 0.0198384820690633 w_k1_IBM_s105
                                 + 0.0335630268005989 w_k1_IBM_s106
                                 + 0.0307452196950628 w_k1_IBM_s107
                                 + 0.0292643610730617 w_k1_IBM_s110
                                 + 0.0162652788862597 w_k1_IBM_s111
                                 + 0.0373460802423162 w_k1_IBM_s112
                                 + 0.0133857010598296 w_k1_IBM_s113
                                 + 0.0166490567882834 w_k1_IBM_s114
                                 + 0.0299453034148407 w_k1_IBM_s115
                                 + 0.018756422235155 w_k1_IBM_s116
                                 + 0.0570224545919108 w_k1_IBM_s117
                                 + 0.0359042409798842 w_k1_INTC_s000
                                 + 0.0692478092252222 w_k1_INTC_s001
                                 + 0.0252682175692973 w_k1_INTC_s002
                                 + 0.0690995122769091 w_k1_INTC_s003
                                 + 0.0312493123380473 w_k1_INTC_s004
                                 + 0.0199082613685034 w_k1_INTC_s005
                                 + 0.0294446980289396 w_k1_INTC_s006
                                 + 0.0256183071287833 w_k1_INTC_s007
                                 + 0.00370267301327923 w_k1_INTC_s010
                                 + 0.0163034780323046 w_k1_INTC_s011
                                 + 0.00558315441337217 w_k1_INTC_s012
                                 + 0.00808640217729662 w_k1_INTC_s013
                                 + 0.0184052151651434 w_k1_INTC_s014
                                 + 0.00635961592840019 w_k1_INTC_s015
                                 + 0.00682630323239598 w_k1_INTC_s016
                                 + 0.00983030351677147 w_k1_INTC_s017
                                 + 0.0370622878643815 w_k1_INTC_s100
                                 + 0.0825808888355165 w_k1_INTC_s101
                                 + 0.0740631715394997 w_k1_INTC_s102
                                 + 0.092133424705728 w_k1_INTC_s103
                                 + 0.0406746497765007 w_k1_INTC_s104
                                 + 0.0177152441342218 w_k1_INTC_s105
                                 + 0.0323912351823772 w_k1_INTC_s106
                                 + 0.0336043138001201 w_k1_INTC_s107
                                 + 0.0275413240788422 w_k1_INTC_s110
                                 + 0.0144888584874346 w_k1_INTC_s111
                                 + 0.033342755853985 w_k1_INTC_s112
                                 + 0.0165893013570862 w_k1_INTC_s113
                                 + 0.0127247220963493 w_k1_INTC_s114
                                 + 0.0373672950317984 w_k1_INTC_s115
                                 + 0.0193928514240841 w_k1_INTC_s116
                                 + 0.0538855936710567 w_k1_INTC_s117
                                 + 0.0368144894986809 w_k1_KO_s000
                                 + 0.0711199951416402 w_k1_KO_s001
                                 + 0.0271806214842563 w_k1_KO_s002
                                 + 0.0714237850792471 w_k1_KO_s003
                                 + 0.0259631912497283 w_k1_KO_s004
                                 + 0.0207023385150549 w_k1_KO_s005
                                 + 0.0307003922553855 w_k1_KO_s006
                                 + 0.0290108642230932 w_k1_KO_s007
                                 + 0.00342608811402232 w_k1_KO_s010
                                 + 0.0179113386154347 w_k1_KO_s011
                                 + 0.00573505866272868 w_k1_KO_s012
                                 + 0.00751467783402314 w_k1_KO_s013
                                 + 0.0155808629452655 w_k1_KO_s014
                                 + 0.00662512736668784 w_k1_KO_s015
                                 + 0.00635959644650201 w_k1_KO_s016
                                 + 0.0105642829164308 w_k1_KO_s017
                                 + 0.0342902746461001 w_k1_KO_s100
                                 + 0.0816776734662525 w_k1_KO_s101
                                 + 0.0751151880830826 w_k1_KO_s102
                                 + 0.0985259985091968 w_k1_KO_s103
                                 + 0.0426046415594408 w_k1_KO_s104
                                 + 0.0183547336401026 w_k1_KO_s105
                                 + 0.0316186778463962 w_k1_KO_s106
                                 + 0.0311802782525862 w_k1_KO_s107
                                 + 0.0314854561309853 w_k1_KO_s110
                                 + 0.0161127408862367 w_k1_KO_s111
                                 + 0.0365020342734343 w_k1_KO_s112
                                 + 0.0140197830132641 w_k1_KO_s113
                                 + 0.015044903647314 w_k1_KO_s114
                                 + 0.0322796473483354 w_k1_KO_s115
                                 + 0.0179349415815424 w_k1_KO_s116
                                 + 0.0589790541420548 w_k1_KO_s117
                                 + 0.0393955373863655 w_k1_MAT_s000
                                 + 0.0709583403197466 w_k1_MAT_s001
                                 + 0.0335036879011036 w_k1_MAT_s002
                                 + 0.0754238963155077 w_k1_MAT_s003
                                 + 0.0259064920847149 w_k1_MAT_s004
                                 + 0.0260769710662222 w_k1_MAT_s005
                                 + 0.0405460209212081 w_k1_MAT_s006
                                 + 0.0288764847403796 w_k1_MAT_s007
                                 + 0.00379512705065611 w_k1_MAT_s010
                                 + 0.0186131076731886 w_k1_MAT_s011
                                 + 0.00638237948888417 w_k1_MAT_s012
                                 + 0.00535034743447593 w_k1_MAT_s013
                                 + 0.0158703719695223 w_k1_MAT_s014
                                 + 0.00635175956852477 w_k1_MAT_s015
                                 + 0.00621830535858214 w_k1_MAT_s016
                                 + 0.0103680244351752 w_k1_MAT_s017
                                 + 0.0380953647491453 w_k1_MAT_s100
                                 + 0.0896559830067535 w_k1_MAT_s101
                                 + 0.0730498743651145 w_k1_MAT_s102
                                 + 0.0914832470910306 w_k1_MAT_s103
                                 + 0.0477443311677148 w_k1_MAT_s104
                                 + 0.0232812074952676 w_k1_MAT_s105
                                 + 0.0388588399924032 w_k1_MAT_s106
                                 + 0.0245264376560576 w_k1_MAT_s107
                                 + 0.0252979839282895 w_k1_MAT_s110
                                 + 0.017481003291754 w_k1_MAT_s111
                                 + 0.0341958604278061 w_k1_MAT_s112
                                 + 0.0129862394233499 w_k1_MAT_s113
                                 + 0.0179610528887064 w_k1_MAT_s114
                                 + 0.0350598900138244 w_k1_MAT_s115
                                 + 0.0192814311526438 w_k1_MAT_s116
                                 + 0.0757089246579715 w_k1_MAT_s117
                                 + 0.0348903121511658 w_k1_MSFT_s000
                                 + 0.0739231780819669 w_k1_MSFT_s001
                                 + 0.0290776406268425 w_k1_MSFT_s002
                                 + 0.0677931249723309 w_k1_MSFT_s003
                                 + 0.0260894366664829 w_k1_MSFT_s004
                                 + 0.0205026269843758 w_k1_MSFT_s005
                                 + 0.0335108809470662 w_k1_MSFT_s006
                                 + 0.026505395371765 w_k1_MSFT_s007
                                 + 0.00422604293036123 w_k1_MSFT_s010
                                 + 0.0145616822485721 w_k1_MSFT_s011
                                 + 0.00582884489999498 w_k1_MSFT_s012
                                 + 0.00678284762247009 w_k1_MSFT_s013
                                 + 0.0163307991725608 w_k1_MSFT_s014
                                 + 0.00702291802714183 w_k1_MSFT_s015
                                 + 0.00684502437296869 w_k1_MSFT_s016
                                 + 0.0091960934947295 w_k1_MSFT_s017
                                 + 0.0363100565102541 w_k1_MSFT_s100
                                 + 0.0842971293044838 w_k1_MSFT_s101
                                 + 0.0710402572423385 w_k1_MSFT_s102
                                 + 0.0901779962418075 w_k1_MSFT_s103
                                 + 0.0436903528692308 w_k1_MSFT_s104
                                 + 0.0184977633508625 w_k1_MSFT_s105
                                 + 0.0326122697414483 w_k1_MSFT_s106
                                 + 0.0323604019566954 w_k1_MSFT_s107
                                 + 0.0279651642174503 w_k1_MSFT_s110
                                 + 0.0148605794039925 w_k1_MSFT_s111
                                 + 0.0356615959337864 w_k1_MSFT_s112
                                 + 0.0141170627985206 w_k1_MSFT_s113
                                 + 0.0161137797399957 w_k1_MSFT_s114
                                 + 0.0310400642465728 w_k1_MSFT_s115
                                 + 0.0197313056889715 w_k1_MSFT_s116
                                 + 0.0602525507649743 w_k1_MSFT_s117
                                 + 0.0346629756227931 w_k1_PG_s000
                                 + 0.0784558119860344 w_k1_PG_s001
                                 + 0.0277955828597268 w_k1_PG_s002
                                 + 0.0647956998842546 w_k1_PG_s003
                                 + 0.0260431434438958 w_k1_PG_s004
                                 + 0.0209062813511227 w_k1_PG_s005
                                 + 0.0310002268693224 w_k1_PG_s006
                                 + 0.0297434738281698 w_k1_PG_s007
                                 + 0.00365233786100071 w_k1_PG_s010
                                 + 0.0170340769686332 w_k1_PG_s011
                                 + 0.00590934291259712 w_k1_PG_s012
                                 + 0.00680430707649785 w_k1_PG_s013
                                 + 0.0157043287911887 w_k1_PG_s014
                                 + 0.00622563161945121 w_k1_PG_s015
                                 + 0.0069571571513814 w_k1_PG_s016
                                 + 0.00919554899209064 w_k1_PG_s017
                                 + 0.0347041083901122 w_k1_PG_s100
                                 + 0.0720661443483837 w_k1_PG_s101
                                 + 0.0758317603796723 w_k1_PG_s102
                                 + 0.0897052794620647 w_k1_PG_s103
                                 + 0.0425858791007543 w_k1_PG_s104
                                 + 0.0192233194488557 w_k1_PG_s105
                                 + 0.0314883159452964 w_k1_PG_s106
                                 + 0.0307183025699321 w_k1_PG_s107
                                 + 0.0313260878717205 w_k1_PG_s110
                                 + 0.0160577440574534 w_k1_PG_s111
                                 + 0.0357980912168679 w_k1_PG_s112
                                 + 0.0129524752558478 w_k1_PG_s113
                                 + 0.014138353896156 w_k1_PG_s114
                                 + 0.0302737401531475 w_k1_PG_s115
                                 + 0.0189539420746355 w_k1_PG_s116
                                 + 0.0561602549624666 w_k1_PG_s117 >= 1.041
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
