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
 w_k1_ADM_s0_return_constraint:  1.0739 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.933426 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.905054 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.27282 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.14342 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.14571 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.20906 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.31279 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.981247 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.19902 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.679075 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.16425 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.10112 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.813471 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.1845 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.0063 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.04285 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.727426 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.0509 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.03524 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.0323 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.966231 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.752991 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.831496 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.616517 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.656509 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.02315 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.26815 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.60488 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.07334 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.06781 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.24784 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.737509 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.581858 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.21649 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.09949 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.634789 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.755053 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.21175 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.901228 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.82422 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.24911 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.976803 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.06349 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.31164 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.42713 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.85542 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.41274 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.672578 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.19367 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.863649 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.733913 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.09713 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.20023 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.47816 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.743101 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.04056 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.05886 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.07565 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.901583 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.881843 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.778463 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.558586 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.485956 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.03108 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.34475 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.57066 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.06589 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.10077 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.66091 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.741399 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.76785 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.33632 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.14508 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.623425 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.738333 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25697 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.898601 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.807204 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.31814 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.962995 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.0438 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.45432 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.33274 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.811258 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.28795 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.455384 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.06185 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.933389 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.623063 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.15717 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.849105 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.70859 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.676091 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.09716 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.03324 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.17137 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.774603 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.872775 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.745721 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.491389 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.438661 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.23815 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.72475 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 2.56223 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.20669 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.01854 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.70033 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.720164 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.565503 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.35904 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.39878 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.591672 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.883595 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.11756 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.948848 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.873398 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.23449 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.0719 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.02121 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.47524 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.10931 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.830129 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.2346 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.62075 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.01795 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.06402 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.722632 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.15798 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.879029 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.37154 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.707039 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.17449 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.998688 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.00956 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.826048 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.823529 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.833662 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.620569 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.58353 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.00586 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.37095 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.76888 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.1723 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.07282 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.38521 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.813289 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.660674 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.40013 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.16258 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.689852 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.806009 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.28975 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.871905 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.891692 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.18864 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.668114 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.12438 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.733264 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.53617 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.09783 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.898839 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.910184 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.0673 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.669329 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.684533 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.764833 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.30052 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.31284 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.07228 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.560342 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.833742 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.45322 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.799472 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.15889 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.812214 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.01422 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.659373 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.28066 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.35622 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.33873 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.945884 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.862849 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.55138 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.805767 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.0657 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.777412 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.18995 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.603048 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.12622 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.07309 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.00221 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.8703 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.21854 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.937072 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.877177 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.17738 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.28034 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.951064 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.20477 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.697295 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.00003 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.12558 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.920936 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.948928 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.812165 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.18578 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.780737 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.05313 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.02554 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.965735 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.845627 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.946636 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.901656 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.623967 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.741458 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.10847 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.28704 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.56811 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.11774 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.0841 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.57196 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.940965 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.845484 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.3654 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.07661 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.788344 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.02094 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19243 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.872298 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.878955 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.33388 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.896797 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.14343 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.99612 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.45774 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.04415 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.09428 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.65506 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.04803 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.84769 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.766009 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.869995 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.899154 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.14846 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.818344 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.764646 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.88444 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.15037 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.756849 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.970416 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.839044 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.736094 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.574345 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.21293 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.25427 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.48292 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.01309 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.955333 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.52652 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.735418 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.717531 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.943531 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.23643 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.589469 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.927086 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.42788 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.881103 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.782836 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.13 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.639872 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.1734 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.839305 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.69293 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.943735 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.14234 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.734592 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.34883 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.665917 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.738028 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.77478 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.35573 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.55797 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.945636 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.427735 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.78377 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.25769 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.780684 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.19273 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.03834 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.910682 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.440471 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.97787 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.57883 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.18084 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.950546 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.956396 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.41675 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.821654 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.02045 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.586899 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.1589 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.523327 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.922379 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09897 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.933595 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.891446 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.24893 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.06439 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.0314 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.22062 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.15401 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.844764 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.22166 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.658405 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.98407 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.08213 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.77467 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.05314 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.854985 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.17388 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.642542 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.01482 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.975323 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.956652 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.755532 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.850362 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.838793 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.655949 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.602472 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.03528 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.29439 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.69903 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.10165 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.07892 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.2547 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.839763 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.692902 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.26055 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.19567 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.689703 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.928377 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09777 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.970457 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.890716 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.10032 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.937207 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.03435 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.12039 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.16372 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.00467 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.08536 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.800643 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.09729 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.934511 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.892113 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.14436 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.956905 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.3019 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.888757 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.04146 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.01326 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.06165 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.922854 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.888597 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.871675 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.759055 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.694396 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.05113 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.26995 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.27709 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.983496 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.994182 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.29766 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.831176 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.804236 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.10383 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.13231 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.703797 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.946599 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.243999631305961 z_k1_s0000
                                 - 0.109000051264676 z_k1_s0001
                                 - 0.167000046766962 z_k1_s0002
                                 - 0.0810000449049468 z_k1_s0003
                                 - 0.0940000566933977 z_k1_s0004
                                 - 0.119000088878886 z_k1_s0005
                                 - 0.147000218938688 z_k1_s0006
                                 - 0.275999603231345 z_k1_s0007
                                 - 0.275000452052494 z_k1_s0010
                                 - 0.79400041274148 z_k1_s0011
                                 - 0.887000186084837 z_k1_s0012
                                 - 0.25800056861565 z_k1_s0013
                                 - 0.538002116668933 z_k1_s0014
                                 - 0.392000420471942 z_k1_s0015
                                 - 0.753002954432528 z_k1_s0016
                                 - 0.469999089316249 z_k1_s0017
                                 - 0.888002621320138 z_k1_s0020
                                 - 0.558001523018624 z_k1_s0021
                                 - 0.438000622472169 z_k1_s0022
                                 - 0.641001395542581 z_k1_s0023
                                 - 1.02000214183992 z_k1_s0024
                                 - 0.248000421674715 z_k1_s0025
                                 - 0.469001881107591 z_k1_s0026
                                 - 0.332000408896656 z_k1_s0027
                                 - 0.449999586660719 z_k1_s0030
                                 - 0.149999999020267 z_k1_s0031
                                 - 0.337000027894873 z_k1_s0032
                                 - 0.296001059634875 z_k1_s0033
                                 - 0.175000135657006 z_k1_s0034
                                 - 0.349999450513847 z_k1_s0035
                                 - 0.14300001000999 z_k1_s0036
                                 - 0.150999938821724 z_k1_s0037
                                 - 0.152000625988765 z_k1_s0040
                                 - 0.126000363543137 z_k1_s0041
                                 - 0.248000762402419 z_k1_s0042
                                 - 0.0600000606585048 z_k1_s0043
                                 - 0.114000469491574 z_k1_s0044
                                 - 0.0990003559268325 z_k1_s0045
                                 - 0.0630000833714531 z_k1_s0046
                                 - 0.122000398859282 z_k1_s0047
                                 - 0.180999680471325 z_k1_s0050
                                 - 0.312999199544952 z_k1_s0051
                                 - 0.851999051924407 z_k1_s0052
                                 - 0.388999495661674 z_k1_s0053
                                 - 0.243999246984246 z_k1_s0054
                                 - 0.424997965957966 z_k1_s0055
                                 - 0.33199871270053 z_k1_s0056
                                 - 0.468999301312115 z_k1_s0057
                                 - 0.207999494472902 z_k1_s0060
                                 - 0.392999474630696 z_k1_s0061
                                 - 0.487998870320936 z_k1_s0062
                                 - 0.127999756544737 z_k1_s0063
                                 - 0.565998497561114 z_k1_s0064
                                 - 0.597999096171074 z_k1_s0065
                                 - 0.297000609795375 z_k1_s0066
                                 - 0.252999640156951 z_k1_s0067
                                 - 0.613002131206931 z_k1_s0070
                                 - 0.155000389207674 z_k1_s0071
                                 - 0.168000392512102 z_k1_s0072
                                 - 0.300999911894351 z_k1_s0073
                                 - 0.221000591176854 z_k1_s0074
                                 - 0.321000945827272 z_k1_s0075
                                 - 0.642001891654543 z_k1_s0076
                                 - 0.254000825911839 z_k1_s0077
                                 - 0.297999930048517 z_k1_s0100
                                 - 0.360999148306679 z_k1_s0101
                                 - 0.13900006198644 z_k1_s0102
                                 - 0.534000290382468 z_k1_s0103
                                 - 0.167000046812471 z_k1_s0104
                                 - 0.309999984573992 z_k1_s0105
                                 - 0.203000210388895 z_k1_s0106
                                 - 0.123999993829597 z_k1_s0107
                                 - 0.167000157837846 z_k1_s0110
                                 - 0.0729999614788615 z_k1_s0111
                                 - 0.0849999823246909 z_k1_s0112
                                 - 0.0520000241986206 z_k1_s0113
                                 - 0.184999990706669 z_k1_s0114
                                 - 0.0579999354215746 z_k1_s0115
                                 - 0.120000208458295 z_k1_s0116
                                 - 0.251999644162733 z_k1_s0117
                                 - 0.459997572567488 z_k1_s0120
                                 - 0.31899913947575 z_k1_s0121
                                 - 0.110999766898878 z_k1_s0122
                                 - 0.330998605952894 z_k1_s0123
                                 - 0.48699872313371 z_k1_s0124
                                 - 0.204999270858321 z_k1_s0125
                                 - 0.242998600132758 z_k1_s0126
                                 - 0.194999323628593 z_k1_s0127
                                 - 0.269998931326448 z_k1_s0130
                                 - 0.192999764495807 z_k1_s0131
                                 - 0.233999866415496 z_k1_s0132
                                 - 0.446000876707084 z_k1_s0133
                                 - 0.428999622995203 z_k1_s0134
                                 - 0.448000868790983 z_k1_s0135
                                 - 0.323000042548523 z_k1_s0136
                                 - 0.299000137541727 z_k1_s0137
                                 - 0.125000126123461 z_k1_s0140
                                 - 0.338999325575678 z_k1_s0141
                                 - 0.123000239801388 z_k1_s0142
                                 - 0.0849999204840919 z_k1_s0143
                                 - 0.0819998844011566 z_k1_s0144
                                 - 0.0699999466692419 z_k1_s0145
                                 - 0.089999946189013 z_k1_s0146
                                 - 0.11899943415811 z_k1_s0147
                                 - 0.128999528128003 z_k1_s0150
                                 - 0.115999838244994 z_k1_s0151
                                 - 0.0909999387783154 z_k1_s0152
                                 - 0.261999087050614 z_k1_s0153
                                 - 0.0699999529063964 z_k1_s0154
                                 - 0.165999519978505 z_k1_s0155
                                 - 0.243999534474117 z_k1_s0156
                                 - 0.0939999032743474 z_k1_s0157
                                 - 0.453001926544915 z_k1_s0160
                                 - 0.618002544902203 z_k1_s0161
                                 - 0.162000263804793 z_k1_s0162
                                 - 0.790002102345653 z_k1_s0163
                                 - 0.193000151107921 z_k1_s0164
                                 - 0.335001827635399 z_k1_s0165
                                 - 0.277000322862313 z_k1_s0166
                                 - 0.318999307287765 z_k1_s0167
                                 - 0.649001143633654 z_k1_s0170
                                 - 0.645001239734523 z_k1_s0171
                                 - 0.282999895263381 z_k1_s0172
                                 - 0.716999509918889 z_k1_s0173
                                 - 0.972001751488307 z_k1_s0174
                                 - 0.203000143680794 z_k1_s0175
                                 - 0.388999858990301 z_k1_s0176
                                 - 0.326000117379012 z_k1_s0177
                                 - 0.134999497544355 z_k1_s1000
                                 - 0.370998954502798 z_k1_s1001
                                 - 0.182999215073866 z_k1_s1002
                                 - 0.11399955064422 z_k1_s1003
                                 - 0.354998565361526 z_k1_s1004
                                 - 0.133999553772709 z_k1_s1005
                                 - 0.145999362330561 z_k1_s1006
                                 - 0.172999777357407 z_k1_s1007
                                 - 0.198999393220057 z_k1_s1010
                                 - 0.19799946794686 z_k1_s1011
                                 - 0.556997675543586 z_k1_s1012
                                 - 0.463997347659804 z_k1_s1013
                                 - 0.209999444522399 z_k1_s1014
                                 - 0.550998123904407 z_k1_s1015
                                 - 0.255999209179101 z_k1_s1016
                                 - 0.475999361928767 z_k1_s1017
                                 - 0.102999857024299 z_k1_s1020
                                 - 0.0689999114919949 z_k1_s1021
                                 - 0.122000138144803 z_k1_s1022
                                 - 0.153999507823177 z_k1_s1023
                                 - 0.304999810362849 z_k1_s1024
                                 - 0.162999263670324 z_k1_s1025
                                 - 0.0849999173306769 z_k1_s1026
                                 - 0.0689999114919949 z_k1_s1027
                                 - 0.0889999119162234 z_k1_s1030
                                 - 0.0999999135125999 z_k1_s1031
                                 - 0.306000246408119 z_k1_s1032
                                 - 0.0839999717905459 z_k1_s1033
                                 - 0.0629999788429094 z_k1_s1034
                                 - 0.150000425768425 z_k1_s1035
                                 - 0.163000225655224 z_k1_s1036
                                 - 0.155999820639694 z_k1_s1037
                                 - 0.343998797278896 z_k1_s1040
                                 - 0.0789998040319664 z_k1_s1041
                                 - 0.30099941386842 z_k1_s1042
                                 - 0.197999329422127 z_k1_s1043
                                 - 0.39299974279099 z_k1_s1044
                                 - 0.250998829671905 z_k1_s1045
                                 - 0.200000471788515 z_k1_s1046
                                 - 0.0989998512108178 z_k1_s1047
                                 - 0.238999077780656 z_k1_s1050
                                 - 0.281999567576149 z_k1_s1051
                                 - 0.0709997227235515 z_k1_s1052
                                 - 0.222999742249694 z_k1_s1053
                                 - 0.301999132738726 z_k1_s1054
                                 - 0.197999494298721 z_k1_s1055
                                 - 0.0909997627847794 z_k1_s1056
                                 - 0.176999475979364 z_k1_s1057
                                 - 0.133999692089947 z_k1_s1060
                                 - 0.0729999638997003 z_k1_s1061
                                 - 0.0799998913969565 z_k1_s1062
                                 - 0.0709997146148665 z_k1_s1063
                                 - 0.108000231385756 z_k1_s1064
                                 - 0.0309999579163207 z_k1_s1065
                                 - 0.0370000127710699 z_k1_s1066
                                 - 0.0959999956763028 z_k1_s1067
                                 - 0.617000500276763 z_k1_s1070
                                 - 0.314000598811441 z_k1_s1071
                                 - 0.158999826105705 z_k1_s1072
                                 - 0.654000497716412 z_k1_s1073
                                 - 0.254999780678893 z_k1_s1074
                                 - 0.3109985822646 z_k1_s1075
                                 - 0.284999569170741 z_k1_s1076
                                 - 0.274999639673458 z_k1_s1077
                                 - 1.09800494983973 z_k1_s1100
                                 - 1.94200683650964 z_k1_s1101
                                 - 1.63500627931604 z_k1_s1102
                                 - 1.30200626633552 z_k1_s1103
                                 - 0.818002430882981 z_k1_s1104
                                 - 1.23699854884591 z_k1_s1105
                                 - 0.595001721574181 z_k1_s1106
                                 - 0.451000685482775 z_k1_s1107
                                 - 0.784001336594472 z_k1_s1110
                                 - 0.689001678221133 z_k1_s1111
                                 - 0.729001705052391 z_k1_s1112
                                 - 1.48800261952643 z_k1_s1113
                                 - 1.47000503955776 z_k1_s1114
                                 - 1.33199927207732 z_k1_s1115
                                 - 0.772001328545094 z_k1_s1116
                                 - 0.843000160117878 z_k1_s1117
                                 - 1.03799847967497 z_k1_s1120
                                 - 0.525998931053353 z_k1_s1121
                                 - 0.927997620003424 z_k1_s1122
                                 - 0.332999876281451 z_k1_s1123
                                 - 0.372999825398782 z_k1_s1124
                                 - 0.940997403546779 z_k1_s1125
                                 - 0.260999767950477 z_k1_s1126
                                 - 0.598000038983216 z_k1_s1127
                                 - 0.754000957102423 z_k1_s1130
                                 - 0.571999245733065 z_k1_s1131
                                 - 0.966998124447409 z_k1_s1132
                                 - 0.270000043243877 z_k1_s1133
                                 - 0.343999762535941 z_k1_s1134
                                 - 0.499001083210962 z_k1_s1135
                                 - 0.57399978347332 z_k1_s1136
                                 - 0.31300001356087 z_k1_s1137
                                 - 0.447999169124062 z_k1_s1140
                                 - 0.199000055875608 z_k1_s1141
                                 - 0.384000336534822 z_k1_s1142
                                 - 0.298000528805646 z_k1_s1143
                                 - 0.127999867478295 z_k1_s1144
                                 - 0.405000488759371 z_k1_s1145
                                 - 0.216000097538345 z_k1_s1146
                                 - 0.369000227803001 z_k1_s1147
                                 - 1.00999919716958 z_k1_s1150
                                 - 1.76400190549425 z_k1_s1151
                                 - 0.394000325017648 z_k1_s1152
                                 - 0.780998953730134 z_k1_s1153
                                 - 0.404999822182854 z_k1_s1154
                                 - 0.80400294307564 z_k1_s1155
                                 - 0.787001199820284 z_k1_s1156
                                 - 0.985997143809084 z_k1_s1157
                                 - 0.32000043594351 z_k1_s1160
                                 - 0.520000708408204 z_k1_s1161
                                 - 0.583001939711515 z_k1_s1162
                                 - 0.689999667251046 z_k1_s1163
                                 - 0.430000840349521 z_k1_s1164
                                 - 1.13200103304931 z_k1_s1165
                                 - 0.739000624931399 z_k1_s1166
                                 - 0.677002322320351 z_k1_s1167
                                 - 0.538998964083604 z_k1_s1170
                                 - 0.670001071391561 z_k1_s1171
                                 - 0.668997520941233 z_k1_s1172
                                 - 0.393999608790047 z_k1_s1173
                                 - 0.510999336320328 z_k1_s1174
                                 - 0.755000654136121 z_k1_s1175
                                 - 1.63499465725886 z_k1_s1176
                                 - 0.528999382124851 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.899 w_k1_ADM_s000
                                 + 0.701751 w_k1_AXP_s000
                                 + 0.850597 w_k1_DIS_s000
                                 + 0.854672 w_k1_FDX_s000
                                 + 0.702122 w_k1_IBM_s000
                                 + 1.10698 w_k1_INTC_s000
                                 + 0.878973 w_k1_KO_s000
                                 + 0.513925 w_k1_MAT_s000
                                 + 1.1188 w_k1_MSFT_s000
                                 + 0.918544 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.63666 w_k1_ADM_s000
                                 + 1.59137 w_k1_AXP_s000 + 1.6707 w_k1_DIS_s000
                                 + 1.381 w_k1_FDX_s000 + 1.52598 w_k1_IBM_s000
                                 + 1.48159 w_k1_INTC_s000
                                 + 1.57366 w_k1_KO_s000 + 1.74055 w_k1_MAT_s000
                                 + 1.3392 w_k1_MSFT_s000 + 1.37859 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.35361 w_k1_ADM_s000
                                 + 0.970195 w_k1_AXP_s000
                                 + 1.32265 w_k1_DIS_s000
                                 + 1.12739 w_k1_FDX_s000
                                 + 1.07886 w_k1_IBM_s000
                                 + 1.40793 w_k1_INTC_s000
                                 + 1.15586 w_k1_KO_s000
                                 + 0.784874 w_k1_MAT_s000
                                 + 1.47922 w_k1_MSFT_s000
                                 + 0.999684 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.07999 w_k1_ADM_s000
                                 + 1.28961 w_k1_AXP_s000
                                 + 1.10324 w_k1_DIS_s000
                                 + 0.997344 w_k1_FDX_s000
                                 + 1.39663 w_k1_IBM_s000
                                 + 1.14113 w_k1_INTC_s000
                                 + 1.34954 w_k1_KO_s000 + 1.65625 w_k1_MAT_s000
                                 + 1.07707 w_k1_MSFT_s000 + 1.1488 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.04024 w_k1_ADM_s000
                                 + 1.05019 w_k1_AXP_s000
                                 + 1.40713 w_k1_DIS_s000 + 1.1115 w_k1_FDX_s000
                                 + 1.29667 w_k1_IBM_s000
                                 + 1.5225 w_k1_INTC_s000 + 1.33794 w_k1_KO_s000
                                 + 1.18518 w_k1_MAT_s000
                                 + 1.33853 w_k1_MSFT_s000
                                 + 1.29108 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.63988 w_k1_ADM_s000
                                 + 1.40419 w_k1_AXP_s000
                                 + 1.18838 w_k1_DIS_s000
                                 + 1.25726 w_k1_FDX_s000
                                 + 0.94572 w_k1_IBM_s000
                                 + 1.45318 w_k1_INTC_s000
                                 + 1.09663 w_k1_KO_s000
                                 + 0.868787 w_k1_MAT_s000
                                 + 1.2044 w_k1_MSFT_s000 + 1.18393 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.999553 w_k1_ADM_s000
                                 + 0.672836 w_k1_AXP_s000
                                 + 0.801217 w_k1_DIS_s000
                                 + 0.754856 w_k1_FDX_s000
                                 + 1.04128 w_k1_IBM_s000
                                 + 0.871705 w_k1_INTC_s000
                                 + 1.14696 w_k1_KO_s000 + 1.21617 w_k1_MAT_s000
                                 + 0.90474 w_k1_MSFT_s000
                                 + 0.998911 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.19857 w_k1_ADM_s000
                                 + 1.08093 w_k1_AXP_s000
                                 + 0.79562 w_k1_DIS_s000
                                 + 0.948252 w_k1_FDX_s000
                                 + 0.810347 w_k1_IBM_s000
                                 + 1.01999 w_k1_INTC_s000
                                 + 0.934234 w_k1_KO_s000
                                 + 0.945453 w_k1_MAT_s000
                                 + 0.918919 w_k1_MSFT_s000
                                 + 0.95301 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.2069 w_k1_ADM_s001
                                 + 1.34563 w_k1_AXP_s001
                                 + 1.83179 w_k1_DIS_s001
                                 + 1.56306 w_k1_FDX_s001
                                 + 0.84381 w_k1_IBM_s001
                                 + 1.2647 w_k1_INTC_s001 + 1.15141 w_k1_KO_s001
                                 + 1.11662 w_k1_MAT_s001
                                 + 1.62211 w_k1_MSFT_s001
                                 + 1.20135 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.717355 w_k1_ADM_s001
                                 + 0.721615 w_k1_AXP_s001
                                 + 0.741235 w_k1_DIS_s001
                                 + 0.818437 w_k1_FDX_s001
                                 + 0.776251 w_k1_IBM_s001
                                 + 0.886526 w_k1_INTC_s001
                                 + 0.739935 w_k1_KO_s001
                                 + 0.603419 w_k1_MAT_s001
                                 + 0.921594 w_k1_MSFT_s001
                                 + 0.841068 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.635659 w_k1_ADM_s001
                                 + 0.713589 w_k1_AXP_s001
                                 + 0.753589 w_k1_DIS_s001
                                 + 0.7211 w_k1_FDX_s001 + 1.22853 w_k1_IBM_s001
                                 + 0.881832 w_k1_INTC_s001
                                 + 0.948884 w_k1_KO_s001
                                 + 0.941744 w_k1_MAT_s001
                                 + 0.833855 w_k1_MSFT_s001
                                 + 0.955256 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.985141 w_k1_ADM_s001
                                 + 1.20906 w_k1_AXP_s001
                                 + 1.45759 w_k1_DIS_s001
                                 + 1.37703 w_k1_FDX_s001
                                 + 1.03192 w_k1_IBM_s001
                                 + 1.03752 w_k1_INTC_s001
                                 + 1.01505 w_k1_KO_s001 + 1.07477 w_k1_MAT_s001
                                 + 1.18221 w_k1_MSFT_s001
                                 + 1.09024 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.866476 w_k1_ADM_s001
                                 + 1.06939 w_k1_AXP_s001
                                 + 1.03212 w_k1_DIS_s001
                                 + 1.05961 w_k1_FDX_s001
                                 + 1.02816 w_k1_IBM_s001
                                 + 1.07306 w_k1_INTC_s001
                                 + 0.959011 w_k1_KO_s001
                                 + 0.891255 w_k1_MAT_s001
                                 + 0.98291 w_k1_MSFT_s001
                                 + 1.07063 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.872687 w_k1_ADM_s001
                                 + 0.823231 w_k1_AXP_s001
                                 + 0.789262 w_k1_DIS_s001
                                 + 0.895898 w_k1_FDX_s001
                                 + 0.812795 w_k1_IBM_s001
                                 + 0.848201 w_k1_INTC_s001
                                 + 0.888354 w_k1_KO_s001
                                 + 0.95128 w_k1_MAT_s001
                                 + 0.930342 w_k1_MSFT_s001
                                 + 0.989295 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.11979 w_k1_ADM_s001
                                 + 1.12665 w_k1_AXP_s001
                                 + 1.21881 w_k1_DIS_s001
                                 + 1.08795 w_k1_FDX_s001
                                 + 1.42849 w_k1_IBM_s001
                                 + 0.954715 w_k1_INTC_s001
                                 + 1.31803 w_k1_KO_s001 + 1.46891 w_k1_MAT_s001
                                 + 1.05802 w_k1_MSFT_s001
                                 + 1.12178 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.11137 w_k1_ADM_s001
                                 + 0.890254 w_k1_AXP_s001
                                 + 1.19601 w_k1_DIS_s001
                                 + 1.16172 w_k1_FDX_s001
                                 + 0.721531 w_k1_IBM_s001
                                 + 0.86267 w_k1_INTC_s001
                                 + 0.85615 w_k1_KO_s001
                                 + 0.742601 w_k1_MAT_s001
                                 + 1.17491 w_k1_MSFT_s001
                                 + 1.07655 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.16425 w_k1_ADM_s002
                                 + 1.29821 w_k1_AXP_s002 + 1.3978 w_k1_DIS_s002
                                 + 1.25623 w_k1_FDX_s002
                                 + 1.09088 w_k1_IBM_s002
                                 + 1.04719 w_k1_INTC_s002
                                 + 1.10379 w_k1_KO_s002 + 1.1765 w_k1_MAT_s002
                                 + 1.14962 w_k1_MSFT_s002
                                 + 1.08105 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.0255 w_k1_ADM_s002
                                 + 0.858487 w_k1_AXP_s002
                                 + 0.932059 w_k1_DIS_s002
                                 + 1.11789 w_k1_FDX_s002
                                 + 0.697355 w_k1_IBM_s002
                                 + 0.996043 w_k1_INTC_s002
                                 + 0.750666 w_k1_KO_s002
                                 + 0.701561 w_k1_MAT_s002
                                 + 1.01991 w_k1_MSFT_s002
                                 + 0.989973 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.821299 w_k1_ADM_s002
                                 + 1.24762 w_k1_AXP_s002
                                 + 1.37603 w_k1_DIS_s002
                                 + 1.11512 w_k1_FDX_s002
                                 + 1.65565 w_k1_IBM_s002
                                 + 1.17842 w_k1_INTC_s002 + 1.3295 w_k1_KO_s002
                                 + 1.57166 w_k1_MAT_s002
                                 + 1.18275 w_k1_MSFT_s002
                                 + 1.21712 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.843833 w_k1_ADM_s002
                                 + 0.693286 w_k1_AXP_s002
                                 + 0.944094 w_k1_DIS_s002
                                 + 0.937018 w_k1_FDX_s002
                                 + 0.810624 w_k1_IBM_s002
                                 + 0.850372 w_k1_INTC_s002
                                 + 0.781054 w_k1_KO_s002
                                 + 0.758315 w_k1_MAT_s002
                                 + 1.04155 w_k1_MSFT_s002
                                 + 0.845562 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.812268 w_k1_ADM_s002
                                 + 0.861425 w_k1_AXP_s002
                                 + 0.771105 w_k1_DIS_s002
                                 + 0.853835 w_k1_FDX_s002
                                 + 1.10045 w_k1_IBM_s002
                                 + 0.849122 w_k1_INTC_s002
                                 + 0.827441 w_k1_KO_s002
                                 + 1.40042 w_k1_MAT_s002
                                 + 0.884027 w_k1_MSFT_s002
                                 + 0.966037 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.14027 w_k1_ADM_s002
                                 + 1.31803 w_k1_AXP_s002
                                 + 1.47167 w_k1_DIS_s002
                                 + 1.49159 w_k1_FDX_s002
                                 + 0.841839 w_k1_IBM_s002
                                 + 1.3353 w_k1_INTC_s002
                                 + 0.925325 w_k1_KO_s002
                                 + 0.903163 w_k1_MAT_s002
                                 + 1.35104 w_k1_MSFT_s002
                                 + 1.17883 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.77037 w_k1_ADM_s002
                                 + 0.849725 w_k1_AXP_s002
                                 + 0.596873 w_k1_DIS_s002
                                 + 0.789025 w_k1_FDX_s002
                                 + 0.800104 w_k1_IBM_s002
                                 + 0.996767 w_k1_INTC_s002
                                 + 0.745851 w_k1_KO_s002
                                 + 0.793864 w_k1_MAT_s002
                                 + 0.831086 w_k1_MSFT_s002
                                 + 0.79542 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.03837 w_k1_ADM_s002
                                 + 1.0261 w_k1_AXP_s002 + 1.08228 w_k1_DIS_s002
                                 + 1.02703 w_k1_FDX_s002
                                 + 1.33481 w_k1_IBM_s002
                                 + 0.985061 w_k1_INTC_s002
                                 + 0.961019 w_k1_KO_s002 + 1.3833 w_k1_MAT_s002
                                 + 1.03603 w_k1_MSFT_s002
                                 + 0.963971 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.948037 w_k1_ADM_s003
                                 + 0.821366 w_k1_AXP_s003
                                 + 0.655587 w_k1_DIS_s003
                                 + 0.807459 w_k1_FDX_s003
                                 + 0.900835 w_k1_IBM_s003
                                 + 0.883588 w_k1_INTC_s003
                                 + 0.876142 w_k1_KO_s003
                                 + 0.740699 w_k1_MAT_s003
                                 + 0.77959 w_k1_MSFT_s003
                                 + 0.923801 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.3516 w_k1_ADM_s003
                                 + 1.32972 w_k1_AXP_s003
                                 + 1.09197 w_k1_DIS_s003
                                 + 1.23138 w_k1_FDX_s003
                                 + 1.03404 w_k1_IBM_s003
                                 + 1.06929 w_k1_INTC_s003
                                 + 1.09267 w_k1_KO_s003 + 1.02688 w_k1_MAT_s003
                                 + 0.95262 w_k1_MSFT_s003 + 1.0504 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.98069 w_k1_ADM_s003
                                 + 1.26167 w_k1_AXP_s003
                                 + 0.754903 w_k1_DIS_s003
                                 + 0.995813 w_k1_FDX_s003
                                 + 1.1445 w_k1_IBM_s003 + 1.1003 w_k1_INTC_s003
                                 + 0.941482 w_k1_KO_s003
                                 + 1.41729 w_k1_MAT_s003
                                 + 0.823513 w_k1_MSFT_s003
                                 + 1.08923 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.953657 w_k1_ADM_s003
                                 + 0.997599 w_k1_AXP_s003
                                 + 1.20057 w_k1_DIS_s003
                                 + 1.11205 w_k1_FDX_s003
                                 + 1.07794 w_k1_IBM_s003
                                 + 0.912159 w_k1_INTC_s003
                                 + 0.96577 w_k1_KO_s003 + 1.17105 w_k1_MAT_s003
                                 + 0.95328 w_k1_MSFT_s003
                                 + 1.06585 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.54342 w_k1_ADM_s003
                                 + 1.18685 w_k1_AXP_s003
                                 + 1.46796 w_k1_DIS_s003
                                 + 1.33515 w_k1_FDX_s003
                                 + 1.10881 w_k1_IBM_s003
                                 + 1.4472 w_k1_INTC_s003 + 1.4236 w_k1_KO_s003
                                 + 0.904418 w_k1_MAT_s003
                                 + 1.38719 w_k1_MSFT_s003
                                 + 1.29371 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.981498 w_k1_ADM_s003
                                 + 0.886131 w_k1_AXP_s003
                                 + 0.882733 w_k1_DIS_s003
                                 + 0.99916 w_k1_FDX_s003
                                 + 0.784853 w_k1_IBM_s003
                                 + 0.962435 w_k1_INTC_s003
                                 + 0.766733 w_k1_KO_s003
                                 + 1.03136 w_k1_MAT_s003
                                 + 0.961978 w_k1_MSFT_s003
                                 + 1.00557 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.963969 w_k1_ADM_s003
                                 + 1.18483 w_k1_AXP_s003
                                 + 1.11299 w_k1_DIS_s003
                                 + 1.04082 w_k1_FDX_s003 + 1.5596 w_k1_IBM_s003
                                 + 0.843466 w_k1_INTC_s003
                                 + 1.17658 w_k1_KO_s003 + 1.57455 w_k1_MAT_s003
                                 + 0.821964 w_k1_MSFT_s003
                                 + 1.06649 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.0062 w_k1_ADM_s003
                                 + 1.87571 w_k1_AXP_s003
                                 + 1.58227 w_k1_DIS_s003
                                 + 1.40197 w_k1_FDX_s003
                                 + 1.43584 w_k1_IBM_s003
                                 + 1.36469 w_k1_INTC_s003
                                 + 1.34984 w_k1_KO_s003 + 2.10507 w_k1_MAT_s003
                                 + 1.23437 w_k1_MSFT_s003
                                 + 1.29512 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.15698 w_k1_ADM_s004
                                 + 0.816503 w_k1_AXP_s004
                                 + 0.804246 w_k1_DIS_s004
                                 + 0.870443 w_k1_FDX_s004
                                 + 0.745197 w_k1_IBM_s004
                                 + 0.918383 w_k1_INTC_s004
                                 + 0.922915 w_k1_KO_s004
                                 + 0.759751 w_k1_MAT_s004
                                 + 0.866024 w_k1_MSFT_s004
                                 + 0.930713 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.28677 w_k1_ADM_s004
                                 + 1.4087 w_k1_AXP_s004 + 1.22397 w_k1_DIS_s004
                                 + 1.25296 w_k1_FDX_s004
                                 + 0.906361 w_k1_IBM_s004
                                 + 1.52123 w_k1_INTC_s004
                                 + 1.17197 w_k1_KO_s004
                                 + 0.981488 w_k1_MAT_s004
                                 + 1.15548 w_k1_MSFT_s004 + 1.1497 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.97558 w_k1_ADM_s004
                                 + 1.08893 w_k1_AXP_s004
                                 + 1.29285 w_k1_DIS_s004
                                 + 1.04851 w_k1_FDX_s004
                                 + 1.11315 w_k1_IBM_s004
                                 + 1.2655 w_k1_INTC_s004 + 1.1435 w_k1_KO_s004
                                 + 1.08749 w_k1_MAT_s004
                                 + 1.02773 w_k1_MSFT_s004
                                 + 1.04941 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.42884 w_k1_ADM_s004
                                 + 1.61182 w_k1_AXP_s004
                                 + 2.11524 w_k1_DIS_s004 + 1.5163 w_k1_FDX_s004
                                 + 1.34819 w_k1_IBM_s004
                                 + 1.60072 w_k1_INTC_s004
                                 + 1.51205 w_k1_KO_s004 + 1.5698 w_k1_MAT_s004
                                 + 1.35401 w_k1_MSFT_s004 + 1.3373 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.11746 w_k1_ADM_s004
                                 + 0.779605 w_k1_AXP_s004
                                 + 0.839543 w_k1_DIS_s004
                                 + 0.922694 w_k1_FDX_s004
                                 + 0.636512 w_k1_IBM_s004
                                 + 1.01827 w_k1_INTC_s004
                                 + 0.897489 w_k1_KO_s004
                                 + 0.853413 w_k1_MAT_s004
                                 + 0.959873 w_k1_MSFT_s004
                                 + 0.896932 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.62854 w_k1_ADM_s004
                                 + 1.2887 w_k1_AXP_s004 + 1.4767 w_k1_DIS_s004
                                 + 1.05048 w_k1_FDX_s004
                                 + 1.55459 w_k1_IBM_s004
                                 + 1.13503 w_k1_INTC_s004
                                 + 1.43914 w_k1_KO_s004 + 1.64123 w_k1_MAT_s004
                                 + 1.01871 w_k1_MSFT_s004
                                 + 1.17421 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.27263 w_k1_ADM_s004
                                 + 1.20215 w_k1_AXP_s004
                                 + 0.997891 w_k1_DIS_s004
                                 + 0.954584 w_k1_FDX_s004
                                 + 1.21506 w_k1_IBM_s004
                                 + 1.12078 w_k1_INTC_s004
                                 + 1.13736 w_k1_KO_s004 + 1.46991 w_k1_MAT_s004
                                 + 0.915174 w_k1_MSFT_s004
                                 + 1.04961 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.93823 w_k1_ADM_s004
                                 + 0.672442 w_k1_AXP_s004
                                 + 0.671599 w_k1_DIS_s004
                                 + 0.678384 w_k1_FDX_s004
                                 + 0.886062 w_k1_IBM_s004
                                 + 0.739214 w_k1_INTC_s004
                                 + 0.813229 w_k1_KO_s004 + 1.0089 w_k1_MAT_s004
                                 + 0.754768 w_k1_MSFT_s004
                                 + 0.826339 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.920646 w_k1_ADM_s005
                                 + 0.773107 w_k1_AXP_s005
                                 + 0.966357 w_k1_DIS_s005
                                 + 0.948848 w_k1_FDX_s005
                                 + 0.855401 w_k1_IBM_s005
                                 + 0.873412 w_k1_INTC_s005
                                 + 0.917863 w_k1_KO_s005
                                 + 0.803959 w_k1_MAT_s005
                                 + 0.962381 w_k1_MSFT_s005
                                 + 0.927681 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.941466 w_k1_ADM_s005
                                 + 0.806145 w_k1_AXP_s005
                                 + 0.811148 w_k1_DIS_s005
                                 + 1.0497 w_k1_FDX_s005
                                 + 0.698338 w_k1_IBM_s005
                                 + 1.14963 w_k1_INTC_s005
                                 + 0.759628 w_k1_KO_s005
                                 + 0.49744 w_k1_MAT_s005
                                 + 0.968002 w_k1_MSFT_s005
                                 + 0.859699 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.0481 w_k1_ADM_s005
                                 + 1.01338 w_k1_AXP_s005
                                 + 1.15955 w_k1_DIS_s005
                                 + 1.15588 w_k1_FDX_s005
                                 + 1.02094 w_k1_IBM_s005
                                 + 1.21319 w_k1_INTC_s005 + 1.1383 w_k1_KO_s005
                                 + 0.786231 w_k1_MAT_s005
                                 + 1.23905 w_k1_MSFT_s005
                                 + 1.00431 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.21903 w_k1_ADM_s005
                                 + 1.09346 w_k1_AXP_s005
                                 + 1.66235 w_k1_DIS_s005
                                 + 1.41323 w_k1_FDX_s005
                                 + 1.13666 w_k1_IBM_s005
                                 + 1.63274 w_k1_INTC_s005
                                 + 1.46834 w_k1_KO_s005 + 1.14795 w_k1_MAT_s005
                                 + 1.4332 w_k1_MSFT_s005 + 1.19911 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.654316 w_k1_ADM_s005
                                 + 0.656134 w_k1_AXP_s005
                                 + 0.663587 w_k1_DIS_s005
                                 + 0.760661 w_k1_FDX_s005
                                 + 0.702443 w_k1_IBM_s005
                                 + 0.88346 w_k1_INTC_s005
                                 + 0.735376 w_k1_KO_s005
                                 + 0.592707 w_k1_MAT_s005
                                 + 0.772602 w_k1_MSFT_s005
                                 + 0.783709 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.686212 w_k1_ADM_s005
                                 + 0.805344 w_k1_AXP_s005
                                 + 0.746832 w_k1_DIS_s005
                                 + 0.824748 w_k1_FDX_s005
                                 + 1.1058 w_k1_IBM_s005
                                 + 1.07726 w_k1_INTC_s005
                                 + 0.847928 w_k1_KO_s005
                                 + 0.845003 w_k1_MAT_s005
                                 + 0.921417 w_k1_MSFT_s005
                                 + 0.783258 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.921269 w_k1_ADM_s005
                                 + 0.923141 w_k1_AXP_s005
                                 + 0.723963 w_k1_DIS_s005
                                 + 0.892454 w_k1_FDX_s005
                                 + 0.91985 w_k1_IBM_s005
                                 + 1.05076 w_k1_INTC_s005
                                 + 0.928987 w_k1_KO_s005
                                 + 0.753125 w_k1_MAT_s005
                                 + 0.869208 w_k1_MSFT_s005
                                 + 0.877482 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.763683 w_k1_ADM_s005
                                 + 1.38435 w_k1_AXP_s005
                                 + 1.02617 w_k1_DIS_s005
                                 + 1.06685 w_k1_FDX_s005
                                 + 1.47396 w_k1_IBM_s005
                                 + 1.18231 w_k1_INTC_s005
                                 + 1.16451 w_k1_KO_s005 + 1.55837 w_k1_MAT_s005
                                 + 1.03821 w_k1_MSFT_s005
                                 + 1.09406 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.18573 w_k1_ADM_s006
                                 + 0.970163 w_k1_AXP_s006
                                 + 1.65804 w_k1_DIS_s006
                                 + 1.17407 w_k1_FDX_s006
                                 + 1.23026 w_k1_IBM_s006
                                 + 0.956246 w_k1_INTC_s006
                                 + 1.21234 w_k1_KO_s006 + 1.44196 w_k1_MAT_s006
                                 + 1.18906 w_k1_MSFT_s006
                                 + 1.21048 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.887707 w_k1_ADM_s006
                                 + 0.713998 w_k1_AXP_s006
                                 + 0.723242 w_k1_DIS_s006
                                 + 0.805159 w_k1_FDX_s006
                                 + 0.608122 w_k1_IBM_s006
                                 + 0.736219 w_k1_INTC_s006
                                 + 0.68152 w_k1_KO_s006
                                 + 0.465503 w_k1_MAT_s006
                                 + 0.869758 w_k1_MSFT_s006
                                 + 0.9561 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.0288 w_k1_ADM_s006
                                 + 1.18841 w_k1_AXP_s006
                                 + 0.881863 w_k1_DIS_s006
                                 + 1.06031 w_k1_FDX_s006
                                 + 0.854982 w_k1_IBM_s006
                                 + 0.948109 w_k1_INTC_s006
                                 + 0.817855 w_k1_KO_s006
                                 + 0.676276 w_k1_MAT_s006
                                 + 0.866495 w_k1_MSFT_s006
                                 + 1.01852 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.25174 w_k1_ADM_s006
                                 + 1.84161 w_k1_AXP_s006
                                 + 1.94802 w_k1_DIS_s006 + 1.4478 w_k1_FDX_s006
                                 + 1.3041 w_k1_IBM_s006
                                 + 1.32333 w_k1_INTC_s006
                                 + 1.61944 w_k1_KO_s006 + 1.4068 w_k1_MAT_s006
                                 + 1.3054 w_k1_MSFT_s006 + 1.28395 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.831984 w_k1_ADM_s006
                                 + 1.20671 w_k1_AXP_s006
                                 + 1.06723 w_k1_DIS_s006
                                 + 1.01565 w_k1_FDX_s006
                                 + 1.10651 w_k1_IBM_s006
                                 + 0.963201 w_k1_INTC_s006
                                 + 0.986702 w_k1_KO_s006
                                 + 1.45921 w_k1_MAT_s006
                                 + 1.03287 w_k1_MSFT_s006
                                 + 1.10859 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.840663 w_k1_ADM_s006
                                 + 0.848956 w_k1_AXP_s006
                                 + 0.707095 w_k1_DIS_s006
                                 + 0.814807 w_k1_FDX_s006
                                 + 0.95867 w_k1_IBM_s006
                                 + 0.768129 w_k1_INTC_s006
                                 + 0.892408 w_k1_KO_s006
                                 + 1.32839 w_k1_MAT_s006
                                 + 0.812217 w_k1_MSFT_s006
                                 + 0.965463 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.08632 w_k1_ADM_s006
                                 + 1.22372 w_k1_AXP_s006
                                 + 1.29113 w_k1_DIS_s006
                                 + 1.25383 w_k1_FDX_s006
                                 + 0.802615 w_k1_IBM_s006
                                 + 0.970204 w_k1_INTC_s006
                                 + 1.05454 w_k1_KO_s006 + 1.04123 w_k1_MAT_s006
                                 + 1.22992 w_k1_MSFT_s006
                                 + 1.12197 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.855608 w_k1_ADM_s006
                                 + 1.13927 w_k1_AXP_s006
                                 + 0.992729 w_k1_DIS_s006
                                 + 0.904048 w_k1_FDX_s006
                                 + 1.34635 w_k1_IBM_s006
                                 + 0.89797 w_k1_INTC_s006
                                 + 1.00288 w_k1_KO_s006 + 1.64784 w_k1_MAT_s006
                                 + 0.93979 w_k1_MSFT_s006
                                 + 0.995069 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.825649 w_k1_ADM_s007
                                 + 0.74145 w_k1_AXP_s007
                                 + 0.939061 w_k1_DIS_s007
                                 + 0.929288 w_k1_FDX_s007
                                 + 1.0629 w_k1_IBM_s007
                                 + 0.859292 w_k1_INTC_s007
                                 + 0.921972 w_k1_KO_s007
                                 + 0.915029 w_k1_MAT_s007
                                 + 0.900288 w_k1_MSFT_s007
                                 + 0.940994 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.53542 w_k1_ADM_s007
                                 + 1.33586 w_k1_AXP_s007
                                 + 1.92797 w_k1_DIS_s007 + 1.5085 w_k1_FDX_s007
                                 + 1.32149 w_k1_IBM_s007
                                 + 0.976511 w_k1_INTC_s007
                                 + 1.5935 w_k1_KO_s007 + 1.36159 w_k1_MAT_s007
                                 + 1.38657 w_k1_MSFT_s007
                                 + 1.25368 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.913019 w_k1_ADM_s007
                                 + 0.707434 w_k1_AXP_s007
                                 + 0.711721 w_k1_DIS_s007
                                 + 0.977722 w_k1_FDX_s007
                                 + 0.779387 w_k1_IBM_s007
                                 + 1.03469 w_k1_INTC_s007
                                 + 0.734869 w_k1_KO_s007
                                 + 0.618899 w_k1_MAT_s007
                                 + 0.916083 w_k1_MSFT_s007
                                 + 0.837705 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.895954 w_k1_ADM_s007
                                 + 0.947072 w_k1_AXP_s007
                                 + 1.10064 w_k1_DIS_s007
                                 + 1.07146 w_k1_FDX_s007
                                 + 0.905265 w_k1_IBM_s007
                                 + 0.964696 w_k1_INTC_s007
                                 + 1.00941 w_k1_KO_s007
                                 + 0.971959 w_k1_MAT_s007
                                 + 1.13253 w_k1_MSFT_s007
                                 + 0.99556 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.09305 w_k1_ADM_s007
                                 + 0.735462 w_k1_AXP_s007
                                 + 0.966747 w_k1_DIS_s007
                                 + 1.08257 w_k1_FDX_s007
                                 + 0.597395 w_k1_IBM_s007
                                 + 0.999779 w_k1_INTC_s007
                                 + 0.860173 w_k1_KO_s007
                                 + 0.373275 w_k1_MAT_s007
                                 + 1.19881 w_k1_MSFT_s007
                                 + 0.926855 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.993835 w_k1_ADM_s007
                                 + 1.19868 w_k1_AXP_s007
                                 + 1.11239 w_k1_DIS_s007
                                 + 1.25695 w_k1_FDX_s007
                                 + 0.807862 w_k1_IBM_s007
                                 + 1.16567 w_k1_INTC_s007
                                 + 0.964173 w_k1_KO_s007
                                 + 0.670005 w_k1_MAT_s007
                                 + 1.20817 w_k1_MSFT_s007
                                 + 0.932303 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.868271 w_k1_ADM_s007
                                 + 1.0102 w_k1_AXP_s007 + 1.33261 w_k1_DIS_s007
                                 + 1.173 w_k1_FDX_s007 + 1.40026 w_k1_IBM_s007
                                 + 0.921895 w_k1_INTC_s007
                                 + 1.07506 w_k1_KO_s007 + 1.22492 w_k1_MAT_s007
                                 + 1.13102 w_k1_MSFT_s007
                                 + 1.05367 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.738391 w_k1_ADM_s007
                                 + 0.622692 w_k1_AXP_s007
                                 + 0.636942 w_k1_DIS_s007
                                 + 0.740847 w_k1_FDX_s007
                                 + 0.750521 w_k1_IBM_s007
                                 + 0.81521 w_k1_INTC_s007
                                 + 0.709729 w_k1_KO_s007
                                 + 0.583408 w_k1_MAT_s007
                                 + 0.821677 w_k1_MSFT_s007
                                 + 0.657004 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.687473 w_k1_ADM_s010
                                 + 0.56851 w_k1_AXP_s010
                                 + 0.600377 w_k1_DIS_s010
                                 + 0.717275 w_k1_FDX_s010
                                 + 0.829967 w_k1_IBM_s010
                                 + 0.884586 w_k1_INTC_s010
                                 + 0.733924 w_k1_KO_s010
                                 + 0.617182 w_k1_MAT_s010
                                 + 0.771591 w_k1_MSFT_s010
                                 + 0.73867 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.768028 w_k1_ADM_s010
                                 + 0.786532 w_k1_AXP_s010
                                 + 0.73574 w_k1_DIS_s010
                                 + 0.861416 w_k1_FDX_s010
                                 + 0.8028 w_k1_IBM_s010
                                 + 1.11248 w_k1_INTC_s010
                                 + 0.844647 w_k1_KO_s010
                                 + 0.633721 w_k1_MAT_s010
                                 + 0.895024 w_k1_MSFT_s010
                                 + 0.840521 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.775095 w_k1_ADM_s010
                                 + 1.10402 w_k1_AXP_s010
                                 + 1.44011 w_k1_DIS_s010
                                 + 1.31303 w_k1_FDX_s010
                                 + 0.802258 w_k1_IBM_s010
                                 + 1.23601 w_k1_INTC_s010
                                 + 1.03655 w_k1_KO_s010 + 1.12733 w_k1_MAT_s010
                                 + 1.18417 w_k1_MSFT_s010
                                 + 1.00011 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.0282 w_k1_ADM_s010
                                 + 1.27759 w_k1_AXP_s010
                                 + 1.35784 w_k1_DIS_s010
                                 + 1.15621 w_k1_FDX_s010
                                 + 1.42651 w_k1_IBM_s010
                                 + 1.46478 w_k1_INTC_s010
                                 + 1.47292 w_k1_KO_s010 + 1.51908 w_k1_MAT_s010
                                 + 1.21447 w_k1_MSFT_s010
                                 + 1.01959 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.79192 w_k1_ADM_s010
                                 + 0.738668 w_k1_AXP_s010
                                 + 1.17538 w_k1_DIS_s010
                                 + 0.932197 w_k1_FDX_s010
                                 + 0.905204 w_k1_IBM_s010
                                 + 1.25924 w_k1_INTC_s010
                                 + 0.996604 w_k1_KO_s010
                                 + 0.894718 w_k1_MAT_s010
                                 + 1.17919 w_k1_MSFT_s010
                                 + 0.919684 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.967274 w_k1_ADM_s010
                                 + 1.16181 w_k1_AXP_s010
                                 + 1.12965 w_k1_DIS_s010
                                 + 1.16829 w_k1_FDX_s010
                                 + 1.04756 w_k1_IBM_s010
                                 + 1.20665 w_k1_INTC_s010
                                 + 0.883341 w_k1_KO_s010
                                 + 1.07849 w_k1_MAT_s010
                                 + 1.02102 w_k1_MSFT_s010
                                 + 0.960811 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.72541 w_k1_ADM_s010
                                 + 1.00876 w_k1_AXP_s010
                                 + 0.839871 w_k1_DIS_s010
                                 + 0.917468 w_k1_FDX_s010
                                 + 1.0857 w_k1_IBM_s010
                                 + 1.19448 w_k1_INTC_s010
                                 + 0.91485 w_k1_KO_s010 + 1.31059 w_k1_MAT_s010
                                 + 0.890943 w_k1_MSFT_s010
                                 + 0.94926 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.788373 w_k1_ADM_s010
                                 + 1.25306 w_k1_AXP_s010
                                 + 1.06101 w_k1_DIS_s010
                                 + 1.10107 w_k1_FDX_s010
                                 + 0.852509 w_k1_IBM_s010
                                 + 1.34429 w_k1_INTC_s010
                                 + 0.989708 w_k1_KO_s010
                                 + 1.15337 w_k1_MAT_s010
                                 + 1.06079 w_k1_MSFT_s010
                                 + 0.893702 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.878812 w_k1_ADM_s011
                                 + 1.21877 w_k1_AXP_s011
                                 + 1.28749 w_k1_DIS_s011
                                 + 1.07546 w_k1_FDX_s011
                                 + 1.34329 w_k1_IBM_s011
                                 + 1.29056 w_k1_INTC_s011
                                 + 1.31456 w_k1_KO_s011 + 1.32272 w_k1_MAT_s011
                                 + 0.866969 w_k1_MSFT_s011
                                 + 1.14057 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.00222 w_k1_ADM_s011
                                 + 0.901447 w_k1_AXP_s011
                                 + 0.938601 w_k1_DIS_s011
                                 + 0.903038 w_k1_FDX_s011
                                 + 1.17097 w_k1_IBM_s011
                                 + 1.10226 w_k1_INTC_s011 + 1.2229 w_k1_KO_s011
                                 + 1.30328 w_k1_MAT_s011
                                 + 0.887534 w_k1_MSFT_s011
                                 + 1.09944 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.01196 w_k1_ADM_s011
                                 + 1.46584 w_k1_AXP_s011
                                 + 1.39256 w_k1_DIS_s011
                                 + 1.34293 w_k1_FDX_s011
                                 + 0.876778 w_k1_IBM_s011
                                 + 1.36748 w_k1_INTC_s011
                                 + 1.12657 w_k1_KO_s011 + 1.34968 w_k1_MAT_s011
                                 + 1.11407 w_k1_MSFT_s011
                                 + 1.44376 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.00173 w_k1_ADM_s011
                                 + 1.27196 w_k1_AXP_s011 + 1.1709 w_k1_DIS_s011
                                 + 0.975684 w_k1_FDX_s011
                                 + 1.81012 w_k1_IBM_s011
                                 + 1.25424 w_k1_INTC_s011
                                 + 1.33794 w_k1_KO_s011 + 1.67287 w_k1_MAT_s011
                                 + 1.16907 w_k1_MSFT_s011
                                 + 1.34149 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.05598 w_k1_ADM_s011
                                 + 1.31657 w_k1_AXP_s011
                                 + 1.12659 w_k1_DIS_s011 + 1.2326 w_k1_FDX_s011
                                 + 0.936527 w_k1_IBM_s011
                                 + 1.01113 w_k1_INTC_s011
                                 + 0.958118 w_k1_KO_s011
                                 + 1.41348 w_k1_MAT_s011
                                 + 0.905035 w_k1_MSFT_s011
                                 + 1.19434 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.689785 w_k1_ADM_s011
                                 + 0.796564 w_k1_AXP_s011
                                 + 0.873888 w_k1_DIS_s011
                                 + 0.745252 w_k1_FDX_s011
                                 + 1.32889 w_k1_IBM_s011
                                 + 0.975886 w_k1_INTC_s011
                                 + 1.12999 w_k1_KO_s011 + 1.47771 w_k1_MAT_s011
                                 + 0.934346 w_k1_MSFT_s011
                                 + 1.00015 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.817232 w_k1_ADM_s011
                                 + 0.876752 w_k1_AXP_s011
                                 + 0.888268 w_k1_DIS_s011
                                 + 0.92349 w_k1_FDX_s011
                                 + 1.01858 w_k1_IBM_s011
                                 + 1.08334 w_k1_INTC_s011 + 1.0008 w_k1_KO_s011
                                 + 1.02309 w_k1_MAT_s011
                                 + 1.01561 w_k1_MSFT_s011
                                 + 1.14596 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.892765 w_k1_ADM_s011
                                 + 0.86142 w_k1_AXP_s011
                                 + 0.738855 w_k1_DIS_s011
                                 + 0.864612 w_k1_FDX_s011
                                 + 0.765854 w_k1_IBM_s011
                                 + 0.961012 w_k1_INTC_s011
                                 + 0.80115 w_k1_KO_s011
                                 + 0.767052 w_k1_MAT_s011
                                 + 0.746445 w_k1_MSFT_s011
                                 + 0.915701 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.70715 w_k1_ADM_s012
                                 + 1.06226 w_k1_AXP_s012
                                 + 1.41513 w_k1_DIS_s012
                                 + 1.27697 w_k1_FDX_s012
                                 + 0.84328 w_k1_IBM_s012
                                 + 1.11492 w_k1_INTC_s012
                                 + 1.20076 w_k1_KO_s012
                                 + 0.866491 w_k1_MAT_s012
                                 + 1.28999 w_k1_MSFT_s012
                                 + 1.05921 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.33101 w_k1_ADM_s012
                                 + 1.46364 w_k1_AXP_s012
                                 + 1.31343 w_k1_DIS_s012
                                 + 1.15809 w_k1_FDX_s012 + 1.8305 w_k1_IBM_s012
                                 + 1.15943 w_k1_INTC_s012
                                 + 1.83301 w_k1_KO_s012 + 2.02806 w_k1_MAT_s012
                                 + 1.2926 w_k1_MSFT_s012 + 1.31787 w_k1_PG_s012
                                 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.01297 w_k1_ADM_s012
                                 + 1.10513 w_k1_AXP_s012
                                 + 0.711994 w_k1_DIS_s012
                                 + 0.814308 w_k1_FDX_s012
                                 + 1.38484 w_k1_IBM_s012
                                 + 0.782335 w_k1_INTC_s012
                                 + 1.04517 w_k1_KO_s012 + 1.45283 w_k1_MAT_s012
                                 + 0.787068 w_k1_MSFT_s012
                                 + 0.857687 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.994424 w_k1_ADM_s012
                                 + 0.872245 w_k1_AXP_s012
                                 + 0.972328 w_k1_DIS_s012
                                 + 0.873883 w_k1_FDX_s012
                                 + 0.897569 w_k1_IBM_s012
                                 + 1.1615 w_k1_INTC_s012 + 1.01163 w_k1_KO_s012
                                 + 0.577897 w_k1_MAT_s012
                                 + 0.961597 w_k1_MSFT_s012
                                 + 0.927901 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.925817 w_k1_ADM_s012
                                 + 0.832458 w_k1_AXP_s012
                                 + 0.641082 w_k1_DIS_s012
                                 + 0.785929 w_k1_FDX_s012
                                 + 0.765432 w_k1_IBM_s012
                                 + 0.74623 w_k1_INTC_s012
                                 + 0.811794 w_k1_KO_s012
                                 + 0.647564 w_k1_MAT_s012
                                 + 0.749371 w_k1_MSFT_s012
                                 + 0.784258 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.15817 w_k1_ADM_s012
                                 + 1.20804 w_k1_AXP_s012
                                 + 0.989966 w_k1_DIS_s012
                                 + 1.15805 w_k1_FDX_s012
                                 + 0.657127 w_k1_IBM_s012
                                 + 1.40839 w_k1_INTC_s012
                                 + 0.827923 w_k1_KO_s012
                                 + 0.495569 w_k1_MAT_s012
                                 + 1.15149 w_k1_MSFT_s012
                                 + 0.961613 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.41897 w_k1_ADM_s012
                                 + 1.43796 w_k1_AXP_s012
                                 + 0.988401 w_k1_DIS_s012
                                 + 1.03485 w_k1_FDX_s012 + 1.1408 w_k1_IBM_s012
                                 + 0.972324 w_k1_INTC_s012
                                 + 1.15037 w_k1_KO_s012 + 1.22842 w_k1_MAT_s012
                                 + 1.05912 w_k1_MSFT_s012
                                 + 1.07044 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.28227 w_k1_ADM_s012
                                 + 1.08769 w_k1_AXP_s012
                                 + 0.823081 w_k1_DIS_s012
                                 + 1.03793 w_k1_FDX_s012
                                 + 0.965899 w_k1_IBM_s012
                                 + 0.936851 w_k1_INTC_s012
                                 + 1.01393 w_k1_KO_s012
                                 + 0.845365 w_k1_MAT_s012
                                 + 1.03482 w_k1_MSFT_s012
                                 + 0.954494 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.02466 w_k1_ADM_s013
                                 + 1.06391 w_k1_AXP_s013
                                 + 0.999979 w_k1_DIS_s013
                                 + 0.986878 w_k1_FDX_s013
                                 + 1.17835 w_k1_IBM_s013
                                 + 0.908093 w_k1_INTC_s013
                                 + 0.937624 w_k1_KO_s013
                                 + 1.50839 w_k1_MAT_s013
                                 + 1.04805 w_k1_MSFT_s013
                                 + 0.974788 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.843469 w_k1_ADM_s013
                                 + 0.716247 w_k1_AXP_s013
                                 + 0.637134 w_k1_DIS_s013
                                 + 0.685702 w_k1_FDX_s013
                                 + 0.910808 w_k1_IBM_s013
                                 + 0.820987 w_k1_INTC_s013
                                 + 0.929853 w_k1_KO_s013
                                 + 0.809511 w_k1_MAT_s013
                                 + 0.833536 w_k1_MSFT_s013
                                 + 0.801694 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.94969 w_k1_ADM_s013
                                 + 0.922374 w_k1_AXP_s013
                                 + 0.81835 w_k1_DIS_s013
                                 + 0.736373 w_k1_FDX_s013
                                 + 1.25289 w_k1_IBM_s013
                                 + 0.783229 w_k1_INTC_s013
                                 + 0.901709 w_k1_KO_s013
                                 + 1.15234 w_k1_MAT_s013
                                 + 0.898678 w_k1_MSFT_s013
                                 + 0.919544 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.00179 w_k1_ADM_s013
                                 + 1.11906 w_k1_AXP_s013
                                 + 1.36643 w_k1_DIS_s013
                                 + 1.05422 w_k1_FDX_s013
                                 + 1.55508 w_k1_IBM_s013
                                 + 1.19787 w_k1_INTC_s013
                                 + 1.33701 w_k1_KO_s013 + 1.51642 w_k1_MAT_s013
                                 + 1.33026 w_k1_MSFT_s013
                                 + 1.09705 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.24389 w_k1_ADM_s013
                                 + 0.96239 w_k1_AXP_s013
                                 + 0.889676 w_k1_DIS_s013
                                 + 1.02843 w_k1_FDX_s013
                                 + 0.842655 w_k1_IBM_s013
                                 + 0.884929 w_k1_INTC_s013
                                 + 0.858987 w_k1_KO_s013
                                 + 0.690437 w_k1_MAT_s013
                                 + 1.06653 w_k1_MSFT_s013
                                 + 1.00739 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.926353 w_k1_ADM_s013
                                 + 0.747885 w_k1_AXP_s013
                                 + 0.733814 w_k1_DIS_s013
                                 + 0.84666 w_k1_FDX_s013
                                 + 0.654532 w_k1_IBM_s013
                                 + 0.87585 w_k1_INTC_s013
                                 + 0.763714 w_k1_KO_s013
                                 + 0.604695 w_k1_MAT_s013
                                 + 0.998246 w_k1_MSFT_s013
                                 + 0.920213 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.31412 w_k1_ADM_s013
                                 + 0.948639 w_k1_AXP_s013
                                 + 1.33821 w_k1_DIS_s013
                                 + 1.13136 w_k1_FDX_s013
                                 + 0.841154 w_k1_IBM_s013
                                 + 0.937351 w_k1_INTC_s013
                                 + 1.03461 w_k1_KO_s013
                                 + 0.862798 w_k1_MAT_s013
                                 + 1.3257 w_k1_MSFT_s013 + 1.05887 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.40084 w_k1_ADM_s013
                                 + 1.38842 w_k1_AXP_s013
                                 + 1.30453 w_k1_DIS_s013
                                 + 1.24703 w_k1_FDX_s013
                                 + 1.02571 w_k1_IBM_s013
                                 + 1.05283 w_k1_INTC_s013
                                 + 1.13376 w_k1_KO_s013 + 1.078 w_k1_MAT_s013
                                 + 1.36428 w_k1_MSFT_s013
                                 + 1.26266 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.908765 w_k1_ADM_s014
                                 + 0.605517 w_k1_AXP_s014
                                 + 0.650838 w_k1_DIS_s014
                                 + 0.725981 w_k1_FDX_s014
                                 + 0.769665 w_k1_IBM_s014
                                 + 0.761867 w_k1_INTC_s014
                                 + 0.684051 w_k1_KO_s014
                                 + 0.858306 w_k1_MAT_s014
                                 + 0.836944 w_k1_MSFT_s014
                                 + 0.783758 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.89851 w_k1_ADM_s014
                                 + 0.858064 w_k1_AXP_s014
                                 + 0.801257 w_k1_DIS_s014
                                 + 0.972515 w_k1_FDX_s014
                                 + 0.923752 w_k1_IBM_s014
                                 + 0.998983 w_k1_INTC_s014
                                 + 0.848928 w_k1_KO_s014
                                 + 1.44837 w_k1_MAT_s014
                                 + 0.994368 w_k1_MSFT_s014
                                 + 0.855608 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.663148 w_k1_ADM_s014
                                 + 0.821647 w_k1_AXP_s014
                                 + 0.584844 w_k1_DIS_s014
                                 + 0.827994 w_k1_FDX_s014
                                 + 0.678071 w_k1_IBM_s014
                                 + 0.879335 w_k1_INTC_s014
                                 + 0.670481 w_k1_KO_s014
                                 + 1.20354 w_k1_MAT_s014
                                 + 0.834273 w_k1_MSFT_s014
                                 + 0.744036 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.981765 w_k1_ADM_s014
                                 + 0.938579 w_k1_AXP_s014
                                 + 1.24955 w_k1_DIS_s014
                                 + 1.07142 w_k1_FDX_s014
                                 + 0.913186 w_k1_IBM_s014
                                 + 1.02771 w_k1_INTC_s014
                                 + 1.02844 w_k1_KO_s014 + 1.28733 w_k1_MAT_s014
                                 + 1.20098 w_k1_MSFT_s014
                                 + 1.07942 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.51663 w_k1_ADM_s014
                                 + 1.8382 w_k1_AXP_s014 + 1.20516 w_k1_DIS_s014
                                 + 1.34921 w_k1_FDX_s014 + 1.2472 w_k1_IBM_s014
                                 + 1.42074 w_k1_INTC_s014
                                 + 1.13528 w_k1_KO_s014 + 2.46279 w_k1_MAT_s014
                                 + 1.26508 w_k1_MSFT_s014 + 1.2727 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.1882 w_k1_ADM_s014
                                 + 1.01543 w_k1_AXP_s014
                                 + 1.09314 w_k1_DIS_s014
                                 + 1.28615 w_k1_FDX_s014
                                 + 0.88127 w_k1_IBM_s014
                                 + 1.4069 w_k1_INTC_s014
                                 + 0.924256 w_k1_KO_s014
                                 + 1.00658 w_k1_MAT_s014
                                 + 1.3583 w_k1_MSFT_s014
                                 + 0.964112 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.68714 w_k1_ADM_s014
                                 + 1.47095 w_k1_AXP_s014
                                 + 1.86026 w_k1_DIS_s014
                                 + 1.51913 w_k1_FDX_s014
                                 + 1.16807 w_k1_IBM_s014
                                 + 1.21423 w_k1_INTC_s014 + 1.4332 w_k1_KO_s014
                                 + 1.63401 w_k1_MAT_s014
                                 + 1.52092 w_k1_MSFT_s014
                                 + 1.16172 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.1361 w_k1_ADM_s014
                                 + 1.25206 w_k1_AXP_s014
                                 + 1.03297 w_k1_DIS_s014
                                 + 1.17892 w_k1_FDX_s014
                                 + 1.15634 w_k1_IBM_s014
                                 + 1.03035 w_k1_INTC_s014
                                 + 1.00418 w_k1_KO_s014 + 1.51293 w_k1_MAT_s014
                                 + 1.10498 w_k1_MSFT_s014
                                 + 1.00016 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.07822 w_k1_ADM_s015
                                 + 0.94077 w_k1_AXP_s015
                                 + 1.35596 w_k1_DIS_s015
                                 + 1.23649 w_k1_FDX_s015
                                 + 0.865757 w_k1_IBM_s015
                                 + 1.13136 w_k1_INTC_s015
                                 + 0.975814 w_k1_KO_s015
                                 + 0.666228 w_k1_MAT_s015
                                 + 1.15843 w_k1_MSFT_s015
                                 + 0.89972 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.04878 w_k1_ADM_s015
                                 + 1.23826 w_k1_AXP_s015 + 1.7685 w_k1_DIS_s015
                                 + 1.24651 w_k1_FDX_s015
                                 + 1.42167 w_k1_IBM_s015
                                 + 1.19315 w_k1_INTC_s015
                                 + 1.29282 w_k1_KO_s015 + 1.35634 w_k1_MAT_s015
                                 + 1.28306 w_k1_MSFT_s015
                                 + 1.14294 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.676176 w_k1_ADM_s015
                                 + 0.661729 w_k1_AXP_s015
                                 + 0.878878 w_k1_DIS_s015
                                 + 0.76515 w_k1_FDX_s015
                                 + 1.29597 w_k1_IBM_s015
                                 + 0.685853 w_k1_INTC_s015
                                 + 1.0398 w_k1_KO_s015 + 1.22192 w_k1_MAT_s015
                                 + 0.801949 w_k1_MSFT_s015
                                 + 0.921994 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.00211 w_k1_ADM_s015
                                 + 0.684044 w_k1_AXP_s015
                                 + 0.755708 w_k1_DIS_s015
                                 + 0.837657 w_k1_FDX_s015
                                 + 0.882582 w_k1_IBM_s015
                                 + 0.584247 w_k1_INTC_s015
                                 + 0.810545 w_k1_KO_s015
                                 + 0.910181 w_k1_MAT_s015
                                 + 0.808118 w_k1_MSFT_s015
                                 + 0.920576 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.798821 w_k1_ADM_s015
                                 + 1.59503 w_k1_AXP_s015
                                 + 1.14635 w_k1_DIS_s015
                                 + 1.26881 w_k1_FDX_s015
                                 + 1.00175 w_k1_IBM_s015
                                 + 0.857966 w_k1_INTC_s015
                                 + 0.916758 w_k1_KO_s015
                                 + 1.77711 w_k1_MAT_s015
                                 + 0.929708 w_k1_MSFT_s015
                                 + 1.05666 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.10272 w_k1_ADM_s015
                                 + 1.14693 w_k1_AXP_s015
                                 + 0.837432 w_k1_DIS_s015
                                 + 1.02726 w_k1_FDX_s015
                                 + 0.933179 w_k1_IBM_s015
                                 + 0.894443 w_k1_INTC_s015
                                 + 1.12241 w_k1_KO_s015 + 1.35095 w_k1_MAT_s015
                                 + 0.88464 w_k1_MSFT_s015 + 1.0482 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.998535 w_k1_ADM_s015
                                 + 0.963817 w_k1_AXP_s015
                                 + 1.13052 w_k1_DIS_s015
                                 + 1.06296 w_k1_FDX_s015
                                 + 0.934486 w_k1_IBM_s015
                                 + 0.774031 w_k1_INTC_s015
                                 + 0.873491 w_k1_KO_s015
                                 + 0.809133 w_k1_MAT_s015
                                 + 1.02697 w_k1_MSFT_s015
                                 + 0.924141 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.740622 w_k1_ADM_s015
                                 + 1.2444 w_k1_AXP_s015 + 1.07817 w_k1_DIS_s015
                                 + 1.01355 w_k1_FDX_s015
                                 + 1.59165 w_k1_IBM_s015
                                 + 1.0011 w_k1_INTC_s015 + 1.04181 w_k1_KO_s015
                                 + 1.62211 w_k1_MAT_s015
                                 + 0.895807 w_k1_MSFT_s015
                                 + 1.03493 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.849834 w_k1_ADM_s016
                                 + 1.05573 w_k1_AXP_s016
                                 + 0.965602 w_k1_DIS_s016
                                 + 0.934204 w_k1_FDX_s016
                                 + 0.956149 w_k1_IBM_s016
                                 + 0.933501 w_k1_INTC_s016
                                 + 1.00491 w_k1_KO_s016 + 1.20622 w_k1_MAT_s016
                                 + 0.959826 w_k1_MSFT_s016
                                 + 0.885938 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.856546 w_k1_ADM_s016
                                 + 0.817056 w_k1_AXP_s016
                                 + 0.880759 w_k1_DIS_s016
                                 + 0.957256 w_k1_FDX_s016
                                 + 0.645946 w_k1_IBM_s016
                                 + 0.928636 w_k1_INTC_s016
                                 + 0.786569 w_k1_KO_s016
                                 + 0.629001 w_k1_MAT_s016
                                 + 0.88604 w_k1_MSFT_s016
                                 + 0.922866 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.06523 w_k1_ADM_s016
                                 + 1.57088 w_k1_AXP_s016
                                 + 1.65054 w_k1_DIS_s016
                                 + 1.27251 w_k1_FDX_s016
                                 + 1.34798 w_k1_IBM_s016
                                 + 1.38579 w_k1_INTC_s016
                                 + 1.30555 w_k1_KO_s016 + 1.92094 w_k1_MAT_s016
                                 + 1.41431 w_k1_MSFT_s016
                                 + 1.40519 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.990088 w_k1_ADM_s016
                                 + 1.10636 w_k1_AXP_s016
                                 + 1.39006 w_k1_DIS_s016
                                 + 1.19027 w_k1_FDX_s016
                                 + 0.934829 w_k1_IBM_s016
                                 + 1.19314 w_k1_INTC_s016
                                 + 1.19117 w_k1_KO_s016 + 1.04019 w_k1_MAT_s016
                                 + 1.14398 w_k1_MSFT_s016
                                 + 1.28978 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.738954 w_k1_ADM_s016
                                 + 0.660086 w_k1_AXP_s016
                                 + 0.735028 w_k1_DIS_s016
                                 + 0.676595 w_k1_FDX_s016
                                 + 1.24097 w_k1_IBM_s016
                                 + 0.67703 w_k1_INTC_s016
                                 + 0.98929 w_k1_KO_s016 + 1.31701 w_k1_MAT_s016
                                 + 0.750212 w_k1_MSFT_s016
                                 + 0.997572 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.09983 w_k1_ADM_s016
                                 + 1.27964 w_k1_AXP_s016
                                 + 0.953858 w_k1_DIS_s016
                                 + 0.94953 w_k1_FDX_s016
                                 + 1.22997 w_k1_IBM_s016
                                 + 1.04821 w_k1_INTC_s016
                                 + 1.16389 w_k1_KO_s016 + 1.68897 w_k1_MAT_s016
                                 + 1.02842 w_k1_MSFT_s016
                                 + 1.14023 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.20555 w_k1_ADM_s016
                                 + 1.15096 w_k1_AXP_s016
                                 + 1.20022 w_k1_DIS_s016
                                 + 0.997595 w_k1_FDX_s016
                                 + 1.76644 w_k1_IBM_s016
                                 + 0.968696 w_k1_INTC_s016
                                 + 1.46452 w_k1_KO_s016 + 1.56436 w_k1_MAT_s016
                                 + 0.99876 w_k1_MSFT_s016
                                 + 1.31194 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.875314 w_k1_ADM_s016
                                 + 0.74056 w_k1_AXP_s016
                                 + 0.690709 w_k1_DIS_s016
                                 + 0.7375 w_k1_FDX_s016
                                 + 0.875039 w_k1_IBM_s016
                                 + 0.926582 w_k1_INTC_s016
                                 + 0.852858 w_k1_KO_s016
                                 + 0.887134 w_k1_MAT_s016
                                 + 0.838532 w_k1_MSFT_s016
                                 + 0.899718 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.950439 w_k1_ADM_s017
                                 + 0.870163 w_k1_AXP_s017
                                 + 0.924577 w_k1_DIS_s017
                                 + 1.01041 w_k1_FDX_s017
                                 + 0.99536 w_k1_IBM_s017
                                 + 0.995827 w_k1_INTC_s017
                                 + 1.04549 w_k1_KO_s017
                                 + 0.863675 w_k1_MAT_s017
                                 + 1.0411 w_k1_MSFT_s017 + 1.05275 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.894718 w_k1_ADM_s017
                                 + 1.06176 w_k1_AXP_s017 + 1.0571 w_k1_DIS_s017
                                 + 1.0442 w_k1_FDX_s017 + 1.50072 w_k1_IBM_s017
                                 + 1.08319 w_k1_INTC_s017
                                 + 1.15433 w_k1_KO_s017 + 1.28335 w_k1_MAT_s017
                                 + 1.01122 w_k1_MSFT_s017
                                 + 1.05875 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.687847 w_k1_ADM_s017
                                 + 0.719452 w_k1_AXP_s017
                                 + 0.604876 w_k1_DIS_s017
                                 + 0.799291 w_k1_FDX_s017
                                 + 0.586658 w_k1_IBM_s017
                                 + 0.777158 w_k1_INTC_s017
                                 + 0.700416 w_k1_KO_s017
                                 + 0.707204 w_k1_MAT_s017
                                 + 0.841111 w_k1_MSFT_s017
                                 + 0.804268 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.900411 w_k1_ADM_s017
                                 + 0.719522 w_k1_AXP_s017
                                 + 0.744278 w_k1_DIS_s017
                                 + 0.782789 w_k1_FDX_s017
                                 + 0.920352 w_k1_IBM_s017
                                 + 0.786791 w_k1_INTC_s017
                                 + 0.808291 w_k1_KO_s017
                                 + 0.895988 w_k1_MAT_s017
                                 + 0.970743 w_k1_MSFT_s017
                                 + 0.818046 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.12823 w_k1_ADM_s017
                                 + 1.41546 w_k1_AXP_s017
                                 + 1.50132 w_k1_DIS_s017
                                 + 1.29325 w_k1_FDX_s017
                                 + 1.26366 w_k1_IBM_s017
                                 + 1.09541 w_k1_INTC_s017
                                 + 1.22985 w_k1_KO_s017 + 1.18948 w_k1_MAT_s017
                                 + 1.22528 w_k1_MSFT_s017
                                 + 1.10731 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.835695 w_k1_ADM_s017
                                 + 1.11872 w_k1_AXP_s017
                                 + 1.11036 w_k1_DIS_s017
                                 + 1.00539 w_k1_FDX_s017
                                 + 1.07374 w_k1_IBM_s017
                                 + 0.996336 w_k1_INTC_s017
                                 + 1.20211 w_k1_KO_s017 + 1.63147 w_k1_MAT_s017
                                 + 1.15809 w_k1_MSFT_s017
                                 + 1.15502 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.12369 w_k1_ADM_s017
                                 + 1.02715 w_k1_AXP_s017
                                 + 1.17934 w_k1_DIS_s017
                                 + 1.17646 w_k1_FDX_s017
                                 + 0.735004 w_k1_IBM_s017
                                 + 1.10328 w_k1_INTC_s017
                                 + 0.950786 w_k1_KO_s017
                                 + 0.789641 w_k1_MAT_s017
                                 + 1.21338 w_k1_MSFT_s017 + 1.0309 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.841159 w_k1_ADM_s017
                                 + 0.714094 w_k1_AXP_s017
                                 + 0.885358 w_k1_DIS_s017
                                 + 0.947883 w_k1_FDX_s017
                                 + 0.662512 w_k1_IBM_s017
                                 + 0.940419 w_k1_INTC_s017
                                 + 0.895855 w_k1_KO_s017
                                 + 0.55971 w_k1_MAT_s017
                                 + 1.08406 w_k1_MSFT_s017
                                 + 0.87109 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.00421 w_k1_ADM_s100
                                 + 0.936667 w_k1_AXP_s100
                                 + 0.860541 w_k1_DIS_s100
                                 + 0.910264 w_k1_FDX_s100
                                 + 1.38086 w_k1_IBM_s100
                                 + 0.948189 w_k1_INTC_s100
                                 + 1.0957 w_k1_KO_s100 + 1.46491 w_k1_MAT_s100
                                 + 1.07948 w_k1_MSFT_s100
                                 + 1.05906 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.981939 w_k1_ADM_s100
                                 + 0.773112 w_k1_AXP_s100
                                 + 0.856272 w_k1_DIS_s100
                                 + 0.933284 w_k1_FDX_s100
                                 + 0.937242 w_k1_IBM_s100
                                 + 0.827216 w_k1_INTC_s100
                                 + 0.857743 w_k1_KO_s100
                                 + 0.70557 w_k1_MAT_s100
                                 + 0.997517 w_k1_MSFT_s100
                                 + 0.972811 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.763999 w_k1_ADM_s100
                                 + 0.734736 w_k1_AXP_s100
                                 + 0.653288 w_k1_DIS_s100
                                 + 0.76985 w_k1_FDX_s100
                                 + 0.922906 w_k1_IBM_s100
                                 + 0.842402 w_k1_INTC_s100
                                 + 0.922682 w_k1_KO_s100
                                 + 0.991809 w_k1_MAT_s100
                                 + 0.823564 w_k1_MSFT_s100
                                 + 0.902263 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.864543 w_k1_ADM_s100
                                 + 0.886211 w_k1_AXP_s100
                                 + 0.804717 w_k1_DIS_s100
                                 + 0.828544 w_k1_FDX_s100
                                 + 1.19379 w_k1_IBM_s100
                                 + 0.760494 w_k1_INTC_s100
                                 + 0.959359 w_k1_KO_s100
                                 + 0.974346 w_k1_MAT_s100
                                 + 0.764619 w_k1_MSFT_s100
                                 + 0.886984 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.25963 w_k1_ADM_s100
                                 + 1.07861 w_k1_AXP_s100
                                 + 1.28777 w_k1_DIS_s100
                                 + 1.19687 w_k1_FDX_s100
                                 + 1.12556 w_k1_IBM_s100
                                 + 1.00281 w_k1_INTC_s100
                                 + 1.23957 w_k1_KO_s100 + 1.24967 w_k1_MAT_s100
                                 + 1.24173 w_k1_MSFT_s100
                                 + 1.26919 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.50681 w_k1_ADM_s100
                                 + 1.4085 w_k1_AXP_s100 + 1.1594 w_k1_DIS_s100
                                 + 1.34828 w_k1_FDX_s100
                                 + 1.11557 w_k1_IBM_s100
                                 + 1.35143 w_k1_INTC_s100
                                 + 1.14664 w_k1_KO_s100 + 1.51157 w_k1_MAT_s100
                                 + 1.14771 w_k1_MSFT_s100
                                 + 1.27865 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.1918 w_k1_ADM_s100
                                 + 1.13516 w_k1_AXP_s100
                                 + 1.27575 w_k1_DIS_s100 + 1.0638 w_k1_FDX_s100
                                 + 1.64786 w_k1_IBM_s100
                                 + 0.968446 w_k1_INTC_s100
                                 + 1.43898 w_k1_KO_s100 + 1.28587 w_k1_MAT_s100
                                 + 1.06569 w_k1_MSFT_s100 + 1.1779 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.884042 w_k1_ADM_s100
                                 + 0.613507 w_k1_AXP_s100
                                 + 0.680448 w_k1_DIS_s100
                                 + 0.743614 w_k1_FDX_s100
                                 + 0.726531 w_k1_IBM_s100
                                 + 0.659604 w_k1_INTC_s100
                                 + 0.739248 w_k1_KO_s100
                                 + 0.745101 w_k1_MAT_s100
                                 + 0.841622 w_k1_MSFT_s100
                                 + 0.826357 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.01005 w_k1_ADM_s101
                                 + 1.19316 w_k1_AXP_s101 + 1.0925 w_k1_DIS_s101
                                 + 1.13422 w_k1_FDX_s101
                                 + 1.21814 w_k1_IBM_s101
                                 + 0.974334 w_k1_INTC_s101
                                 + 1.0732 w_k1_KO_s101 + 0.831713 w_k1_MAT_s101
                                 + 1.05968 w_k1_MSFT_s101
                                 + 1.05222 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.1407 w_k1_ADM_s101
                                 + 1.70163 w_k1_AXP_s101
                                 + 1.06781 w_k1_DIS_s101
                                 + 1.15669 w_k1_FDX_s101 + 1.2727 w_k1_IBM_s101
                                 + 1.27612 w_k1_INTC_s101
                                 + 1.20718 w_k1_KO_s101 + 1.47919 w_k1_MAT_s101
                                 + 1.11992 w_k1_MSFT_s101
                                 + 1.10657 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.06202 w_k1_ADM_s101
                                 + 1.00928 w_k1_AXP_s101
                                 + 0.826529 w_k1_DIS_s101
                                 + 0.923849 w_k1_FDX_s101
                                 + 1.0177 w_k1_IBM_s101
                                 + 0.868897 w_k1_INTC_s101
                                 + 0.734858 w_k1_KO_s101
                                 + 0.893763 w_k1_MAT_s101
                                 + 0.879301 w_k1_MSFT_s101
                                 + 0.901384 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.34608 w_k1_ADM_s101
                                 + 1.44944 w_k1_AXP_s101
                                 + 1.61436 w_k1_DIS_s101
                                 + 1.35706 w_k1_FDX_s101
                                 + 1.22794 w_k1_IBM_s101
                                 + 1.2205 w_k1_INTC_s101 + 1.1998 w_k1_KO_s101
                                 + 1.03727 w_k1_MAT_s101
                                 + 1.3059 w_k1_MSFT_s101 + 1.19792 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.999872 w_k1_ADM_s101
                                 + 1.04035 w_k1_AXP_s101
                                 + 0.872144 w_k1_DIS_s101
                                 + 1.0722 w_k1_FDX_s101
                                 + 0.634099 w_k1_IBM_s101
                                 + 0.883065 w_k1_INTC_s101
                                 + 0.697707 w_k1_KO_s101
                                 + 0.721653 w_k1_MAT_s101
                                 + 0.916943 w_k1_MSFT_s101
                                 + 0.904258 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.880484 w_k1_ADM_s101
                                 + 1.0802 w_k1_AXP_s101 + 1.0294 w_k1_DIS_s101
                                 + 0.995046 w_k1_FDX_s101
                                 + 1.13211 w_k1_IBM_s101
                                 + 1.00836 w_k1_INTC_s101
                                 + 1.06007 w_k1_KO_s101 + 1.07964 w_k1_MAT_s101
                                 + 0.961882 w_k1_MSFT_s101
                                 + 1.00121 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.851979 w_k1_ADM_s101
                                 + 0.780466 w_k1_AXP_s101
                                 + 1.29831 w_k1_DIS_s101
                                 + 1.11717 w_k1_FDX_s101
                                 + 0.873216 w_k1_IBM_s101
                                 + 1.15903 w_k1_INTC_s101
                                 + 1.01127 w_k1_KO_s101
                                 + 0.632322 w_k1_MAT_s101
                                 + 1.10607 w_k1_MSFT_s101
                                 + 1.06228 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.763921 w_k1_ADM_s101
                                 + 0.715877 w_k1_AXP_s101
                                 + 0.664515 w_k1_DIS_s101
                                 + 0.757117 w_k1_FDX_s101
                                 + 0.79537 w_k1_IBM_s101
                                 + 0.706528 w_k1_INTC_s101
                                 + 0.724596 w_k1_KO_s101
                                 + 0.543873 w_k1_MAT_s101
                                 + 0.839092 w_k1_MSFT_s101
                                 + 0.763047 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.41041 w_k1_ADM_s102
                                 + 1.35043 w_k1_AXP_s102
                                 + 1.35214 w_k1_DIS_s102 + 1.1922 w_k1_FDX_s102
                                 + 1.05521 w_k1_IBM_s102
                                 + 1.47151 w_k1_INTC_s102
                                 + 1.09475 w_k1_KO_s102 + 1.11914 w_k1_MAT_s102
                                 + 1.22229 w_k1_MSFT_s102
                                 + 1.10702 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.12997 w_k1_ADM_s102
                                 + 1.41289 w_k1_AXP_s102
                                 + 1.80991 w_k1_DIS_s102
                                 + 1.29511 w_k1_FDX_s102
                                 + 1.27409 w_k1_IBM_s102
                                 + 1.52967 w_k1_INTC_s102
                                 + 1.38564 w_k1_KO_s102 + 1.22672 w_k1_MAT_s102
                                 + 1.26191 w_k1_MSFT_s102
                                 + 1.15798 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.03738 w_k1_ADM_s102
                                 + 1.13185 w_k1_AXP_s102
                                 + 1.46901 w_k1_DIS_s102
                                 + 0.973469 w_k1_FDX_s102
                                 + 1.59852 w_k1_IBM_s102
                                 + 1.09982 w_k1_INTC_s102
                                 + 1.34897 w_k1_KO_s102 + 1.38602 w_k1_MAT_s102
                                 + 0.93106 w_k1_MSFT_s102
                                 + 1.14755 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.04559 w_k1_ADM_s102
                                 + 0.905733 w_k1_AXP_s102
                                 + 1.13424 w_k1_DIS_s102
                                 + 0.958936 w_k1_FDX_s102
                                 + 0.882789 w_k1_IBM_s102
                                 + 1.22769 w_k1_INTC_s102
                                 + 0.952922 w_k1_KO_s102
                                 + 0.854046 w_k1_MAT_s102
                                 + 1.03085 w_k1_MSFT_s102
                                 + 1.15575 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.943785 w_k1_ADM_s102
                                 + 1.00537 w_k1_AXP_s102
                                 + 0.980706 w_k1_DIS_s102
                                 + 0.971573 w_k1_FDX_s102
                                 + 1.14826 w_k1_IBM_s102
                                 + 1.13241 w_k1_INTC_s102
                                 + 0.971748 w_k1_KO_s102
                                 + 1.12456 w_k1_MAT_s102
                                 + 1.06695 w_k1_MSFT_s102
                                 + 0.95692 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.924084 w_k1_ADM_s102
                                 + 0.704225 w_k1_AXP_s102
                                 + 0.82093 w_k1_DIS_s102
                                 + 0.797104 w_k1_FDX_s102
                                 + 0.81511 w_k1_IBM_s102
                                 + 1.11779 w_k1_INTC_s102
                                 + 0.843196 w_k1_KO_s102
                                 + 0.785343 w_k1_MAT_s102
                                 + 0.950554 w_k1_MSFT_s102
                                 + 0.89418 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.682916 w_k1_ADM_s102
                                 + 0.752545 w_k1_AXP_s102
                                 + 0.517518 w_k1_DIS_s102
                                 + 0.67258 w_k1_FDX_s102
                                 + 0.878322 w_k1_IBM_s102
                                 + 1.08401 w_k1_INTC_s102
                                 + 0.757907 w_k1_KO_s102 + 0.7541 w_k1_MAT_s102
                                 + 0.706583 w_k1_MSFT_s102
                                 + 0.859264 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.03221 w_k1_ADM_s102
                                 + 1.02017 w_k1_AXP_s102
                                 + 1.13422 w_k1_DIS_s102
                                 + 1.17002 w_k1_FDX_s102
                                 + 0.725759 w_k1_IBM_s102
                                 + 1.11963 w_k1_INTC_s102
                                 + 0.799493 w_k1_KO_s102
                                 + 0.567337 w_k1_MAT_s102
                                 + 1.14657 w_k1_MSFT_s102
                                 + 1.01612 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.894321 w_k1_ADM_s103
                                 + 0.853129 w_k1_AXP_s103
                                 + 0.977113 w_k1_DIS_s103
                                 + 0.972969 w_k1_FDX_s103
                                 + 0.688695 w_k1_IBM_s103
                                 + 0.890358 w_k1_INTC_s103
                                 + 0.926605 w_k1_KO_s103
                                 + 0.85819 w_k1_MAT_s103
                                 + 1.14802 w_k1_MSFT_s103
                                 + 1.04977 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.758216 w_k1_ADM_s103
                                 + 0.893501 w_k1_AXP_s103
                                 + 0.833621 w_k1_DIS_s103
                                 + 0.76677 w_k1_FDX_s103
                                 + 1.23345 w_k1_IBM_s103
                                 + 0.72422 w_k1_INTC_s103
                                 + 0.914693 w_k1_KO_s103
                                 + 1.67635 w_k1_MAT_s103
                                 + 0.85595 w_k1_MSFT_s103
                                 + 0.94342 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.07789 w_k1_ADM_s103
                                 + 1.43221 w_k1_AXP_s103
                                 + 1.18301 w_k1_DIS_s103
                                 + 1.18333 w_k1_FDX_s103
                                 + 0.918795 w_k1_IBM_s103
                                 + 1.12118 w_k1_INTC_s103
                                 + 1.04766 w_k1_KO_s103 + 1.84347 w_k1_MAT_s103
                                 + 1.18493 w_k1_MSFT_s103
                                 + 1.06657 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.711847 w_k1_ADM_s103
                                 + 0.617888 w_k1_AXP_s103
                                 + 0.508786 w_k1_DIS_s103
                                 + 0.637862 w_k1_FDX_s103
                                 + 0.802898 w_k1_IBM_s103
                                 + 0.688551 w_k1_INTC_s103
                                 + 0.632492 w_k1_KO_s103
                                 + 0.775062 w_k1_MAT_s103
                                 + 0.644443 w_k1_MSFT_s103
                                 + 0.760563 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.06897 w_k1_ADM_s103
                                 + 1.86748 w_k1_AXP_s103
                                 + 1.92475 w_k1_DIS_s103
                                 + 1.49885 w_k1_FDX_s103
                                 + 1.56191 w_k1_IBM_s103
                                 + 1.40644 w_k1_INTC_s103 + 1.6587 w_k1_KO_s103
                                 + 2.17913 w_k1_MAT_s103
                                 + 1.44452 w_k1_MSFT_s103
                                 + 1.38748 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.807926 w_k1_ADM_s103
                                 + 0.747627 w_k1_AXP_s103
                                 + 1.03798 w_k1_DIS_s103
                                 + 0.833673 w_k1_FDX_s103
                                 + 0.998068 w_k1_IBM_s103
                                 + 0.865518 w_k1_INTC_s103
                                 + 0.914151 w_k1_KO_s103
                                 + 1.01888 w_k1_MAT_s103
                                 + 0.990699 w_k1_MSFT_s103
                                 + 0.876641 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.1427 w_k1_ADM_s103
                                 + 1.60007 w_k1_AXP_s103 + 1.5398 w_k1_DIS_s103
                                 + 1.34715 w_k1_FDX_s103
                                 + 1.17102 w_k1_IBM_s103
                                 + 1.21281 w_k1_INTC_s103
                                 + 1.26623 w_k1_KO_s103 + 1.6395 w_k1_MAT_s103
                                 + 1.26033 w_k1_MSFT_s103 + 1.2488 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.887516 w_k1_ADM_s103
                                 + 1.12103 w_k1_AXP_s103
                                 + 0.898672 w_k1_DIS_s103
                                 + 1.00347 w_k1_FDX_s103
                                 + 0.969437 w_k1_IBM_s103
                                 + 0.790529 w_k1_INTC_s103
                                 + 0.816472 w_k1_KO_s103
                                 + 1.26578 w_k1_MAT_s103
                                 + 0.871334 w_k1_MSFT_s103
                                 + 0.953827 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.878704 w_k1_ADM_s104
                                 + 0.707462 w_k1_AXP_s104
                                 + 0.939409 w_k1_DIS_s104
                                 + 0.999823 w_k1_FDX_s104
                                 + 0.947973 w_k1_IBM_s104
                                 + 0.948633 w_k1_INTC_s104
                                 + 0.910654 w_k1_KO_s104
                                 + 0.956306 w_k1_MAT_s104
                                 + 1.0381 w_k1_MSFT_s104 + 0.98152 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.3003 w_k1_ADM_s104
                                 + 1.40068 w_k1_AXP_s104
                                 + 1.63298 w_k1_DIS_s104
                                 + 1.43541 w_k1_FDX_s104
                                 + 1.27682 w_k1_IBM_s104
                                 + 1.24049 w_k1_INTC_s104
                                 + 1.32003 w_k1_KO_s104 + 1.80902 w_k1_MAT_s104
                                 + 1.30885 w_k1_MSFT_s104
                                 + 1.29073 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.700395 w_k1_ADM_s104
                                 + 0.805867 w_k1_AXP_s104
                                 + 0.763488 w_k1_DIS_s104
                                 + 0.791334 w_k1_FDX_s104
                                 + 0.893757 w_k1_IBM_s104
                                 + 0.873722 w_k1_INTC_s104
                                 + 0.742504 w_k1_KO_s104
                                 + 1.13896 w_k1_MAT_s104
                                 + 0.899322 w_k1_MSFT_s104
                                 + 0.829723 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.847565 w_k1_ADM_s104
                                 + 0.843258 w_k1_AXP_s104
                                 + 0.880163 w_k1_DIS_s104
                                 + 0.85956 w_k1_FDX_s104
                                 + 1.23981 w_k1_IBM_s104
                                 + 0.778228 w_k1_INTC_s104
                                 + 0.953734 w_k1_KO_s104
                                 + 1.60874 w_k1_MAT_s104
                                 + 0.859122 w_k1_MSFT_s104
                                 + 0.988912 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.12289 w_k1_ADM_s104
                                 + 1.29232 w_k1_AXP_s104
                                 + 1.08506 w_k1_DIS_s104 + 1.056 w_k1_FDX_s104
                                 + 1.32041 w_k1_IBM_s104
                                 + 1.13829 w_k1_INTC_s104
                                 + 1.23153 w_k1_KO_s104 + 2.00795 w_k1_MAT_s104
                                 + 1.15739 w_k1_MSFT_s104
                                 + 1.12168 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.10158 w_k1_ADM_s104
                                 + 1.11119 w_k1_AXP_s104
                                 + 1.25675 w_k1_DIS_s104 + 1.2338 w_k1_FDX_s104
                                 + 0.904038 w_k1_IBM_s104
                                 + 1.17954 w_k1_INTC_s104
                                 + 1.08041 w_k1_KO_s104 + 1.21476 w_k1_MAT_s104
                                 + 1.21211 w_k1_MSFT_s104
                                 + 1.16823 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.673774 w_k1_ADM_s104
                                 + 0.689857 w_k1_AXP_s104
                                 + 0.787051 w_k1_DIS_s104
                                 + 0.883269 w_k1_FDX_s104
                                 + 0.662594 w_k1_IBM_s104
                                 + 0.913969 w_k1_INTC_s104
                                 + 0.670621 w_k1_KO_s104
                                 + 0.739801 w_k1_MAT_s104
                                 + 0.922302 w_k1_MSFT_s104
                                 + 0.930309 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.960861 w_k1_ADM_s104
                                 + 1.11058 w_k1_AXP_s104
                                 + 1.15124 w_k1_DIS_s104
                                 + 0.969735 w_k1_FDX_s104
                                 + 1.1059 w_k1_IBM_s104
                                 + 1.15692 w_k1_INTC_s104
                                 + 1.08915 w_k1_KO_s104
                                 + 0.979444 w_k1_MAT_s104
                                 + 1.14597 w_k1_MSFT_s104
                                 + 1.10179 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.792002 w_k1_ADM_s105
                                 + 0.824636 w_k1_AXP_s105
                                 + 0.883532 w_k1_DIS_s105
                                 + 0.894415 w_k1_FDX_s105
                                 + 1.10733 w_k1_IBM_s105
                                 + 1.04424 w_k1_INTC_s105
                                 + 1.16458 w_k1_KO_s105 + 1.03304 w_k1_MAT_s105
                                 + 1.02178 w_k1_MSFT_s105
                                 + 0.99492 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.830697 w_k1_ADM_s105
                                 + 0.992584 w_k1_AXP_s105
                                 + 0.883838 w_k1_DIS_s105
                                 + 1.00239 w_k1_FDX_s105
                                 + 0.849552 w_k1_IBM_s105
                                 + 1.07623 w_k1_INTC_s105
                                 + 0.80916 w_k1_KO_s105
                                 + 0.724331 w_k1_MAT_s105
                                 + 0.989077 w_k1_MSFT_s105
                                 + 0.85411 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.18279 w_k1_ADM_s105
                                 + 1.51602 w_k1_AXP_s105
                                 + 1.45813 w_k1_DIS_s105
                                 + 1.39298 w_k1_FDX_s105
                                 + 0.985417 w_k1_IBM_s105
                                 + 1.322 w_k1_INTC_s105 + 1.2886 w_k1_KO_s105
                                 + 1.03743 w_k1_MAT_s105
                                 + 1.27519 w_k1_MSFT_s105
                                 + 1.07884 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.726812 w_k1_ADM_s105
                                 + 0.698211 w_k1_AXP_s105
                                 + 0.7462 w_k1_DIS_s105
                                 + 0.804904 w_k1_FDX_s105
                                 + 0.816245 w_k1_IBM_s105
                                 + 0.937987 w_k1_INTC_s105
                                 + 0.943618 w_k1_KO_s105
                                 + 0.714167 w_k1_MAT_s105
                                 + 0.886215 w_k1_MSFT_s105
                                 + 0.767546 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.919968 w_k1_ADM_s105
                                 + 1.23394 w_k1_AXP_s105
                                 + 1.09812 w_k1_DIS_s105
                                 + 1.13623 w_k1_FDX_s105
                                 + 0.94398 w_k1_IBM_s105
                                 + 1.16758 w_k1_INTC_s105
                                 + 1.10767 w_k1_KO_s105
                                 + 0.943165 w_k1_MAT_s105
                                 + 1.09595 w_k1_MSFT_s105
                                 + 1.10046 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.07323 w_k1_ADM_s105
                                 + 1.33268 w_k1_AXP_s105
                                 + 1.40702 w_k1_DIS_s105
                                 + 1.22472 w_k1_FDX_s105
                                 + 1.47634 w_k1_IBM_s105
                                 + 1.26056 w_k1_INTC_s105
                                 + 1.32599 w_k1_KO_s105 + 1.27846 w_k1_MAT_s105
                                 + 1.20362 w_k1_MSFT_s105
                                 + 1.14818 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.25656 w_k1_ADM_s105
                                 + 1.46606 w_k1_AXP_s105
                                 + 1.62489 w_k1_DIS_s105
                                 + 1.11195 w_k1_FDX_s105
                                 + 1.70661 w_k1_IBM_s105
                                 + 1.27474 w_k1_INTC_s105
                                 + 2.16869 w_k1_KO_s105 + 2.29679 w_k1_MAT_s105
                                 + 1.35239 w_k1_MSFT_s105
                                 + 1.15525 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.776956 w_k1_ADM_s105
                                 + 1.03429 w_k1_AXP_s105
                                 + 0.845374 w_k1_DIS_s105
                                 + 0.922722 w_k1_FDX_s105
                                 + 1.17344 w_k1_IBM_s105
                                 + 1.03551 w_k1_INTC_s105
                                 + 1.13755 w_k1_KO_s105 + 1.02423 w_k1_MAT_s105
                                 + 0.942286 w_k1_MSFT_s105
                                 + 0.925794 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.671246 w_k1_ADM_s106
                                 + 0.851892 w_k1_AXP_s106
                                 + 1.07294 w_k1_DIS_s106
                                 + 0.929297 w_k1_FDX_s106
                                 + 1.07357 w_k1_IBM_s106
                                 + 0.772113 w_k1_INTC_s106
                                 + 1.01529 w_k1_KO_s106 + 1.44022 w_k1_MAT_s106
                                 + 0.91689 w_k1_MSFT_s106
                                 + 1.05702 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.810942 w_k1_ADM_s106
                                 + 1.41997 w_k1_AXP_s106
                                 + 0.935017 w_k1_DIS_s106
                                 + 1.03782 w_k1_FDX_s106
                                 + 1.39622 w_k1_IBM_s106
                                 + 1.02741 w_k1_INTC_s106
                                 + 1.15763 w_k1_KO_s106 + 3.05031 w_k1_MAT_s106
                                 + 1.1167 w_k1_MSFT_s106 + 1.12138 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.14449 w_k1_ADM_s106
                                 + 1.17397 w_k1_AXP_s106
                                 + 0.88966 w_k1_DIS_s106 + 1.109 w_k1_FDX_s106
                                 + 0.846448 w_k1_IBM_s106
                                 + 1.16667 w_k1_INTC_s106
                                 + 1.01323 w_k1_KO_s106
                                 + 0.912714 w_k1_MAT_s106
                                 + 0.907017 w_k1_MSFT_s106
                                 + 1.02539 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.665126 w_k1_ADM_s106
                                 + 0.881354 w_k1_AXP_s106
                                 + 0.71221 w_k1_DIS_s106
                                 + 0.75242 w_k1_FDX_s106
                                 + 1.25907 w_k1_IBM_s106
                                 + 0.823929 w_k1_INTC_s106
                                 + 0.991557 w_k1_KO_s106
                                 + 1.67929 w_k1_MAT_s106
                                 + 0.767633 w_k1_MSFT_s106
                                 + 1.04081 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.94332 w_k1_ADM_s106
                                 + 0.928794 w_k1_AXP_s106
                                 + 0.63678 w_k1_DIS_s106
                                 + 0.847558 w_k1_FDX_s106
                                 + 0.593251 w_k1_IBM_s106
                                 + 0.931759 w_k1_INTC_s106
                                 + 0.813569 w_k1_KO_s106
                                 + 0.949193 w_k1_MAT_s106
                                 + 0.807144 w_k1_MSFT_s106
                                 + 1.03011 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.11859 w_k1_ADM_s106
                                 + 1.29546 w_k1_AXP_s106
                                 + 1.01258 w_k1_DIS_s106
                                 + 1.32078 w_k1_FDX_s106
                                 + 0.809502 w_k1_IBM_s106
                                 + 1.00424 w_k1_INTC_s106
                                 + 0.921188 w_k1_KO_s106
                                 + 0.900626 w_k1_MAT_s106
                                 + 0.92714 w_k1_MSFT_s106
                                 + 1.08675 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.07225 w_k1_ADM_s106
                                 + 1.31827 w_k1_AXP_s106
                                 + 1.41781 w_k1_DIS_s106
                                 + 1.33854 w_k1_FDX_s106
                                 + 0.984296 w_k1_IBM_s106
                                 + 1.32143 w_k1_INTC_s106
                                 + 1.24251 w_k1_KO_s106 + 1.02254 w_k1_MAT_s106
                                 + 1.17213 w_k1_MSFT_s106
                                 + 1.16906 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.984061 w_k1_ADM_s106
                                 + 1.79451 w_k1_AXP_s106
                                 + 1.65364 w_k1_DIS_s106
                                 + 1.50482 w_k1_FDX_s106
                                 + 1.50328 w_k1_IBM_s106
                                 + 1.16396 w_k1_INTC_s106
                                 + 1.56981 w_k1_KO_s106 + 2.13562 w_k1_MAT_s106
                                 + 1.3834 w_k1_MSFT_s106 + 1.60058 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.853684 w_k1_ADM_s107
                                 + 0.795907 w_k1_AXP_s107
                                 + 0.671202 w_k1_DIS_s107
                                 + 0.803133 w_k1_FDX_s107
                                 + 0.694418 w_k1_IBM_s107
                                 + 0.919962 w_k1_INTC_s107
                                 + 0.721888 w_k1_KO_s107
                                 + 0.841045 w_k1_MAT_s107
                                 + 0.902742 w_k1_MSFT_s107
                                 + 0.885745 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.10782 w_k1_ADM_s107
                                 + 1.11156 w_k1_AXP_s107
                                 + 0.825617 w_k1_DIS_s107
                                 + 0.841775 w_k1_FDX_s107
                                 + 1.44616 w_k1_IBM_s107
                                 + 1.01587 w_k1_INTC_s107
                                 + 1.14207 w_k1_KO_s107 + 1.15131 w_k1_MAT_s107
                                 + 0.85088 w_k1_MSFT_s107
                                 + 1.08149 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.58377 w_k1_ADM_s107
                                 + 1.88056 w_k1_AXP_s107
                                 + 1.69959 w_k1_DIS_s107
                                 + 1.47186 w_k1_FDX_s107
                                 + 1.34758 w_k1_IBM_s107
                                 + 1.4051 w_k1_INTC_s107 + 1.29019 w_k1_KO_s107
                                 + 1.16498 w_k1_MAT_s107
                                 + 1.16169 w_k1_MSFT_s107
                                 + 1.31769 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.905012 w_k1_ADM_s107
                                 + 0.87532 w_k1_AXP_s107
                                 + 0.973038 w_k1_DIS_s107
                                 + 0.990665 w_k1_FDX_s107
                                 + 1.05855 w_k1_IBM_s107
                                 + 1.01864 w_k1_INTC_s107
                                 + 0.895382 w_k1_KO_s107
                                 + 0.850413 w_k1_MAT_s107
                                 + 0.977457 w_k1_MSFT_s107
                                 + 1.01221 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.04844 w_k1_ADM_s107
                                 + 1.04363 w_k1_AXP_s107
                                 + 1.18479 w_k1_DIS_s107 + 1.2102 w_k1_FDX_s107
                                 + 0.7587 w_k1_IBM_s107
                                 + 1.34418 w_k1_INTC_s107
                                 + 1.12038 w_k1_KO_s107 + 1.07463 w_k1_MAT_s107
                                 + 1.30516 w_k1_MSFT_s107
                                 + 1.06423 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.27359 w_k1_ADM_s107
                                 + 1.67679 w_k1_AXP_s107
                                 + 1.11845 w_k1_DIS_s107
                                 + 1.18136 w_k1_FDX_s107
                                 + 1.10635 w_k1_IBM_s107
                                 + 1.01668 w_k1_INTC_s107
                                 + 1.04422 w_k1_KO_s107 + 1.47872 w_k1_MAT_s107
                                 + 1.01977 w_k1_MSFT_s107
                                 + 1.16539 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.28204 w_k1_ADM_s107
                                 + 1.33785 w_k1_AXP_s107
                                 + 1.56328 w_k1_DIS_s107
                                 + 1.20019 w_k1_FDX_s107
                                 + 1.69512 w_k1_IBM_s107
                                 + 1.28783 w_k1_INTC_s107
                                 + 1.44519 w_k1_KO_s107 + 1.31542 w_k1_MAT_s107
                                 + 1.27438 w_k1_MSFT_s107
                                 + 1.20288 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.04323 w_k1_ADM_s107
                                 + 1.0536 w_k1_AXP_s107 + 0.81031 w_k1_DIS_s107
                                 + 0.897675 w_k1_FDX_s107
                                 + 0.953411 w_k1_IBM_s107
                                 + 0.658351 w_k1_INTC_s107
                                 + 0.836448 w_k1_KO_s107
                                 + 1.57605 w_k1_MAT_s107
                                 + 0.810544 w_k1_MSFT_s107
                                 + 1.03751 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.02473 w_k1_ADM_s110
                                 + 0.861973 w_k1_AXP_s110
                                 + 1.51985 w_k1_DIS_s110
                                 + 1.14408 w_k1_FDX_s110
                                 + 1.24606 w_k1_IBM_s110
                                 + 1.05957 w_k1_INTC_s110
                                 + 1.30037 w_k1_KO_s110
                                 + 0.861107 w_k1_MAT_s110
                                 + 1.14647 w_k1_MSFT_s110
                                 + 1.10762 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.95561 w_k1_ADM_s110
                                 + 1.00426 w_k1_AXP_s110
                                 + 1.11863 w_k1_DIS_s110
                                 + 1.07752 w_k1_FDX_s110
                                 + 0.95406 w_k1_IBM_s110
                                 + 1.15639 w_k1_INTC_s110
                                 + 1.03585 w_k1_KO_s110
                                 + 0.855169 w_k1_MAT_s110
                                 + 1.03435 w_k1_MSFT_s110
                                 + 0.879666 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.725958 w_k1_ADM_s110
                                 + 0.747754 w_k1_AXP_s110
                                 + 0.793013 w_k1_DIS_s110
                                 + 0.87043 w_k1_FDX_s110
                                 + 0.664818 w_k1_IBM_s110
                                 + 0.971016 w_k1_INTC_s110
                                 + 0.755939 w_k1_KO_s110
                                 + 0.685459 w_k1_MAT_s110
                                 + 0.832351 w_k1_MSFT_s110
                                 + 0.865902 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.717673 w_k1_ADM_s110
                                 + 0.864784 w_k1_AXP_s110
                                 + 0.81489 w_k1_DIS_s110
                                 + 0.845571 w_k1_FDX_s110
                                 + 1.32596 w_k1_IBM_s110
                                 + 0.976937 w_k1_INTC_s110
                                 + 1.05162 w_k1_KO_s110 + 1.03991 w_k1_MAT_s110
                                 + 0.795058 w_k1_MSFT_s110
                                 + 0.947137 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.857779 w_k1_ADM_s110
                                 + 1.02612 w_k1_AXP_s110
                                 + 0.87427 w_k1_DIS_s110
                                 + 0.959845 w_k1_FDX_s110
                                 + 1.1527 w_k1_IBM_s110
                                 + 0.968778 w_k1_INTC_s110
                                 + 0.912853 w_k1_KO_s110
                                 + 1.13388 w_k1_MAT_s110
                                 + 0.840257 w_k1_MSFT_s110
                                 + 1.00952 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.13231 w_k1_ADM_s110
                                 + 1.5854 w_k1_AXP_s110 + 1.13532 w_k1_DIS_s110
                                 + 1.22912 w_k1_FDX_s110
                                 + 1.27764 w_k1_IBM_s110
                                 + 1.04216 w_k1_INTC_s110
                                 + 0.978306 w_k1_KO_s110
                                 + 1.40474 w_k1_MAT_s110
                                 + 0.974836 w_k1_MSFT_s110
                                 + 1.04931 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.720852 w_k1_ADM_s110
                                 + 0.812907 w_k1_AXP_s110
                                 + 0.606902 w_k1_DIS_s110
                                 + 0.763675 w_k1_FDX_s110
                                 + 0.832779 w_k1_IBM_s110
                                 + 0.76055 w_k1_INTC_s110
                                 + 0.664868 w_k1_KO_s110
                                 + 0.652815 w_k1_MAT_s110
                                 + 0.617974 w_k1_MSFT_s110
                                 + 0.750034 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.11213 w_k1_ADM_s110
                                 + 1.22525 w_k1_AXP_s110
                                 + 1.59434 w_k1_DIS_s110
                                 + 1.32249 w_k1_FDX_s110
                                 + 0.852494 w_k1_IBM_s110
                                 + 1.38609 w_k1_INTC_s110
                                 + 1.12506 w_k1_KO_s110
                                 + 0.885578 w_k1_MAT_s110
                                 + 1.26861 w_k1_MSFT_s110
                                 + 1.03861 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.24996 w_k1_ADM_s111
                                 + 1.31282 w_k1_AXP_s111
                                 + 1.88944 w_k1_DIS_s111
                                 + 1.41384 w_k1_FDX_s111
                                 + 1.32311 w_k1_IBM_s111
                                 + 1.07402 w_k1_INTC_s111 + 1.427 w_k1_KO_s111
                                 + 1.62758 w_k1_MAT_s111
                                 + 1.3987 w_k1_MSFT_s111 + 1.15924 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.27286 w_k1_ADM_s111
                                 + 1.31255 w_k1_AXP_s111 + 1.4741 w_k1_DIS_s111
                                 + 1.54207 w_k1_FDX_s111
                                 + 0.799858 w_k1_IBM_s111
                                 + 1.16749 w_k1_INTC_s111
                                 + 0.990684 w_k1_KO_s111
                                 + 0.978035 w_k1_MAT_s111
                                 + 1.35425 w_k1_MSFT_s111
                                 + 1.06709 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.665912 w_k1_ADM_s111
                                 + 0.675414 w_k1_AXP_s111
                                 + 0.715462 w_k1_DIS_s111
                                 + 0.727337 w_k1_FDX_s111
                                 + 0.787937 w_k1_IBM_s111
                                 + 0.667958 w_k1_INTC_s111
                                 + 0.705926 w_k1_KO_s111
                                 + 0.929371 w_k1_MAT_s111
                                 + 0.845974 w_k1_MSFT_s111
                                 + 0.874769 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.835538 w_k1_ADM_s111
                                 + 0.819839 w_k1_AXP_s111
                                 + 0.736755 w_k1_DIS_s111
                                 + 0.794632 w_k1_FDX_s111
                                 + 1.10063 w_k1_IBM_s111
                                 + 0.8247 w_k1_INTC_s111
                                 + 0.886116 w_k1_KO_s111
                                 + 1.07362 w_k1_MAT_s111
                                 + 1.03189 w_k1_MSFT_s111
                                 + 0.914595 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.0616 w_k1_ADM_s111
                                 + 1.31913 w_k1_AXP_s111
                                 + 1.10967 w_k1_DIS_s111
                                 + 1.12025 w_k1_FDX_s111
                                 + 1.05328 w_k1_IBM_s111
                                 + 0.8817 w_k1_INTC_s111
                                 + 0.943663 w_k1_KO_s111
                                 + 1.49953 w_k1_MAT_s111
                                 + 1.07505 w_k1_MSFT_s111
                                 + 0.985392 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.920669 w_k1_ADM_s111
                                 + 0.913936 w_k1_AXP_s111
                                 + 1.09058 w_k1_DIS_s111
                                 + 1.05653 w_k1_FDX_s111
                                 + 0.710748 w_k1_IBM_s111
                                 + 0.898724 w_k1_INTC_s111
                                 + 0.902833 w_k1_KO_s111
                                 + 0.957399 w_k1_MAT_s111
                                 + 1.2591 w_k1_MSFT_s111
                                 + 0.969662 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.713507 w_k1_ADM_s111
                                 + 0.713036 w_k1_AXP_s111
                                 + 0.732591 w_k1_DIS_s111
                                 + 0.841372 w_k1_FDX_s111
                                 + 0.68466 w_k1_IBM_s111
                                 + 0.701522 w_k1_INTC_s111
                                 + 0.629923 w_k1_KO_s111
                                 + 0.683788 w_k1_MAT_s111
                                 + 0.797317 w_k1_MSFT_s111
                                 + 0.78627 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.728627 w_k1_ADM_s111
                                 + 0.829787 w_k1_AXP_s111
                                 + 0.915172 w_k1_DIS_s111
                                 + 0.738618 w_k1_FDX_s111
                                 + 1.58468 w_k1_IBM_s111
                                 + 0.937016 w_k1_INTC_s111
                                 + 1.04261 w_k1_KO_s111 + 1.60928 w_k1_MAT_s111
                                 + 1.00951 w_k1_MSFT_s111
                                 + 0.978028 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.0288 w_k1_ADM_s112
                                 + 1.09061 w_k1_AXP_s112
                                 + 0.873861 w_k1_DIS_s112
                                 + 1.0676 w_k1_FDX_s112
                                 + 0.752423 w_k1_IBM_s112
                                 + 0.940144 w_k1_INTC_s112
                                 + 0.861999 w_k1_KO_s112
                                 + 0.675493 w_k1_MAT_s112
                                 + 0.950688 w_k1_MSFT_s112
                                 + 0.904217 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.09964 w_k1_ADM_s112
                                 + 1.05489 w_k1_AXP_s112
                                 + 0.90838 w_k1_DIS_s112
                                 + 0.841485 w_k1_FDX_s112
                                 + 1.42398 w_k1_IBM_s112
                                 + 0.861285 w_k1_INTC_s112
                                 + 1.28625 w_k1_KO_s112 + 1.00242 w_k1_MAT_s112
                                 + 0.871651 w_k1_MSFT_s112
                                 + 1.05531 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.07192 w_k1_ADM_s112
                                 + 1.0845 w_k1_AXP_s112 + 1.17655 w_k1_DIS_s112
                                 + 1.07572 w_k1_FDX_s112
                                 + 1.19458 w_k1_IBM_s112
                                 + 1.14308 w_k1_INTC_s112
                                 + 1.16095 w_k1_KO_s112
                                 + 0.822631 w_k1_MAT_s112
                                 + 1.01769 w_k1_MSFT_s112
                                 + 1.02891 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.34397 w_k1_ADM_s112
                                 + 1.48159 w_k1_AXP_s112
                                 + 1.38926 w_k1_DIS_s112
                                 + 1.27679 w_k1_FDX_s112
                                 + 0.96711 w_k1_IBM_s112
                                 + 0.942427 w_k1_INTC_s112
                                 + 1.1714 w_k1_KO_s112 + 1.07569 w_k1_MAT_s112
                                 + 1.11005 w_k1_MSFT_s112
                                 + 1.16739 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.29739 w_k1_ADM_s112
                                 + 1.57571 w_k1_AXP_s112
                                 + 1.79431 w_k1_DIS_s112
                                 + 1.35188 w_k1_FDX_s112 + 1.3789 w_k1_IBM_s112
                                 + 1.11436 w_k1_INTC_s112
                                 + 1.43031 w_k1_KO_s112 + 1.47251 w_k1_MAT_s112
                                 + 1.34603 w_k1_MSFT_s112
                                 + 1.18196 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.941495 w_k1_ADM_s112
                                 + 0.77438 w_k1_AXP_s112
                                 + 0.620526 w_k1_DIS_s112
                                 + 0.763932 w_k1_FDX_s112
                                 + 0.85444 w_k1_IBM_s112
                                 + 0.83844 w_k1_INTC_s112 + 0.7742 w_k1_KO_s112
                                 + 0.692089 w_k1_MAT_s112
                                 + 0.893023 w_k1_MSFT_s112
                                 + 0.882883 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.43559 w_k1_ADM_s112
                                 + 1.42335 w_k1_AXP_s112
                                 + 1.22154 w_k1_DIS_s112
                                 + 1.36595 w_k1_FDX_s112
                                 + 0.87425 w_k1_IBM_s112
                                 + 1.24347 w_k1_INTC_s112
                                 + 1.12502 w_k1_KO_s112
                                 + 0.714939 w_k1_MAT_s112
                                 + 1.10819 w_k1_MSFT_s112
                                 + 1.16434 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.921211 w_k1_ADM_s112
                                 + 0.886942 w_k1_AXP_s112
                                 + 0.899355 w_k1_DIS_s112
                                 + 0.903335 w_k1_FDX_s112
                                 + 0.934736 w_k1_IBM_s112
                                 + 1.07201 w_k1_INTC_s112
                                 + 0.960653 w_k1_KO_s112
                                 + 0.758839 w_k1_MAT_s112
                                 + 0.955284 w_k1_MSFT_s112
                                 + 0.88917 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.11136 w_k1_ADM_s113
                                 + 1.06952 w_k1_AXP_s113
                                 + 0.990003 w_k1_DIS_s113
                                 + 0.963934 w_k1_FDX_s113
                                 + 1.01553 w_k1_IBM_s113
                                 + 0.929288 w_k1_INTC_s113
                                 + 1.10463 w_k1_KO_s113
                                 + 0.988078 w_k1_MAT_s113
                                 + 0.899222 w_k1_MSFT_s113
                                 + 0.944364 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.01933 w_k1_ADM_s113
                                 + 0.784038 w_k1_AXP_s113 + 1.02 w_k1_DIS_s113
                                 + 0.887119 w_k1_FDX_s113
                                 + 1.03412 w_k1_IBM_s113
                                 + 0.95292 w_k1_INTC_s113
                                 + 1.07284 w_k1_KO_s113
                                 + 0.856194 w_k1_MAT_s113
                                 + 0.925326 w_k1_MSFT_s113
                                 + 1.04535 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.55307 w_k1_ADM_s113
                                 + 1.49925 w_k1_AXP_s113
                                 + 1.25163 w_k1_DIS_s113
                                 + 1.20502 w_k1_FDX_s113
                                 + 1.30058 w_k1_IBM_s113
                                 + 1.19099 w_k1_INTC_s113
                                 + 1.21083 w_k1_KO_s113 + 1.3091 w_k1_MAT_s113
                                 + 1.14665 w_k1_MSFT_s113
                                 + 1.09748 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.785768 w_k1_ADM_s113
                                 + 0.56408 w_k1_AXP_s113
                                 + 0.496537 w_k1_DIS_s113
                                 + 0.668013 w_k1_FDX_s113
                                 + 0.68295 w_k1_IBM_s113
                                 + 0.664753 w_k1_INTC_s113
                                 + 0.689663 w_k1_KO_s113
                                 + 0.552609 w_k1_MAT_s113
                                 + 0.651539 w_k1_MSFT_s113
                                 + 0.686505 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.31813 w_k1_ADM_s113
                                 + 1.19046 w_k1_AXP_s113 + 1.4186 w_k1_DIS_s113
                                 + 1.33782 w_k1_FDX_s113
                                 + 0.846147 w_k1_IBM_s113
                                 + 1.07205 w_k1_INTC_s113
                                 + 1.17282 w_k1_KO_s113
                                 + 0.828497 w_k1_MAT_s113
                                 + 1.19703 w_k1_MSFT_s113
                                 + 0.990573 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.16667 w_k1_ADM_s113
                                 + 0.76599 w_k1_AXP_s113
                                 + 0.830496 w_k1_DIS_s113
                                 + 0.97296 w_k1_FDX_s113
                                 + 0.637096 w_k1_IBM_s113
                                 + 0.964275 w_k1_INTC_s113
                                 + 0.798383 w_k1_KO_s113
                                 + 0.408894 w_k1_MAT_s113
                                 + 0.858619 w_k1_MSFT_s113
                                 + 0.901105 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.907238 w_k1_ADM_s113
                                 + 0.690795 w_k1_AXP_s113
                                 + 0.711911 w_k1_DIS_s113
                                 + 0.813298 w_k1_FDX_s113
                                 + 0.812425 w_k1_IBM_s113
                                 + 1.02196 w_k1_INTC_s113
                                 + 0.846296 w_k1_KO_s113
                                 + 0.658669 w_k1_MAT_s113
                                 + 0.947823 w_k1_MSFT_s113
                                 + 0.904705 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.909059 w_k1_ADM_s113
                                 + 1.11357 w_k1_AXP_s113
                                 + 1.27104 w_k1_DIS_s113
                                 + 1.00673 w_k1_FDX_s113 + 1.7393 w_k1_IBM_s113
                                 + 0.957162 w_k1_INTC_s113
                                 + 1.34218 w_k1_KO_s113 + 1.67032 w_k1_MAT_s113
                                 + 0.958569 w_k1_MSFT_s113
                                 + 1.12334 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.915406 w_k1_ADM_s114
                                 + 0.828462 w_k1_AXP_s114
                                 + 0.776768 w_k1_DIS_s114
                                 + 0.826844 w_k1_FDX_s114
                                 + 0.989929 w_k1_IBM_s114
                                 + 0.759191 w_k1_INTC_s114
                                 + 0.928577 w_k1_KO_s114
                                 + 1.33341 w_k1_MAT_s114
                                 + 0.843817 w_k1_MSFT_s114
                                 + 0.903618 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.33553 w_k1_ADM_s114
                                 + 1.60713 w_k1_AXP_s114
                                 + 1.64871 w_k1_DIS_s114
                                 + 1.28878 w_k1_FDX_s114
                                 + 1.70306 w_k1_IBM_s114
                                 + 1.3595 w_k1_INTC_s114 + 1.77452 w_k1_KO_s114
                                 + 2.40651 w_k1_MAT_s114
                                 + 1.29393 w_k1_MSFT_s114
                                 + 1.12231 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.93459 w_k1_ADM_s114
                                 + 1.12481 w_k1_AXP_s114
                                 + 1.11193 w_k1_DIS_s114
                                 + 0.942609 w_k1_FDX_s114
                                 + 1.29905 w_k1_IBM_s114
                                 + 0.865428 w_k1_INTC_s114
                                 + 1.09398 w_k1_KO_s114 + 2.24631 w_k1_MAT_s114
                                 + 0.978793 w_k1_MSFT_s114
                                 + 0.995322 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.02713 w_k1_ADM_s114
                                 + 1.11799 w_k1_AXP_s114
                                 + 0.919249 w_k1_DIS_s114
                                 + 1.03121 w_k1_FDX_s114
                                 + 0.887204 w_k1_IBM_s114
                                 + 0.956677 w_k1_INTC_s114
                                 + 1.06047 w_k1_KO_s114 + 1.23262 w_k1_MAT_s114
                                 + 1.0251 w_k1_MSFT_s114 + 1.0345 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.945294 w_k1_ADM_s114
                                 + 1.24928 w_k1_AXP_s114
                                 + 0.840752 w_k1_DIS_s114
                                 + 1.0874 w_k1_FDX_s114 + 1.38976 w_k1_IBM_s114
                                 + 1.20171 w_k1_INTC_s114
                                 + 1.02084 w_k1_KO_s114 + 1.71715 w_k1_MAT_s114
                                 + 1.05289 w_k1_MSFT_s114
                                 + 1.00763 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.851207 w_k1_ADM_s114
                                 + 0.77949 w_k1_AXP_s114
                                 + 0.698322 w_k1_DIS_s114
                                 + 0.796935 w_k1_FDX_s114
                                 + 0.774676 w_k1_IBM_s114
                                 + 0.821897 w_k1_INTC_s114
                                 + 0.676341 w_k1_KO_s114
                                 + 0.739674 w_k1_MAT_s114
                                 + 0.753447 w_k1_MSFT_s114
                                 + 0.803344 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.4421 w_k1_ADM_s114
                                 + 1.39048 w_k1_AXP_s114 + 1.6173 w_k1_DIS_s114
                                 + 1.43997 w_k1_FDX_s114
                                 + 1.14857 w_k1_IBM_s114
                                 + 1.5566 w_k1_INTC_s114 + 1.29087 w_k1_KO_s114
                                 + 1.64511 w_k1_MAT_s114 + 1.407 w_k1_MSFT_s114
                                 + 1.12986 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.22183 w_k1_ADM_s114
                                 + 1.17576 w_k1_AXP_s114
                                 + 1.29556 w_k1_DIS_s114
                                 + 1.17145 w_k1_FDX_s114
                                 + 1.02522 w_k1_IBM_s114
                                 + 0.949498 w_k1_INTC_s114
                                 + 1.05113 w_k1_KO_s114 + 1.36565 w_k1_MAT_s114
                                 + 1.09677 w_k1_MSFT_s114
                                 + 1.21479 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.00362 w_k1_ADM_s115
                                 + 0.935189 w_k1_AXP_s115
                                 + 1.02644 w_k1_DIS_s115
                                 + 1.07899 w_k1_FDX_s115
                                 + 0.883791 w_k1_IBM_s115
                                 + 1.25258 w_k1_INTC_s115
                                 + 1.01976 w_k1_KO_s115
                                 + 0.984322 w_k1_MAT_s115
                                 + 1.05592 w_k1_MSFT_s115
                                 + 1.08242 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.739425 w_k1_ADM_s115
                                 + 0.731965 w_k1_AXP_s115
                                 + 0.735459 w_k1_DIS_s115
                                 + 0.842213 w_k1_FDX_s115
                                 + 0.790167 w_k1_IBM_s115
                                 + 1.01483 w_k1_INTC_s115
                                 + 0.767039 w_k1_KO_s115
                                 + 0.732464 w_k1_MAT_s115
                                 + 0.769129 w_k1_MSFT_s115
                                 + 0.826217 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.25548 w_k1_ADM_s115
                                 + 1.39691 w_k1_AXP_s115
                                 + 1.52043 w_k1_DIS_s115
                                 + 1.18048 w_k1_FDX_s115
                                 + 1.53059 w_k1_IBM_s115
                                 + 1.24848 w_k1_INTC_s115
                                 + 1.35546 w_k1_KO_s115 + 1.79983 w_k1_MAT_s115
                                 + 1.13433 w_k1_MSFT_s115
                                 + 1.11373 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.723591 w_k1_ADM_s115
                                 + 0.851744 w_k1_AXP_s115
                                 + 0.721515 w_k1_DIS_s115
                                 + 0.75831 w_k1_FDX_s115
                                 + 1.23913 w_k1_IBM_s115
                                 + 1.13816 w_k1_INTC_s115
                                 + 0.937831 w_k1_KO_s115
                                 + 1.06457 w_k1_MAT_s115
                                 + 0.719533 w_k1_MSFT_s115
                                 + 0.898062 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.797063 w_k1_ADM_s115
                                 + 0.991257 w_k1_AXP_s115
                                 + 0.961752 w_k1_DIS_s115
                                 + 0.939176 w_k1_FDX_s115
                                 + 1.74654 w_k1_IBM_s115
                                 + 1.41027 w_k1_INTC_s115 + 1.2531 w_k1_KO_s115
                                 + 1.3218 w_k1_MAT_s115
                                 + 1.07942 w_k1_MSFT_s115
                                 + 1.05356 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.967377 w_k1_ADM_s115
                                 + 0.866702 w_k1_AXP_s115
                                 + 1.40268 w_k1_DIS_s115
                                 + 1.15398 w_k1_FDX_s115
                                 + 0.924193 w_k1_IBM_s115
                                 + 1.11207 w_k1_INTC_s115
                                 + 1.13336 w_k1_KO_s115
                                 + 0.896623 w_k1_MAT_s115
                                 + 1.15648 w_k1_MSFT_s115
                                 + 1.00711 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.983887 w_k1_ADM_s115
                                 + 1.18459 w_k1_AXP_s115
                                 + 1.11217 w_k1_DIS_s115
                                 + 1.00102 w_k1_FDX_s115 + 1.2431 w_k1_IBM_s115
                                 + 1.06717 w_k1_INTC_s115
                                 + 1.05806 w_k1_KO_s115 + 1.28149 w_k1_MAT_s115
                                 + 1.05055 w_k1_MSFT_s115
                                 + 0.979887 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.27075 w_k1_ADM_s115
                                 + 1.47905 w_k1_AXP_s115
                                 + 1.47194 w_k1_DIS_s115
                                 + 1.38084 w_k1_FDX_s115
                                 + 0.943833 w_k1_IBM_s115
                                 + 1.30411 w_k1_INTC_s115
                                 + 1.20163 w_k1_KO_s115 + 1.0815 w_k1_MAT_s115
                                 + 1.21019 w_k1_MSFT_s115
                                 + 1.14381 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.05246 w_k1_ADM_s116
                                 + 0.829591 w_k1_AXP_s116
                                 + 0.830578 w_k1_DIS_s116
                                 + 0.990744 w_k1_FDX_s116
                                 + 0.675325 w_k1_IBM_s116
                                 + 1.00503 w_k1_INTC_s116
                                 + 0.774851 w_k1_KO_s116
                                 + 0.637773 w_k1_MAT_s116
                                 + 0.919469 w_k1_MSFT_s116
                                 + 0.852148 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.54862 w_k1_ADM_s116
                                 + 1.45191 w_k1_AXP_s116
                                 + 1.70044 w_k1_DIS_s116
                                 + 1.62044 w_k1_FDX_s116
                                 + 1.13037 w_k1_IBM_s116
                                 + 1.54093 w_k1_INTC_s116
                                 + 1.54318 w_k1_KO_s116 + 1.44989 w_k1_MAT_s116
                                 + 1.46796 w_k1_MSFT_s116
                                 + 1.36189 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.24772 w_k1_ADM_s116
                                 + 1.04025 w_k1_AXP_s116 + 1.181 w_k1_DIS_s116
                                 + 1.26367 w_k1_FDX_s116
                                 + 0.73914 w_k1_IBM_s116
                                 + 1.09533 w_k1_INTC_s116
                                 + 1.01783 w_k1_KO_s116 + 0.45113 w_k1_MAT_s116
                                 + 1.12831 w_k1_MSFT_s116
                                 + 1.06506 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.945031 w_k1_ADM_s116
                                 + 0.844647 w_k1_AXP_s116
                                 + 0.854656 w_k1_DIS_s116
                                 + 0.844194 w_k1_FDX_s116
                                 + 1.39989 w_k1_IBM_s116
                                 + 0.864269 w_k1_INTC_s116
                                 + 1.16561 w_k1_KO_s116 + 1.06447 w_k1_MAT_s116
                                 + 0.827856 w_k1_MSFT_s116
                                 + 1.07783 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.832282 w_k1_ADM_s116
                                 + 0.723261 w_k1_AXP_s116
                                 + 0.760979 w_k1_DIS_s116
                                 + 0.860299 w_k1_FDX_s116
                                 + 0.810287 w_k1_IBM_s116
                                 + 1.08641 w_k1_INTC_s116
                                 + 0.87525 w_k1_KO_s116
                                 + 0.815212 w_k1_MAT_s116
                                 + 0.977951 w_k1_MSFT_s116
                                 + 1.0142 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.25448 w_k1_ADM_s116
                                 + 1.0583 w_k1_AXP_s116 + 1.21711 w_k1_DIS_s116
                                 + 1.1701 w_k1_FDX_s116 + 1.29457 w_k1_IBM_s116
                                 + 1.03952 w_k1_INTC_s116
                                 + 1.06513 w_k1_KO_s116 + 1.50175 w_k1_MAT_s116
                                 + 1.16288 w_k1_MSFT_s116
                                 + 1.20567 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.738965 w_k1_ADM_s116
                                 + 0.738839 w_k1_AXP_s116
                                 + 0.515379 w_k1_DIS_s116
                                 + 0.725701 w_k1_FDX_s116
                                 + 0.870276 w_k1_IBM_s116
                                 + 0.73594 w_k1_INTC_s116
                                 + 0.79847 w_k1_KO_s116 + 0.79587 w_k1_MAT_s116
                                 + 0.803615 w_k1_MSFT_s116
                                 + 0.821161 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.25646 w_k1_ADM_s116
                                 + 0.911279 w_k1_AXP_s116
                                 + 0.859357 w_k1_DIS_s116
                                 + 0.953118 w_k1_FDX_s116
                                 + 0.987633 w_k1_IBM_s116
                                 + 0.994587 w_k1_INTC_s116
                                 + 1.11013 w_k1_KO_s116
                                 + 0.932125 w_k1_MAT_s116
                                 + 1.05549 w_k1_MSFT_s116
                                 + 1.07161 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.21822 w_k1_ADM_s117
                                 + 1.6144 w_k1_AXP_s117 + 1.57932 w_k1_DIS_s117
                                 + 1.28067 w_k1_FDX_s117
                                 + 1.37535 w_k1_IBM_s117
                                 + 1.11086 w_k1_INTC_s117
                                 + 1.34144 w_k1_KO_s117 + 1.65327 w_k1_MAT_s117
                                 + 1.21212 w_k1_MSFT_s117
                                 + 1.25454 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.849678 w_k1_ADM_s117
                                 + 1.04059 w_k1_AXP_s117
                                 + 0.787475 w_k1_DIS_s117
                                 + 0.855119 w_k1_FDX_s117
                                 + 1.24092 w_k1_IBM_s117
                                 + 0.963439 w_k1_INTC_s117
                                 + 1.12784 w_k1_KO_s117 + 1.29939 w_k1_MAT_s117
                                 + 0.833359 w_k1_MSFT_s117
                                 + 0.951552 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.22948 w_k1_ADM_s117
                                 + 1.15802 w_k1_AXP_s117
                                 + 1.09289 w_k1_DIS_s117 + 1.0245 w_k1_FDX_s117
                                 + 1.16459 w_k1_IBM_s117
                                 + 0.980953 w_k1_INTC_s117
                                 + 1.21157 w_k1_KO_s117 + 1.18279 w_k1_MAT_s117
                                 + 1.01339 w_k1_MSFT_s117
                                 + 1.04299 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.22895 w_k1_ADM_s117
                                 + 1.37488 w_k1_AXP_s117
                                 + 1.38133 w_k1_DIS_s117
                                 + 1.38935 w_k1_FDX_s117
                                 + 0.868461 w_k1_IBM_s117
                                 + 1.26868 w_k1_INTC_s117
                                 + 1.01908 w_k1_KO_s117 + 0.96412 w_k1_MAT_s117
                                 + 1.22904 w_k1_MSFT_s117
                                 + 1.06491 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.805049 w_k1_ADM_s117
                                 + 1.12294 w_k1_AXP_s117
                                 + 1.08745 w_k1_DIS_s117
                                 + 0.960828 w_k1_FDX_s117
                                 + 1.73246 w_k1_IBM_s117
                                 + 1.1451 w_k1_INTC_s117 + 1.38479 w_k1_KO_s117
                                 + 1.82369 w_k1_MAT_s117
                                 + 1.10932 w_k1_MSFT_s117
                                 + 1.04868 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.12895 w_k1_ADM_s117
                                 + 1.13414 w_k1_AXP_s117
                                 + 1.09813 w_k1_DIS_s117
                                 + 1.08068 w_k1_FDX_s117
                                 + 0.901518 w_k1_IBM_s117
                                 + 1.02007 w_k1_INTC_s117
                                 + 0.890898 w_k1_KO_s117
                                 + 0.879051 w_k1_MAT_s117
                                 + 0.994134 w_k1_MSFT_s117
                                 + 0.963531 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.910313 w_k1_ADM_s117
                                 + 0.857551 w_k1_AXP_s117
                                 + 0.768663 w_k1_DIS_s117
                                 + 0.932358 w_k1_FDX_s117
                                 + 0.749207 w_k1_IBM_s117
                                 + 0.870344 w_k1_INTC_s117
                                 + 0.810196 w_k1_KO_s117
                                 + 0.691538 w_k1_MAT_s117
                                 + 0.840513 w_k1_MSFT_s117
                                 + 0.849404 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.12132 w_k1_ADM_s117
                                 + 1.37627 w_k1_AXP_s117
                                 + 1.20196 w_k1_DIS_s117
                                 + 1.22149 w_k1_FDX_s117 + 0.943 w_k1_IBM_s117
                                 + 1.12054 w_k1_INTC_s117
                                 + 0.960118 w_k1_KO_s117
                                 + 1.03158 w_k1_MAT_s117
                                 + 1.14459 w_k1_MSFT_s117
                                 + 1.03238 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0148195170325929 w_k1_ADM_s000
                                 + 0.0389328246033804 w_k1_ADM_s001
                                 + 0.0432406200187415 w_k1_ADM_s002
                                 + 0.0214553848184722 w_k1_ADM_s003
                                 + 0.0114892747711063 w_k1_ADM_s004
                                 + 0.0294381522872495 w_k1_ADM_s005
                                 + 0.0277049890497851 w_k1_ADM_s006
                                 + 0.0247275513915528 w_k1_ADM_s007
                                 + 0.0181604394330674 w_k1_ADM_s010
                                 + 0.00916438578777847 w_k1_ADM_s011
                                 + 0.0293461350096894 w_k1_ADM_s012
                                 + 0.0290042043318745 w_k1_ADM_s013
                                 + 0.0107778252126955 w_k1_ADM_s014
                                 + 0.0113706095587072 w_k1_ADM_s015
                                 + 0.0299328878222515 w_k1_ADM_s016
                                 + 0.0401180601728783 w_k1_ADM_s017
                                 + 0.0171425473835112 w_k1_ADM_s100
                                 + 0.0291982700685015 w_k1_ADM_s101
                                 + 0.0107857016680552 w_k1_ADM_s102
                                 + 0.010582927842129 w_k1_ADM_s103
                                 + 0.0173130333635062 w_k1_ADM_s104
                                 + 0.0141179659528888 w_k1_ADM_s105
                                 + 0.00558626462564617 w_k1_ADM_s106
                                 + 0.0303398242768187 w_k1_ADM_s107
                                 + 0.0813513110433601 w_k1_ADM_s110
                                 + 0.0753765552037 w_k1_ADM_s111
                                 + 0.0538402681770337 w_k1_ADM_s112
                                 + 0.0497589156158126 w_k1_ADM_s113
                                 + 0.0256892587646897 w_k1_ADM_s114
                                 + 0.0650564629353031 w_k1_ADM_s115
                                 + 0.0569624196167953 w_k1_ADM_s116
                                 + 0.0587790135142296 w_k1_ADM_s117
                                 + 0.0127422741990497 w_k1_AXP_s000
                                 + 0.0405273353897606 w_k1_AXP_s001
                                 + 0.0456742215044195 w_k1_AXP_s002
                                 + 0.0226005261667445 w_k1_AXP_s003
                                 + 0.0104260128435399 w_k1_AXP_s004
                                 + 0.031391156176824 w_k1_AXP_s005
                                 + 0.0314041407551353 w_k1_AXP_s006
                                 + 0.0241951657211482 w_k1_AXP_s007
                                 + 0.0213272219892864 w_k1_AXP_s010
                                 + 0.0107213275807849 w_k1_AXP_s011
                                 + 0.0258149235697478 w_k1_AXP_s012
                                 + 0.0260989348448028 w_k1_AXP_s013
                                 + 0.0105060682280208 w_k1_AXP_s014
                                 + 0.0115861928105859 w_k1_AXP_s015
                                 + 0.0322283296666191 w_k1_AXP_s016
                                 + 0.0420436091335286 w_k1_AXP_s017
                                 + 0.0149226966976625 w_k1_AXP_s100
                                 + 0.0316327964978259 w_k1_AXP_s101
                                 + 0.0106993539274479 w_k1_AXP_s102
                                 + 0.0132092543485588 w_k1_AXP_s103
                                 + 0.0179825733318886 w_k1_AXP_s104
                                 + 0.0169333205111959 w_k1_AXP_s105
                                 + 0.00735822510879283 w_k1_AXP_s106
                                 + 0.031702049792715 w_k1_AXP_s107
                                 + 0.0908204288675719 w_k1_AXP_s110
                                 + 0.0805236747238459 w_k1_AXP_s111
                                 + 0.0540501973441658 w_k1_AXP_s112
                                 + 0.0439377321329976 w_k1_AXP_s113
                                 + 0.0266585930608048 w_k1_AXP_s114
                                 + 0.0694022465517623 w_k1_AXP_s115
                                 + 0.0488167764000284 w_k1_AXP_s116
                                 + 0.06444006353403 w_k1_AXP_s117
                                 + 0.013109543455723 w_k1_DIS_s000
                                 + 0.0448134152826754 w_k1_DIS_s001
                                 + 0.0475996346816077 w_k1_DIS_s002
                                 + 0.0203251122060404 w_k1_DIS_s003
                                 + 0.0111071355821875 w_k1_DIS_s004
                                 + 0.0326433240149697 w_k1_DIS_s005
                                 + 0.0297031866592652 w_k1_DIS_s006
                                 + 0.0291339628064348 w_k1_DIS_s007
                                 + 0.0221831537168724 w_k1_DIS_s010
                                 + 0.0101467140312951 w_k1_DIS_s011
                                 + 0.0238663850645133 w_k1_DIS_s012
                                 + 0.0272659925642618 w_k1_DIS_s013
                                 + 0.00996816885372822 w_k1_DIS_s014
                                 + 0.0125448854982304 w_k1_DIS_s015
                                 + 0.0336147081594453 w_k1_DIS_s016
                                 + 0.0441878796149739 w_k1_DIS_s017
                                 + 0.0156162881115548 w_k1_DIS_s100
                                 + 0.0303728841245085 w_k1_DIS_s101
                                 + 0.0117322187755095 w_k1_DIS_s102
                                 + 0.0124320142948001 w_k1_DIS_s103
                                 + 0.0186949603055135 w_k1_DIS_s104
                                 + 0.0163805039602786 w_k1_DIS_s105
                                 + 0.00645139992688507 w_k1_DIS_s106
                                 + 0.0289830469192507 w_k1_DIS_s107
                                 + 0.0939846339693533 w_k1_DIS_s110
                                 + 0.0853577087333394 w_k1_DIS_s111
                                 + 0.0504915895952183 w_k1_DIS_s112
                                 + 0.0438318007798084 w_k1_DIS_s113
                                 + 0.0259483877180682 w_k1_DIS_s114
                                 + 0.0734048302506263 w_k1_DIS_s115
                                 + 0.050958964477059 w_k1_DIS_s116
                                 + 0.059316162101014 w_k1_DIS_s117
                                 + 0.0125490410298342 w_k1_FDX_s000
                                 + 0.0436107360143354 w_k1_FDX_s001
                                 + 0.047802300222617 w_k1_FDX_s002
                                 + 0.0215671173893045 w_k1_FDX_s003
                                 + 0.00993278101011487 w_k1_FDX_s004
                                 + 0.0336761328335798 w_k1_FDX_s005
                                 + 0.029266034770516 w_k1_FDX_s006
                                 + 0.0287421537462385 w_k1_FDX_s007
                                 + 0.0216527176414389 w_k1_FDX_s010
                                 + 0.0101036481210156 w_k1_FDX_s011
                                 + 0.0241048344103925 w_k1_FDX_s012
                                 + 0.026000825252419 w_k1_FDX_s013
                                 + 0.0109102164764797 w_k1_FDX_s014
                                 + 0.0120717221159215 w_k1_FDX_s015
                                 + 0.0312151896280613 w_k1_FDX_s016
                                 + 0.0434451446166132 w_k1_FDX_s017
                                 + 0.0159398338967251 w_k1_FDX_s100
                                 + 0.0301879881156553 w_k1_FDX_s101
                                 + 0.0104275539609484 w_k1_FDX_s102
                                 + 0.0117455646781874 w_k1_FDX_s103
                                 + 0.0186306631383611 w_k1_FDX_s104
                                 + 0.0162497539694346 w_k1_FDX_s105
                                 + 0.0066889711434201 w_k1_FDX_s106
                                 + 0.0290668815816406 w_k1_FDX_s107
                                 + 0.0922926100172081 w_k1_FDX_s110
                                 + 0.0820985140893143 w_k1_FDX_s111
                                 + 0.0509403992991917 w_k1_FDX_s112
                                 + 0.0430751141205632 w_k1_FDX_s113
                                 + 0.0250140111325447 w_k1_FDX_s114
                                 + 0.0709027085721623 w_k1_FDX_s115
                                 + 0.0535492439377351 w_k1_FDX_s116
                                 + 0.0597346614733103 w_k1_FDX_s117
                                 + 0.0124209782655463 w_k1_IBM_s000
                                 + 0.0449087837253666 w_k1_IBM_s001
                                 + 0.0475226182833774 w_k1_IBM_s002
                                 + 0.0217382387770628 w_k1_IBM_s003
                                 + 0.00995542185306401 w_k1_IBM_s004
                                 + 0.0332344016134244 w_k1_IBM_s005
                                 + 0.0285760981825523 w_k1_IBM_s006
                                 + 0.0274076507369014 w_k1_IBM_s007
                                 + 0.0221243202353986 w_k1_IBM_s010
                                 + 0.0104402057351096 w_k1_IBM_s011
                                 + 0.0239568309997583 w_k1_IBM_s012
                                 + 0.0271379189704669 w_k1_IBM_s013
                                 + 0.00977072311727266 w_k1_IBM_s014
                                 + 0.0122842341731431 w_k1_IBM_s015
                                 + 0.0320921448249445 w_k1_IBM_s016
                                 + 0.0438801667401373 w_k1_IBM_s017
                                 + 0.0175444808332957 w_k1_IBM_s100
                                 + 0.0299010962296459 w_k1_IBM_s101
                                 + 0.0113538332491419 w_k1_IBM_s102
                                 + 0.0112345303767713 w_k1_IBM_s103
                                 + 0.0192930830180447 w_k1_IBM_s104
                                 + 0.0169660598880702 w_k1_IBM_s105
                                 + 0.0067279102906015 w_k1_IBM_s106
                                 + 0.0307194633814251 w_k1_IBM_s107
                                 + 0.0943769146883734 w_k1_IBM_s110
                                 + 0.0816005790243644 w_k1_IBM_s111
                                 + 0.0506614728746669 w_k1_IBM_s112
                                 + 0.0441900006104552 w_k1_IBM_s113
                                 + 0.0266365020301997 w_k1_IBM_s114
                                 + 0.0721663660026326 w_k1_IBM_s115
                                 + 0.0532638419381532 w_k1_IBM_s116
                                 + 0.0598373952136764 w_k1_IBM_s117
                                 + 0.014848533515259 w_k1_INTC_s000
                                 + 0.041357220096415 w_k1_INTC_s001
                                 + 0.0453872678283603 w_k1_INTC_s002
                                 + 0.0211560456541079 w_k1_INTC_s003
                                 + 0.0113040385418795 w_k1_INTC_s004
                                 + 0.0376344274388362 w_k1_INTC_s005
                                 + 0.0264014243666414 w_k1_INTC_s006
                                 + 0.0253636441929496 w_k1_INTC_s007
                                 + 0.0261273557393726 w_k1_INTC_s010
                                 + 0.0109328102086853 w_k1_INTC_s011
                                 + 0.0242510221180491 w_k1_INTC_s012
                                 + 0.0251073748475207 w_k1_INTC_s013
                                 + 0.0107627686087628 w_k1_INTC_s014
                                 + 0.00991336069847805 w_k1_INTC_s015
                                 + 0.0320958589564119 w_k1_INTC_s016
                                 + 0.0413176540730543 w_k1_INTC_s017
                                 + 0.0146834716065023 w_k1_INTC_s100
                                 + 0.0287091059623269 w_k1_INTC_s101
                                 + 0.012773331753961 w_k1_INTC_s102
                                 + 0.0109202778642365 w_k1_INTC_s103
                                 + 0.018821454858198 w_k1_INTC_s104
                                 + 0.0175758543918819 w_k1_INTC_s105
                                 + 0.00622690661504948 w_k1_INTC_s106
                                 + 0.0298323145195871 w_k1_INTC_s107
                                 + 0.0942799500963971 w_k1_INTC_s110
                                 + 0.0718521767727553 w_k1_INTC_s111
                                 + 0.0497373997759058 w_k1_INTC_s112
                                 + 0.0431307775028812 w_k1_INTC_s113
                                 + 0.024013505458961 w_k1_INTC_s114
                                 + 0.0802704960085994 w_k1_INTC_s115
                                 + 0.0521891232228456 w_k1_INTC_s116
                                 + 0.0577144371521048 w_k1_INTC_s117
                                 + 0.0137105667101692 w_k1_KO_s000
                                 + 0.042667460214314 w_k1_KO_s001
                                 + 0.0422435652165187 w_k1_KO_s002
                                 + 0.0205087547142048 w_k1_KO_s003
                                 + 0.0107792184410129 w_k1_KO_s004
                                 + 0.0333928563594939 w_k1_KO_s005
                                 + 0.0278546168003533 w_k1_KO_s006
                                 + 0.02609516014897 w_k1_KO_s007
                                 + 0.0220295350963736 w_k1_KO_s010
                                 + 0.010389116145126 w_k1_KO_s011
                                 + 0.0263025726061231 w_k1_KO_s012
                                 + 0.0262374941033959 w_k1_KO_s013
                                 + 0.0094945459688478 w_k1_KO_s014
                                 + 0.0114438421741894 w_k1_KO_s015
                                 + 0.0335242607911612 w_k1_KO_s016
                                 + 0.0430217970410549 w_k1_KO_s017
                                 + 0.0167603200418218 w_k1_KO_s100
                                 + 0.0275301157966612 w_k1_KO_s101
                                 + 0.0107310243761724 w_k1_KO_s102
                                 + 0.011230368779935 w_k1_KO_s103
                                 + 0.0182700077643833 w_k1_KO_s104
                                 + 0.0180418988784537 w_k1_KO_s105
                                 + 0.00685111118137821 w_k1_KO_s106
                                 + 0.0284708520014282 w_k1_KO_s107
                                 + 0.0890450083100016 w_k1_KO_s110
                                 + 0.0758950108702542 w_k1_KO_s111
                                 + 0.0516888072904883 w_k1_KO_s112
                                 + 0.0451135498915476 w_k1_KO_s113
                                 + 0.0257652168820942 w_k1_KO_s114
                                 + 0.0708574003935686 w_k1_KO_s115
                                 + 0.053717916403479 w_k1_KO_s116
                                 + 0.0590356653032214 w_k1_KO_s117
                                 + 0.0123486249537689 w_k1_MAT_s000
                                 + 0.0420631891420137 w_k1_MAT_s001
                                 + 0.050946723013359 w_k1_MAT_s002
                                 + 0.0237387986620376 w_k1_MAT_s003
                                 + 0.0107849940852122 w_k1_MAT_s004
                                 + 0.0290229027117111 w_k1_MAT_s005
                                 + 0.0333939692598377 w_k1_MAT_s006
                                 + 0.0240064986069729 w_k1_MAT_s007
                                 + 0.0227339878494862 w_k1_MAT_s010
                                 + 0.0118101461095837 w_k1_MAT_s011
                                 + 0.0227838411258654 w_k1_MAT_s012
                                 + 0.0267757821293792 w_k1_MAT_s013
                                 + 0.0145525113529971 w_k1_MAT_s014
                                 + 0.0129150048072614 w_k1_MAT_s015
                                 + 0.0360440379855146 w_k1_MAT_s016
                                 + 0.0420785106968653 w_k1_MAT_s017
                                 + 0.0171492239077176 w_k1_MAT_s100
                                 + 0.0260468713416641 w_k1_MAT_s101
                                 + 0.0107477768446093 w_k1_MAT_s102
                                 + 0.0162803905044006 w_k1_MAT_s103
                                 + 0.0247218860412298 w_k1_MAT_s104
                                 + 0.0161233789932032 w_k1_MAT_s105
                                 + 0.0098119362361 w_k1_MAT_s106
                                 + 0.031640569111172 w_k1_MAT_s107
                                 + 0.0853394317010474 w_k1_MAT_s110
                                 + 0.0958903503794894 w_k1_MAT_s111
                                 + 0.041909213921578 w_k1_MAT_s112
                                 + 0.0403978314471587 w_k1_MAT_s113
                                 + 0.0368479978986831 w_k1_MAT_s114
                                 + 0.0715790035698913 w_k1_MAT_s115
                                 + 0.0522524868950957 w_k1_MAT_s116
                                 + 0.0620480456770359 w_k1_MAT_s117
                                 + 0.0140899604411904 w_k1_MSFT_s000
                                 + 0.0446486832360485 w_k1_MSFT_s001
                                 + 0.0474616835075543 w_k1_MSFT_s002
                                 + 0.0193678483523172 w_k1_MSFT_s003
                                 + 0.00973362256448297 w_k1_MSFT_s004
                                 + 0.034459683658207 w_k1_MSFT_s005
                                 + 0.0285243920117939 w_k1_MSFT_s006
                                 + 0.0284917613371583 w_k1_MSFT_s007
                                 + 0.0219200454115293 w_k1_MSFT_s010
                                 + 0.00896662266600047 w_k1_MSFT_s011
                                 + 0.0247153339534877 w_k1_MSFT_s012
                                 + 0.0298830939710658 w_k1_MSFT_s013
                                 + 0.0111359922472799 w_k1_MSFT_s014
                                 + 0.011296904688023 w_k1_MSFT_s015
                                 + 0.0314870477871655 w_k1_MSFT_s016
                                 + 0.0451344490304984 w_k1_MSFT_s017
                                 + 0.0164948028517307 w_k1_MSFT_s100
                                 + 0.0293343499053495 w_k1_MSFT_s101
                                 + 0.0110483908966588 w_k1_MSFT_s102
                                 + 0.0118546269400999 w_k1_MSFT_s103
                                 + 0.0195830887809926 w_k1_MSFT_s104
                                 + 0.0167043101525992 w_k1_MSFT_s105
                                 + 0.00623533281523408 w_k1_MSFT_s106
                                 + 0.0288419456450878 w_k1_MSFT_s107
                                 + 0.0849665667530926 w_k1_MSFT_s110
                                 + 0.0890583077988009 w_k1_MSFT_s111
                                 + 0.049622590848385 w_k1_MSFT_s112
                                 + 0.0417633613386142 w_k1_MSFT_s113
                                 + 0.0246539534147593 w_k1_MSFT_s114
                                 + 0.0681911050830401 w_k1_MSFT_s115
                                 + 0.0533194028540232 w_k1_MSFT_s116
                                 + 0.056710347953415 w_k1_MSFT_s117
                                 + 0.0130651079034812 w_k1_PG_s000
                                 + 0.0444126135786988 w_k1_PG_s001
                                 + 0.0455828980378109 w_k1_PG_s002
                                 + 0.0218225283131979 w_k1_PG_s003
                                 + 0.0101226121990812 w_k1_PG_s004
                                 + 0.0308766314919649 w_k1_PG_s005
                                 + 0.0307869207148021 w_k1_PG_s006
                                 + 0.0255899379602955 w_k1_PG_s007
                                 + 0.019619853699576 w_k1_PG_s010
                                 + 0.0111044482397355 w_k1_PG_s011
                                 + 0.0233527413370355 w_k1_PG_s012
                                 + 0.0268635362882591 w_k1_PG_s013
                                 + 0.00966709857412035 w_k1_PG_s014
                                 + 0.0114447592044945 w_k1_PG_s015
                                 + 0.0344316312265836 w_k1_PG_s016
                                 + 0.0417604923965955 w_k1_PG_s017
                                 + 0.0170694501452652 w_k1_PG_s100
                                 + 0.0286310314264848 w_k1_PG_s101
                                 + 0.0109267026807887 w_k1_PG_s102
                                 + 0.0114928856590535 w_k1_PG_s103
                                 + 0.0191434276535008 w_k1_PG_s104
                                 + 0.0155507306795538 w_k1_PG_s105
                                 + 0.00721281586286654 w_k1_PG_s106
                                 + 0.0301954015267994 w_k1_PG_s107
                                 + 0.0861189324924004 w_k1_PG_s110
                                 + 0.0781429988787537 w_k1_PG_s111
                                 + 0.0494451220873821 w_k1_PG_s112
                                 + 0.0421792340988784 w_k1_PG_s113
                                 + 0.024652842719718 w_k1_PG_s114
                                 + 0.0682626381212718 w_k1_PG_s115
                                 + 0.0547875337211483 w_k1_PG_s116
                                 + 0.0562930917091453 w_k1_PG_s117 >= 1.07
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
