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
 w_k1_ADM_s0_return_constraint:  1.15027 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.929316 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.867328 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.17614 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.28231 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.931968 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.893269 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.11742 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.45949 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.33836 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.14792 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.1136 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.13682 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.829598 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.00175 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.54525 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.07272 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.756337 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.04076 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.937157 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.828636 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.24502 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.94846 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.879276 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.922778 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.967718 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.11353 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.85763 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.45389 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.961414 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.872525 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.914937 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.89552 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.839295 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.966327 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.40354 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.16832 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.30087 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.30375 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.874467 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.852841 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.12101 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.38489 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.934507 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.775228 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.1299 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.29801 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.57059 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.02321 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.849016 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.958105 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.689912 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.03768 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.77345 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.03458 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.69996 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.1183 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.06112 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.888608 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.24112 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.25238 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.742741 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.09587 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.946658 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.43724 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.797608 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.36562 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.828923 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.781363 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.976438 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.782987 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.697545 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.06987 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.40916 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.11692 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.59525 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.32075 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.87053 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.78061 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.2901 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.57033 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.926078 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.686933 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.26854 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.94512 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.4749 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.2255 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.04628 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.05478 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.703215 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.888664 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.78751 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.981965 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.592956 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.36012 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.951268 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.83289 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.31568 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.14861 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.673583 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.971538 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.955229 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.09013 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.725151 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.58341 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.07848 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.687211 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.873379 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.678041 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.658832 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.15624 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.45882 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.11441 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.57635 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17753 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.909969 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.888931 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.19391 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.2958 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.944359 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.742599 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.01442 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.49024 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.3614 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.19401 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.06657 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.984809 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.802754 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.875331 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.38354 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.07082 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.703687 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.05572 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.891275 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.884481 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.30179 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.0168 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.783358 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.882505 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.03773 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.15619 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.810585 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.39733 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.983338 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.732575 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.0839 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.861737 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.822943 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.928499 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.45511 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.12616 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.3415 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20236 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.91174 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.871115 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.21578 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.36978 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.988735 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.17228 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.59914 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.08299 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.17461 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.834083 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.572648 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.0325 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.716199 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.46212 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.34799 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.691555 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.872196 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.66514 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.15142 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.754717 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.811607 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.80098 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.751235 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.32996 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.600026 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.948861 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.10333 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.13425 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.945359 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.00916 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.560408 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.702324 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.442566 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.80328 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.851608 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.04878 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.49237 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15435 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.971252 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.895446 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.02293 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.21699 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.882322 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.865287 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.08081 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.23972 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.21205 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.946148 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.08871 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.979208 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.781894 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.06913 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.31034 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.12808 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.824398 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.1899 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.13944 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.02003 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.18859 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.932322 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.759476 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.10087 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.908929 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.00171 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.708518 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.09727 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.969417 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.947288 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.16253 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.866833 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.847224 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.05744 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.41511 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.12236 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.27331 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1784 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.955461 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.83654 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.26307 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.22864 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.923151 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.966169 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.34879 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.39373 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.27836 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.04589 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.819796 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.950711 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.758953 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.20146 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.45601 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.835511 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.754265 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.42904 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.00677 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.836929 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.945038 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.43268 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.838402 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.1035 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.784906 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.08056 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.978156 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.33942 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.05884 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.896989 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.749121 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.836999 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.650845 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.34578 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.983394 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.15401 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.28323 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.18988 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.879994 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.00162 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.37674 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.26806 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.05257 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.13899 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.14808 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.01318 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.57997 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.76745 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.565249 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.807953 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.676599 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.31441 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.28633 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.48409 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.858622 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.43899 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.02569 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.763954 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.740717 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.83442 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.909311 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.45036 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.842481 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.21371 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.25015 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.5848 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.92566 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.886974 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.597524 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.681691 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.448471 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.61794 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.929811 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.05974 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.58582 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19868 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.940831 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.846755 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.12166 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.26324 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.945419 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.891442 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.13727 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.3834 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.23904 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.23499 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.14107 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.02357 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.782706 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.906448 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.23073 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.984317 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.821908 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.22632 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.914724 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.953109 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.12547 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.869671 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.772089 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.890015 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.985029 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.04957 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.878027 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.31335 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.07931 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.796346 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.00071 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.841782 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.849741 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.01177 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.32064 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.05696 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.33446 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10884 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.924649 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.896121 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.16739 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.19658 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.00728 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.850604 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.15573 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.3058 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.14262 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.09477 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.956808 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.04461 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.857089 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.10027 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.34945 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.901275 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.767067 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.0955 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.933859 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.91753 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.02512 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.11611 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.852291 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.06423 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.958478 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.12469 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.896806 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.28489 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.921098 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.900698 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.926555 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.872614 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.755217 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.1133 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.16663 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.06098 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.13343 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.165000617374864 z_k1_s0000
                                 - 0.311001065439496 z_k1_s0001
                                 - 0.199000721718682 z_k1_s0002
                                 - 0.326001666475791 z_k1_s0003
                                 - 0.163000611236992 z_k1_s0004
                                 - 0.259001793858446 z_k1_s0005
                                 - 0.657002793294004 z_k1_s0006
                                 - 0.14000065165192 z_k1_s0007
                                 - 0.665001467466787 z_k1_s0010
                                 - 0.424999335506099 z_k1_s0011
                                 - 0.522999683722483 z_k1_s0012
                                 - 0.546999896918552 z_k1_s0013
                                 - 0.188000207500029 z_k1_s0014
                                 - 0.358000568503545 z_k1_s0015
                                 - 0.258000411454513 z_k1_s0016
                                 - 0.170000361003515 z_k1_s0017
                                 - 0.0990000640821048 z_k1_s0020
                                 - 0.110000143980182 z_k1_s0021
                                 - 0.118000059178655 z_k1_s0022
                                 - 0.101000042881723 z_k1_s0023
                                 - 0.247000722255858 z_k1_s0024
                                 - 0.181000242867396 z_k1_s0025
                                 - 0.326000343341191 z_k1_s0026
                                 - 0.128000084176864 z_k1_s0027
                                 - 0.197999525159018 z_k1_s0030
                                 - 0.161000562437351 z_k1_s0031
                                 - 0.205000597917128 z_k1_s0032
                                 - 0.152999517713791 z_k1_s0033
                                 - 0.154999461644692 z_k1_s0034
                                 - 0.110999933764897 z_k1_s0035
                                 - 0.0999999883449507 z_k1_s0036
                                 - 0.185999861573612 z_k1_s0037
                                 - 0.4070002765461 z_k1_s0040
                                 - 0.601001111789318 z_k1_s0041
                                 - 0.706999817860072 z_k1_s0042
                                 - 0.187999978876799 z_k1_s0043
                                 - 0.247000188278595 z_k1_s0044
                                 - 0.338000082433838 z_k1_s0045
                                 - 0.355001288171828 z_k1_s0046
                                 - 0.486000788268881 z_k1_s0047
                                 - 0.367999494150166 z_k1_s0050
                                 - 0.21999995757894 z_k1_s0051
                                 - 0.081999971397603 z_k1_s0052
                                 - 0.131999974547364 z_k1_s0053
                                 - 0.260999943277741 z_k1_s0054
                                 - 0.0750000047246412 z_k1_s0055
                                 - 0.166000697073301 z_k1_s0056
                                 - 0.103000012116508 z_k1_s0057
                                 - 0.500001350099469 z_k1_s0060
                                 - 0.159000029363652 z_k1_s0061
                                 - 0.793998502870146 z_k1_s0062
                                 - 0.339000101639482 z_k1_s0063
                                 - 0.370001412833586 z_k1_s0064
                                 - 0.576998634110939 z_k1_s0065
                                 - 0.220999893352002 z_k1_s0066
                                 - 0.487998862721209 z_k1_s0067
                                 - 0.236000159102697 z_k1_s0070
                                 - 0.560001456889063 z_k1_s0071
                                 - 0.774999641531093 z_k1_s0072
                                 - 0.375001276613577 z_k1_s0073
                                 - 0.306000247563716 z_k1_s0074
                                 - 0.23500035049063 z_k1_s0075
                                 - 0.236000159102697 z_k1_s0076
                                 - 1.02300012618177 z_k1_s0077
                                 - 0.104000145926445 z_k1_s0100
                                 - 0.164000160145467 z_k1_s0101
                                 - 0.385000015438637 z_k1_s0102
                                 - 0.237001102206347 z_k1_s0103
                                 - 0.232000570420815 z_k1_s0104
                                 - 0.503000285962994 z_k1_s0105
                                 - 0.367001284614403 z_k1_s0106
                                 - 0.282001340270877 z_k1_s0107
                                 - 0.213999610305433 z_k1_s0110
                                 - 0.16899997521316 z_k1_s0111
                                 - 0.104999920315279 z_k1_s0112
                                 - 0.277999843303308 z_k1_s0113
                                 - 0.215999673242864 z_k1_s0114
                                 - 0.170000006681876 z_k1_s0115
                                 - 0.481999138921581 z_k1_s0116
                                 - 0.146999995301387 z_k1_s0117
                                 - 0.225999557231099 z_k1_s0120
                                 - 0.264000297764095 z_k1_s0121
                                 - 0.475998010170018 z_k1_s0122
                                 - 0.127999835131621 z_k1_s0123
                                 - 0.352999313925113 z_k1_s0124
                                 - 0.16399967209072 z_k1_s0125
                                 - 0.454998354176771 z_k1_s0126
                                 - 0.422999826566344 z_k1_s0127
                                 - 0.244999419072702 z_k1_s0130
                                 - 0.19699966764186 z_k1_s0131
                                 - 0.219999631073138 z_k1_s0132
                                 - 0.194999622974019 z_k1_s0133
                                 - 0.477999010337471 z_k1_s0134
                                 - 0.304998558112363 z_k1_s0135
                                 - 0.397000172633928 z_k1_s0136
                                 - 0.163999700970933 z_k1_s0137
                                 - 0.212999481352852 z_k1_s0140
                                 - 0.0779999256812662 z_k1_s0141
                                 - 0.284000374469916 z_k1_s0142
                                 - 0.0690000121564642 z_k1_s0143
                                 - 0.0929999236891956 z_k1_s0144
                                 - 0.0799999538423087 z_k1_s0145
                                 - 0.0929999236891956 z_k1_s0146
                                 - 0.155999424962754 z_k1_s0147
                                 - 0.232000029273745 z_k1_s0150
                                 - 0.192999862046295 z_k1_s0151
                                 - 0.257000091563145 z_k1_s0152
                                 - 0.713000632245925 z_k1_s0153
                                 - 0.301999678264173 z_k1_s0154
                                 - 0.325999691358007 z_k1_s0155
                                 - 0.426999989711173 z_k1_s0156
                                 - 0.468998553125681 z_k1_s0157
                                 - 0.381999421778037 z_k1_s0160
                                 - 0.524998923088342 z_k1_s0161
                                 - 0.175999484958583 z_k1_s0162
                                 - 0.158999571169876 z_k1_s0163
                                 - 0.305998448196724 z_k1_s0164
                                 - 0.288999817904717 z_k1_s0165
                                 - 0.591997450817334 z_k1_s0166
                                 - 0.137999768265947 z_k1_s0167
                                 - 0.149999724420875 z_k1_s0170
                                 - 0.309999278850027 z_k1_s0171
                                 - 0.302999498924088 z_k1_s0172
                                 - 0.487998375674295 z_k1_s0173
                                 - 0.35999842889141 z_k1_s0174
                                 - 0.112999949070834 z_k1_s0175
                                 - 0.4529994760446 z_k1_s0176
                                 - 0.349999104282406 z_k1_s0177
                                 - 0.212000560994208 z_k1_s1000
                                 - 0.085000097551226 z_k1_s1001
                                 - 0.118000170722941 z_k1_s1002
                                 - 0.198000208229881 z_k1_s1003
                                 - 0.152000532864003 z_k1_s1004
                                 - 0.0670000891121644 z_k1_s1005
                                 - 0.0680001472811032 z_k1_s1006
                                 - 0.254000465285147 z_k1_s1007
                                 - 0.120000280019811 z_k1_s1010
                                 - 0.109000262191348 z_k1_s1011
                                 - 0.256000158334501 z_k1_s1012
                                 - 0.132000370741948 z_k1_s1013
                                 - 0.256000158334501 z_k1_s1014
                                 - 0.296001506077566 z_k1_s1015
                                 - 0.2040004446736 z_k1_s1016
                                 - 0.195000102231313 z_k1_s1017
                                 - 0.287001153516959 z_k1_s1020
                                 - 0.161000595094671 z_k1_s1021
                                 - 0.372001055149134 z_k1_s1022
                                 - 0.11500054221095 z_k1_s1023
                                 - 0.235001464519461 z_k1_s1024
                                 - 0.123000494364411 z_k1_s1025
                                 - 0.138000486652481 z_k1_s1026
                                 - 0.209000800017416 z_k1_s1027
                                 - 0.311000578632366 z_k1_s1030
                                 - 0.684999605118835 z_k1_s1031
                                 - 0.393999884694675 z_k1_s1032
                                 - 0.240000085294282 z_k1_s1033
                                 - 0.283000228241227 z_k1_s1034
                                 - 0.142000135575819 z_k1_s1035
                                 - 0.245000016146217 z_k1_s1036
                                 - 0.536998928380438 z_k1_s1037
                                 - 0.182999969913075 z_k1_s1040
                                 - 0.125999995620471 z_k1_s1041
                                 - 0.145999941145966 z_k1_s1042
                                 - 0.120999959414119 z_k1_s1043
                                 - 0.255000252124647 z_k1_s1044
                                 - 0.390000033896674 z_k1_s1045
                                 - 0.342000642955276 z_k1_s1046
                                 - 0.429999526347839 z_k1_s1047
                                 - 0.0800001408838967 z_k1_s1050
                                 - 0.142000602239408 z_k1_s1051
                                 - 0.0760001392997095 z_k1_s1052
                                 - 0.107000096977084 z_k1_s1053
                                 - 0.0650000803431184 z_k1_s1054
                                 - 0.206999808981307 z_k1_s1055
                                 - 0.166000611744532 z_k1_s1056
                                 - 0.249000371616035 z_k1_s1057
                                 - 0.337001749050368 z_k1_s1060
                                 - 0.134000533611704 z_k1_s1061
                                 - 0.158000599846526 z_k1_s1062
                                 - 0.209001395797985 z_k1_s1063
                                 - 0.290000800337429 z_k1_s1064
                                 - 0.0820003901029235 z_k1_s1065
                                 - 0.232000640269944 z_k1_s1066
                                 - 0.196000540917711 z_k1_s1067
                                 - 0.0960001580206606 z_k1_s1070
                                 - 0.0540000273865357 z_k1_s1071
                                 - 0.249000563616303 z_k1_s1072
                                 - 0.0770000754956667 z_k1_s1073
                                 - 0.218000584338898 z_k1_s1074
                                 - 0.0650000803431184 z_k1_s1075
                                 - 0.15800011657547 z_k1_s1076
                                 - 0.0670000631351869 z_k1_s1077
                                 - 2.09799645118729 z_k1_s1100
                                 - 1.9710009534508 z_k1_s1101
                                 - 1.14600177375291 z_k1_s1102
                                 - 0.475999366913276 z_k1_s1103
                                 - 1.93500047297771 z_k1_s1104
                                 - 1.12999930665811 z_k1_s1105
                                 - 0.542998796318806 z_k1_s1106
                                 - 0.841998505167486 z_k1_s1107
                                 - 0.384999930378261 z_k1_s1110
                                 - 0.489000541165724 z_k1_s1111
                                 - 0.369998824649997 z_k1_s1112
                                 - 0.292999862773644 z_k1_s1113
                                 - 0.331999665069234 z_k1_s1114
                                 - 0.219999716359173 z_k1_s1115
                                 - 1.00399867453007 z_k1_s1116
                                 - 0.320999764601217 z_k1_s1117
                                 - 0.343000900222369 z_k1_s1120
                                 - 0.632002680541035 z_k1_s1121
                                 - 0.862999187062922 z_k1_s1122
                                 - 1.03000469709121 z_k1_s1123
                                 - 0.607001007426274 z_k1_s1124
                                 - 1.08700431085722 z_k1_s1125
                                 - 0.376000837297718 z_k1_s1126
                                 - 1.16800054830832 z_k1_s1127
                                 - 0.412999194438295 z_k1_s1130
                                 - 0.227999819147287 z_k1_s1131
                                 - 0.213999752217263 z_k1_s1132
                                 - 0.304001056195172 z_k1_s1133
                                 - 0.639999882518336 z_k1_s1134
                                 - 0.186999861423424 z_k1_s1135
                                 - 0.286998592068082 z_k1_s1136
                                 - 0.506999246683111 z_k1_s1137
                                 - 0.312999891786294 z_k1_s1140
                                 - 0.402999869768291 z_k1_s1141
                                 - 0.76399847517232 z_k1_s1142
                                 - 0.442000172106993 z_k1_s1143
                                 - 0.976998843422825 z_k1_s1144
                                 - 0.539001581215727 z_k1_s1145
                                 - 0.20300000909714 z_k1_s1146
                                 - 0.427001463975984 z_k1_s1147
                                 - 1.3479988634319 z_k1_s1150
                                 - 0.32599978086436 z_k1_s1151
                                 - 0.544999742100173 z_k1_s1152
                                 - 0.774000347293421 z_k1_s1153
                                 - 0.486999984424804 z_k1_s1154
                                 - 0.486999984424804 z_k1_s1155
                                 - 1.17399848560642 z_k1_s1156
                                 - 0.382999695103861 z_k1_s1157
                                 - 1.21500160639681 z_k1_s1160
                                 - 0.753001197893832 z_k1_s1161
                                 - 3.11899877154579 z_k1_s1162
                                 - 1.21400218993607 z_k1_s1163
                                 - 0.790001099768364 z_k1_s1164
                                 - 3.00500618974675 z_k1_s1165
                                 - 1.35400780904996 z_k1_s1166
                                 - 1.98300882752039 z_k1_s1167
                                 - 0.224999807491185 z_k1_s1170
                                 - 0.550998703585217 z_k1_s1171
                                 - 0.590999825387279 z_k1_s1172
                                 - 0.819999350958948 z_k1_s1173
                                 - 0.359999849625785 z_k1_s1174
                                 - 0.632999032780833 z_k1_s1175
                                 - 0.201999753604582 z_k1_s1176
                                 - 0.558998927945629 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.889818 w_k1_ADM_s000
                                 + 1.0833 w_k1_AXP_s000
                                 + 0.831143 w_k1_DIS_s000
                                 + 0.986162 w_k1_FDX_s000
                                 + 0.799503 w_k1_IBM_s000
                                 + 1.02101 w_k1_INTC_s000
                                 + 0.79129 w_k1_KO_s000
                                 + 0.643309 w_k1_MAT_s000
                                 + 0.810761 w_k1_MSFT_s000
                                 + 0.993213 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.651545 w_k1_ADM_s000
                                 + 0.750192 w_k1_AXP_s000
                                 + 0.832359 w_k1_DIS_s000
                                 + 0.792799 w_k1_FDX_s000
                                 + 0.902253 w_k1_IBM_s000
                                 + 0.955507 w_k1_INTC_s000
                                 + 0.933528 w_k1_KO_s000
                                 + 0.777155 w_k1_MAT_s000
                                 + 0.724802 w_k1_MSFT_s000
                                 + 0.889713 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.03048 w_k1_ADM_s000
                                 + 1.21388 w_k1_AXP_s000
                                 + 1.56025 w_k1_DIS_s000
                                 + 1.24937 w_k1_FDX_s000 + 1.4697 w_k1_IBM_s000
                                 + 1.07246 w_k1_INTC_s000
                                 + 1.37502 w_k1_KO_s000 + 1.6413 w_k1_MAT_s000
                                 + 1.27234 w_k1_MSFT_s000
                                 + 1.41116 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.29431 w_k1_ADM_s000
                                 + 1.33893 w_k1_AXP_s000
                                 + 1.44905 w_k1_DIS_s000
                                 + 1.33123 w_k1_FDX_s000
                                 + 1.04256 w_k1_IBM_s000
                                 + 1.18517 w_k1_INTC_s000 + 1.3543 w_k1_KO_s000
                                 + 1.16667 w_k1_MAT_s000
                                 + 1.18169 w_k1_MSFT_s000
                                 + 1.27989 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.1235 w_k1_ADM_s000
                                 + 1.00519 w_k1_AXP_s000
                                 + 1.17508 w_k1_DIS_s000
                                 + 0.842631 w_k1_FDX_s000
                                 + 1.55944 w_k1_IBM_s000
                                 + 0.724633 w_k1_INTC_s000
                                 + 1.15494 w_k1_KO_s000
                                 + 0.996835 w_k1_MAT_s000
                                 + 0.807539 w_k1_MSFT_s000
                                 + 1.06091 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.934666 w_k1_ADM_s000
                                 + 0.933042 w_k1_AXP_s000
                                 + 0.915957 w_k1_DIS_s000
                                 + 1.06914 w_k1_FDX_s000
                                 + 0.714149 w_k1_IBM_s000
                                 + 1.05787 w_k1_INTC_s000
                                 + 0.936538 w_k1_KO_s000
                                 + 0.719328 w_k1_MAT_s000
                                 + 0.901807 w_k1_MSFT_s000
                                 + 1.16126 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.818862 w_k1_ADM_s000
                                 + 1.07197 w_k1_AXP_s000
                                 + 1.17766 w_k1_DIS_s000
                                 + 0.980386 w_k1_FDX_s000
                                 + 1.10163 w_k1_IBM_s000
                                 + 0.974865 w_k1_INTC_s000
                                 + 1.2024 w_k1_KO_s000 + 0.960737 w_k1_MAT_s000
                                 + 0.941463 w_k1_MSFT_s000
                                 + 1.08969 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.722543 w_k1_ADM_s000
                                 + 0.596529 w_k1_AXP_s000
                                 + 0.752691 w_k1_DIS_s000
                                 + 0.775716 w_k1_FDX_s000
                                 + 0.69427 w_k1_IBM_s000
                                 + 0.716508 w_k1_INTC_s000
                                 + 0.681663 w_k1_KO_s000
                                 + 0.598214 w_k1_MAT_s000
                                 + 0.722397 w_k1_MSFT_s000
                                 + 0.826852 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.23378 w_k1_ADM_s001
                                 + 1.23751 w_k1_AXP_s001
                                 + 1.20795 w_k1_DIS_s001
                                 + 1.23963 w_k1_FDX_s001
                                 + 1.11002 w_k1_IBM_s001
                                 + 1.06746 w_k1_INTC_s001
                                 + 1.16203 w_k1_KO_s001 + 1.3447 w_k1_MAT_s001
                                 + 1.02643 w_k1_MSFT_s001
                                 + 1.08676 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.959793 w_k1_ADM_s001
                                 + 1.01308 w_k1_AXP_s001
                                 + 0.771995 w_k1_DIS_s001
                                 + 0.882208 w_k1_FDX_s001
                                 + 0.940211 w_k1_IBM_s001
                                 + 0.892408 w_k1_INTC_s001
                                 + 0.954954 w_k1_KO_s001
                                 + 1.15753 w_k1_MAT_s001
                                 + 0.822376 w_k1_MSFT_s001
                                 + 0.960768 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.774435 w_k1_ADM_s001
                                 + 0.85107 w_k1_AXP_s001
                                 + 0.762584 w_k1_DIS_s001
                                 + 0.953221 w_k1_FDX_s001
                                 + 0.67606 w_k1_IBM_s001
                                 + 0.802847 w_k1_INTC_s001
                                 + 0.761213 w_k1_KO_s001
                                 + 0.895543 w_k1_MAT_s001
                                 + 0.869826 w_k1_MSFT_s001
                                 + 0.86903 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.812603 w_k1_ADM_s001
                                 + 0.763031 w_k1_AXP_s001
                                 + 0.8158 w_k1_DIS_s001
                                 + 0.818929 w_k1_FDX_s001 + 1.17 w_k1_IBM_s001
                                 + 0.861148 w_k1_INTC_s001
                                 + 0.959041 w_k1_KO_s001
                                 + 1.13794 w_k1_MAT_s001
                                 + 0.848834 w_k1_MSFT_s001
                                 + 0.990253 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.1524 w_k1_ADM_s001
                                 + 1.58438 w_k1_AXP_s001
                                 + 1.65205 w_k1_DIS_s001
                                 + 1.34374 w_k1_FDX_s001
                                 + 1.56369 w_k1_IBM_s001
                                 + 1.1014 w_k1_INTC_s001 + 1.44173 w_k1_KO_s001
                                 + 2.27373 w_k1_MAT_s001
                                 + 1.18988 w_k1_MSFT_s001
                                 + 1.19888 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.694275 w_k1_ADM_s001
                                 + 0.58466 w_k1_AXP_s001
                                 + 0.653092 w_k1_DIS_s001
                                 + 0.731302 w_k1_FDX_s001
                                 + 0.852415 w_k1_IBM_s001
                                 + 0.900213 w_k1_INTC_s001
                                 + 0.805701 w_k1_KO_s001
                                 + 0.96464 w_k1_MAT_s001
                                 + 0.809381 w_k1_MSFT_s001
                                 + 0.748728 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.933787 w_k1_ADM_s001
                                 + 1.00399 w_k1_AXP_s001
                                 + 0.871512 w_k1_DIS_s001
                                 + 0.854292 w_k1_FDX_s001
                                 + 1.61667 w_k1_IBM_s001
                                 + 0.874221 w_k1_INTC_s001
                                 + 1.22261 w_k1_KO_s001 + 2.12299 w_k1_MAT_s001
                                 + 0.862161 w_k1_MSFT_s001
                                 + 1.0486 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.764488 w_k1_ADM_s001
                                 + 0.899752 w_k1_AXP_s001
                                 + 1.24175 w_k1_DIS_s001
                                 + 1.04779 w_k1_FDX_s001
                                 + 0.852516 w_k1_IBM_s001
                                 + 1.05736 w_k1_INTC_s001
                                 + 1.05458 w_k1_KO_s001
                                 + 0.895996 w_k1_MAT_s001
                                 + 1.0009 w_k1_MSFT_s001 + 1.02485 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.818826 w_k1_ADM_s002
                                 + 0.837298 w_k1_AXP_s002
                                 + 0.916239 w_k1_DIS_s002
                                 + 0.94546 w_k1_FDX_s002
                                 + 0.854888 w_k1_IBM_s002
                                 + 1.12292 w_k1_INTC_s002
                                 + 1.03851 w_k1_KO_s002
                                 + 0.705434 w_k1_MAT_s002
                                 + 1.11367 w_k1_MSFT_s002
                                 + 0.834414 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.799985 w_k1_ADM_s002
                                 + 0.692758 w_k1_AXP_s002
                                 + 0.963173 w_k1_DIS_s002
                                 + 0.866641 w_k1_FDX_s002
                                 + 1.23748 w_k1_IBM_s002
                                 + 0.782357 w_k1_INTC_s002
                                 + 1.06917 w_k1_KO_s002 + 1.19074 w_k1_MAT_s002
                                 + 1.01793 w_k1_MSFT_s002
                                 + 0.820612 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.07134 w_k1_ADM_s002
                                 + 1.51625 w_k1_AXP_s002
                                 + 0.980566 w_k1_DIS_s002
                                 + 1.13992 w_k1_FDX_s002
                                 + 0.915805 w_k1_IBM_s002
                                 + 1.04394 w_k1_INTC_s002
                                 + 0.779136 w_k1_KO_s002
                                 + 1.19586 w_k1_MAT_s002
                                 + 0.999048 w_k1_MSFT_s002
                                 + 1.01685 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.00238 w_k1_ADM_s002
                                 + 1.05474 w_k1_AXP_s002
                                 + 0.929416 w_k1_DIS_s002
                                 + 1.03687 w_k1_FDX_s002
                                 + 1.10532 w_k1_IBM_s002
                                 + 1.08298 w_k1_INTC_s002
                                 + 0.963016 w_k1_KO_s002
                                 + 1.04636 w_k1_MAT_s002
                                 + 1.07248 w_k1_MSFT_s002
                                 + 0.886655 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.10096 w_k1_ADM_s002
                                 + 1.2396 w_k1_AXP_s002 + 1.24279 w_k1_DIS_s002
                                 + 1.307 w_k1_FDX_s002 + 0.661528 w_k1_IBM_s002
                                 + 1.13814 w_k1_INTC_s002
                                 + 1.06022 w_k1_KO_s002 + 1.26314 w_k1_MAT_s002
                                 + 1.21421 w_k1_MSFT_s002
                                 + 1.09163 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.862763 w_k1_ADM_s002
                                 + 0.816127 w_k1_AXP_s002
                                 + 0.619585 w_k1_DIS_s002
                                 + 0.814415 w_k1_FDX_s002
                                 + 0.722253 w_k1_IBM_s002
                                 + 1.14909 w_k1_INTC_s002
                                 + 0.689585 w_k1_KO_s002
                                 + 0.560898 w_k1_MAT_s002
                                 + 0.862964 w_k1_MSFT_s002
                                 + 0.803767 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.988718 w_k1_ADM_s002
                                 + 1.43527 w_k1_AXP_s002
                                 + 1.18072 w_k1_DIS_s002
                                 + 1.09264 w_k1_FDX_s002 + 1.6009 w_k1_IBM_s002
                                 + 1.02903 w_k1_INTC_s002
                                 + 1.18557 w_k1_KO_s002 + 1.58661 w_k1_MAT_s002
                                 + 1.03594 w_k1_MSFT_s002
                                 + 1.12096 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.685125 w_k1_ADM_s002
                                 + 0.888253 w_k1_AXP_s002
                                 + 0.693548 w_k1_DIS_s002
                                 + 0.781105 w_k1_FDX_s002
                                 + 1.17576 w_k1_IBM_s002
                                 + 0.929246 w_k1_INTC_s002
                                 + 0.896038 w_k1_KO_s002 + 0.8812 w_k1_MAT_s002
                                 + 0.885872 w_k1_MSFT_s002
                                 + 0.877234 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.07287 w_k1_ADM_s003
                                 + 1.28404 w_k1_AXP_s003
                                 + 1.25259 w_k1_DIS_s003
                                 + 1.30312 w_k1_FDX_s003
                                 + 0.833579 w_k1_IBM_s003
                                 + 1.16768 w_k1_INTC_s003
                                 + 0.953182 w_k1_KO_s003
                                 + 0.845112 w_k1_MAT_s003
                                 + 1.06906 w_k1_MSFT_s003
                                 + 1.24169 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.32119 w_k1_ADM_s003
                                 + 1.0991 w_k1_AXP_s003 + 1.32684 w_k1_DIS_s003
                                 + 1.08787 w_k1_FDX_s003
                                 + 1.23065 w_k1_IBM_s003
                                 + 0.994251 w_k1_INTC_s003
                                 + 1.31257 w_k1_KO_s003 + 1.08695 w_k1_MAT_s003
                                 + 1.13669 w_k1_MSFT_s003
                                 + 1.28519 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.908709 w_k1_ADM_s003
                                 + 1.10294 w_k1_AXP_s003
                                 + 0.916785 w_k1_DIS_s003
                                 + 0.952795 w_k1_FDX_s003
                                 + 1.13347 w_k1_IBM_s003
                                 + 1.12285 w_k1_INTC_s003
                                 + 0.984231 w_k1_KO_s003
                                 + 0.954307 w_k1_MAT_s003
                                 + 1.02286 w_k1_MSFT_s003
                                 + 1.14215 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.99971 w_k1_ADM_s003
                                 + 0.773989 w_k1_AXP_s003
                                 + 0.99304 w_k1_DIS_s003
                                 + 0.988198 w_k1_FDX_s003
                                 + 0.698051 w_k1_IBM_s003
                                 + 1.03355 w_k1_INTC_s003
                                 + 0.896316 w_k1_KO_s003
                                 + 0.579808 w_k1_MAT_s003
                                 + 1.02074 w_k1_MSFT_s003
                                 + 0.944801 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.874986 w_k1_ADM_s003
                                 + 0.596968 w_k1_AXP_s003
                                 + 0.669906 w_k1_DIS_s003
                                 + 0.779209 w_k1_FDX_s003
                                 + 0.673989 w_k1_IBM_s003
                                 + 0.736286 w_k1_INTC_s003
                                 + 0.687837 w_k1_KO_s003
                                 + 0.471917 w_k1_MAT_s003
                                 + 0.88491 w_k1_MSFT_s003
                                 + 0.941549 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.06346 w_k1_ADM_s003
                                 + 1.04161 w_k1_AXP_s003
                                 + 1.15562 w_k1_DIS_s003
                                 + 0.934932 w_k1_FDX_s003
                                 + 1.56623 w_k1_IBM_s003
                                 + 1.00826 w_k1_INTC_s003
                                 + 1.20071 w_k1_KO_s003 + 1.12753 w_k1_MAT_s003
                                 + 1.00479 w_k1_MSFT_s003
                                 + 1.21036 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.35872 w_k1_ADM_s003
                                 + 1.52312 w_k1_AXP_s003
                                 + 1.51939 w_k1_DIS_s003 + 1.2989 w_k1_FDX_s003
                                 + 1.54207 w_k1_IBM_s003
                                 + 0.940598 w_k1_INTC_s003
                                 + 1.36565 w_k1_KO_s003 + 1.39375 w_k1_MAT_s003
                                 + 1.0186 w_k1_MSFT_s003 + 1.36448 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.857274 w_k1_ADM_s003
                                 + 0.788017 w_k1_AXP_s003
                                 + 0.913851 w_k1_DIS_s003
                                 + 0.780216 w_k1_FDX_s003
                                 + 1.12627 w_k1_IBM_s003
                                 + 0.815948 w_k1_INTC_s003
                                 + 1.11864 w_k1_KO_s003 + 0.73373 w_k1_MAT_s003
                                 + 0.776895 w_k1_MSFT_s003
                                 + 1.07101 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.02267 w_k1_ADM_s004
                                 + 1.06922 w_k1_AXP_s004
                                 + 1.20307 w_k1_DIS_s004
                                 + 1.01766 w_k1_FDX_s004
                                 + 1.09721 w_k1_IBM_s004
                                 + 1.37773 w_k1_INTC_s004
                                 + 1.18147 w_k1_KO_s004
                                 + 0.963982 w_k1_MAT_s004
                                 + 1.11258 w_k1_MSFT_s004
                                 + 1.03337 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.865408 w_k1_ADM_s004
                                 + 0.761188 w_k1_AXP_s004
                                 + 0.693469 w_k1_DIS_s004
                                 + 0.821216 w_k1_FDX_s004
                                 + 0.709111 w_k1_IBM_s004
                                 + 0.971653 w_k1_INTC_s004
                                 + 0.716583 w_k1_KO_s004
                                 + 0.68522 w_k1_MAT_s004
                                 + 0.813175 w_k1_MSFT_s004
                                 + 0.839801 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.864344 w_k1_ADM_s004
                                 + 0.991763 w_k1_AXP_s004
                                 + 0.900331 w_k1_DIS_s004
                                 + 0.846622 w_k1_FDX_s004
                                 + 1.13765 w_k1_IBM_s004
                                 + 1.05176 w_k1_INTC_s004
                                 + 1.04396 w_k1_KO_s004 + 1.16599 w_k1_MAT_s004
                                 + 0.864389 w_k1_MSFT_s004
                                 + 0.921327 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.39543 w_k1_ADM_s004
                                 + 1.39357 w_k1_AXP_s004
                                 + 1.86211 w_k1_DIS_s004
                                 + 1.28066 w_k1_FDX_s004
                                 + 1.48061 w_k1_IBM_s004
                                 + 1.27258 w_k1_INTC_s004
                                 + 1.57168 w_k1_KO_s004 + 1.7164 w_k1_MAT_s004
                                 + 1.29855 w_k1_MSFT_s004 + 1.1922 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.24771 w_k1_ADM_s004
                                 + 1.30025 w_k1_AXP_s004
                                 + 1.19028 w_k1_DIS_s004
                                 + 1.22887 w_k1_FDX_s004
                                 + 1.40821 w_k1_IBM_s004
                                 + 1.22401 w_k1_INTC_s004
                                 + 1.26938 w_k1_KO_s004 + 1.39823 w_k1_MAT_s004
                                 + 1.09259 w_k1_MSFT_s004
                                 + 1.12511 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.20941 w_k1_ADM_s004
                                 + 0.890557 w_k1_AXP_s004
                                 + 1.24328 w_k1_DIS_s004
                                 + 1.07868 w_k1_FDX_s004
                                 + 0.795036 w_k1_IBM_s004
                                 + 1.42008 w_k1_INTC_s004
                                 + 1.19237 w_k1_KO_s004
                                 + 0.749641 w_k1_MAT_s004
                                 + 1.06559 w_k1_MSFT_s004
                                 + 1.13852 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.37613 w_k1_ADM_s004
                                 + 1.56536 w_k1_AXP_s004
                                 + 1.28612 w_k1_DIS_s004
                                 + 1.24711 w_k1_FDX_s004
                                 + 0.859275 w_k1_IBM_s004
                                 + 1.32665 w_k1_INTC_s004
                                 + 1.07205 w_k1_KO_s004 + 1.16346 w_k1_MAT_s004
                                 + 1.11936 w_k1_MSFT_s004
                                 + 1.15819 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.960469 w_k1_ADM_s004
                                 + 0.831498 w_k1_AXP_s004
                                 + 0.907004 w_k1_DIS_s004
                                 + 1.00122 w_k1_FDX_s004
                                 + 0.736835 w_k1_IBM_s004
                                 + 1.22259 w_k1_INTC_s004
                                 + 0.910071 w_k1_KO_s004
                                 + 0.74307 w_k1_MAT_s004
                                 + 1.05766 w_k1_MSFT_s004
                                 + 0.994185 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.852838 w_k1_ADM_s005
                                 + 0.953302 w_k1_AXP_s005
                                 + 1.03369 w_k1_DIS_s005
                                 + 1.08883 w_k1_FDX_s005
                                 + 0.913195 w_k1_IBM_s005
                                 + 1.02375 w_k1_INTC_s005
                                 + 0.864778 w_k1_KO_s005
                                 + 0.549388 w_k1_MAT_s005
                                 + 0.962644 w_k1_MSFT_s005
                                 + 0.981664 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.04769 w_k1_ADM_s005
                                 + 1.16263 w_k1_AXP_s005
                                 + 1.04364 w_k1_DIS_s005
                                 + 1.11215 w_k1_FDX_s005
                                 + 1.11707 w_k1_IBM_s005
                                 + 1.11559 w_k1_INTC_s005 + 1.1645 w_k1_KO_s005
                                 + 1.07088 w_k1_MAT_s005
                                 + 1.02303 w_k1_MSFT_s005
                                 + 1.07456 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.25339 w_k1_ADM_s005
                                 + 1.0706 w_k1_AXP_s005 + 1.36946 w_k1_DIS_s005
                                 + 1.35258 w_k1_FDX_s005
                                 + 0.94139 w_k1_IBM_s005
                                 + 1.06286 w_k1_INTC_s005
                                 + 1.15914 w_k1_KO_s005 + 0.58637 w_k1_MAT_s005
                                 + 1.06468 w_k1_MSFT_s005
                                 + 1.17489 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.890619 w_k1_ADM_s005
                                 + 0.894564 w_k1_AXP_s005
                                 + 0.802856 w_k1_DIS_s005
                                 + 0.86405 w_k1_FDX_s005
                                 + 1.40178 w_k1_IBM_s005
                                 + 0.884082 w_k1_INTC_s005
                                 + 0.98091 w_k1_KO_s005 + 1.31551 w_k1_MAT_s005
                                 + 0.793757 w_k1_MSFT_s005
                                 + 1.09571 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.673838 w_k1_ADM_s005
                                 + 0.793708 w_k1_AXP_s005
                                 + 0.764265 w_k1_DIS_s005
                                 + 0.830549 w_k1_FDX_s005
                                 + 0.916517 w_k1_IBM_s005
                                 + 0.845331 w_k1_INTC_s005
                                 + 0.858471 w_k1_KO_s005
                                 + 0.68639 w_k1_MAT_s005
                                 + 0.751167 w_k1_MSFT_s005
                                 + 0.862603 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.845785 w_k1_ADM_s005
                                 + 1.21947 w_k1_AXP_s005
                                 + 1.40658 w_k1_DIS_s005 + 1.0834 w_k1_FDX_s005
                                 + 1.65867 w_k1_IBM_s005
                                 + 0.937654 w_k1_INTC_s005
                                 + 1.33525 w_k1_KO_s005 + 1.57835 w_k1_MAT_s005
                                 + 0.988247 w_k1_MSFT_s005
                                 + 1.0825 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.745705 w_k1_ADM_s005
                                 + 0.845478 w_k1_AXP_s005
                                 + 0.539494 w_k1_DIS_s005
                                 + 0.846368 w_k1_FDX_s005
                                 + 0.660747 w_k1_IBM_s005
                                 + 0.792619 w_k1_INTC_s005
                                 + 0.639206 w_k1_KO_s005
                                 + 0.551529 w_k1_MAT_s005
                                 + 0.730674 w_k1_MSFT_s005
                                 + 0.794157 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.1398 w_k1_ADM_s005
                                 + 1.56394 w_k1_AXP_s005
                                 + 1.27386 w_k1_DIS_s005
                                 + 1.31715 w_k1_FDX_s005 + 1.4366 w_k1_IBM_s005
                                 + 1.22259 w_k1_INTC_s005
                                 + 1.36578 w_k1_KO_s005 + 1.54213 w_k1_MAT_s005
                                 + 1.10395 w_k1_MSFT_s005
                                 + 1.09037 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.805198 w_k1_ADM_s006
                                 + 1.10339 w_k1_AXP_s006
                                 + 1.32922 w_k1_DIS_s006
                                 + 1.06815 w_k1_FDX_s006
                                 + 1.38072 w_k1_IBM_s006
                                 + 1.30177 w_k1_INTC_s006 + 1.5985 w_k1_KO_s006
                                 + 1.65846 w_k1_MAT_s006
                                 + 1.09734 w_k1_MSFT_s006
                                 + 1.17983 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.746471 w_k1_ADM_s006
                                 + 0.673501 w_k1_AXP_s006
                                 + 1.01457 w_k1_DIS_s006
                                 + 0.818469 w_k1_FDX_s006
                                 + 1.01025 w_k1_IBM_s006
                                 + 0.942439 w_k1_INTC_s006
                                 + 1.02132 w_k1_KO_s006 + 1.12327 w_k1_MAT_s006
                                 + 0.988495 w_k1_MSFT_s006
                                 + 0.863592 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.894549 w_k1_ADM_s006
                                 + 1.12798 w_k1_AXP_s006
                                 + 1.00048 w_k1_DIS_s006
                                 + 1.05721 w_k1_FDX_s006
                                 + 0.958892 w_k1_IBM_s006
                                 + 1.12147 w_k1_INTC_s006
                                 + 1.13519 w_k1_KO_s006
                                 + 0.868475 w_k1_MAT_s006
                                 + 0.977761 w_k1_MSFT_s006
                                 + 1.01786 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.932606 w_k1_ADM_s006
                                 + 0.814813 w_k1_AXP_s006
                                 + 0.812745 w_k1_DIS_s006
                                 + 0.811595 w_k1_FDX_s006
                                 + 0.915453 w_k1_IBM_s006
                                 + 0.769078 w_k1_INTC_s006
                                 + 1.06438 w_k1_KO_s006 + 1.0202 w_k1_MAT_s006
                                 + 0.833067 w_k1_MSFT_s006
                                 + 0.888066 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.21689 w_k1_ADM_s006
                                 + 1.76782 w_k1_AXP_s006
                                 + 1.48437 w_k1_DIS_s006
                                 + 1.32844 w_k1_FDX_s006
                                 + 1.22953 w_k1_IBM_s006
                                 + 1.25875 w_k1_INTC_s006
                                 + 1.43111 w_k1_KO_s006 + 1.79537 w_k1_MAT_s006
                                 + 1.26959 w_k1_MSFT_s006 + 1.172 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.777204 w_k1_ADM_s006
                                 + 0.689929 w_k1_AXP_s006
                                 + 0.682578 w_k1_DIS_s006
                                 + 0.838387 w_k1_FDX_s006
                                 + 0.676921 w_k1_IBM_s006
                                 + 0.988221 w_k1_INTC_s006
                                 + 0.802195 w_k1_KO_s006
                                 + 0.629963 w_k1_MAT_s006
                                 + 0.914056 w_k1_MSFT_s006
                                 + 0.788649 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.662192 w_k1_ADM_s006
                                 + 0.923049 w_k1_AXP_s006
                                 + 0.788842 w_k1_DIS_s006
                                 + 0.803561 w_k1_FDX_s006
                                 + 1.2831 w_k1_IBM_s006
                                 + 0.956842 w_k1_INTC_s006
                                 + 1.2084 w_k1_KO_s006 + 1.15804 w_k1_MAT_s006
                                 + 0.753186 w_k1_MSFT_s006
                                 + 0.833978 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.95538 w_k1_ADM_s006
                                 + 0.957079 w_k1_AXP_s006
                                 + 1.45381 w_k1_DIS_s006
                                 + 1.18476 w_k1_FDX_s006
                                 + 1.03752 w_k1_IBM_s006
                                 + 0.98712 w_k1_INTC_s006
                                 + 1.29811 w_k1_KO_s006 + 1.20746 w_k1_MAT_s006
                                 + 1.16411 w_k1_MSFT_s006
                                 + 1.13433 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.931791 w_k1_ADM_s007
                                 + 0.776034 w_k1_AXP_s007
                                 + 0.61372 w_k1_DIS_s007
                                 + 0.825878 w_k1_FDX_s007
                                 + 0.624263 w_k1_IBM_s007
                                 + 0.945437 w_k1_INTC_s007
                                 + 0.733519 w_k1_KO_s007
                                 + 0.402144 w_k1_MAT_s007
                                 + 0.814715 w_k1_MSFT_s007
                                 + 0.870964 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.998886 w_k1_ADM_s007
                                 + 1.07614 w_k1_AXP_s007
                                 + 0.930923 w_k1_DIS_s007
                                 + 0.823833 w_k1_FDX_s007
                                 + 1.30855 w_k1_IBM_s007
                                 + 1.10528 w_k1_INTC_s007
                                 + 1.16121 w_k1_KO_s007 + 1.13245 w_k1_MAT_s007
                                 + 1.06837 w_k1_MSFT_s007
                                 + 1.06587 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.17307 w_k1_ADM_s007
                                 + 0.943074 w_k1_AXP_s007
                                 + 1.39519 w_k1_DIS_s007
                                 + 1.07265 w_k1_FDX_s007
                                 + 1.27334 w_k1_IBM_s007
                                 + 1.18675 w_k1_INTC_s007
                                 + 1.30638 w_k1_KO_s007 + 1.04092 w_k1_MAT_s007
                                 + 1.17155 w_k1_MSFT_s007
                                 + 0.989335 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.28466 w_k1_ADM_s007
                                 + 1.14577 w_k1_AXP_s007
                                 + 1.02899 w_k1_DIS_s007
                                 + 1.20021 w_k1_FDX_s007
                                 + 0.684418 w_k1_IBM_s007
                                 + 1.13696 w_k1_INTC_s007
                                 + 1.08611 w_k1_KO_s007
                                 + 0.730343 w_k1_MAT_s007
                                 + 1.13664 w_k1_MSFT_s007
                                 + 1.04299 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.974206 w_k1_ADM_s007
                                 + 0.856083 w_k1_AXP_s007
                                 + 0.739889 w_k1_DIS_s007
                                 + 0.883902 w_k1_FDX_s007
                                 + 0.964746 w_k1_IBM_s007
                                 + 0.942964 w_k1_INTC_s007
                                 + 0.850554 w_k1_KO_s007
                                 + 0.838761 w_k1_MAT_s007
                                 + 0.899017 w_k1_MSFT_s007
                                 + 0.88896 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.04196 w_k1_ADM_s007
                                 + 0.819802 w_k1_AXP_s007
                                 + 1.09849 w_k1_DIS_s007
                                 + 0.982308 w_k1_FDX_s007
                                 + 0.790808 w_k1_IBM_s007
                                 + 1.05633 w_k1_INTC_s007
                                 + 1.02665 w_k1_KO_s007
                                 + 0.713433 w_k1_MAT_s007
                                 + 1.1069 w_k1_MSFT_s007 + 1.00239 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.728577 w_k1_ADM_s007
                                 + 0.603987 w_k1_AXP_s007
                                 + 0.596178 w_k1_DIS_s007
                                 + 0.645227 w_k1_FDX_s007
                                 + 0.726753 w_k1_IBM_s007
                                 + 0.737404 w_k1_INTC_s007
                                 + 0.651789 w_k1_KO_s007
                                 + 0.597275 w_k1_MAT_s007
                                 + 0.835963 w_k1_MSFT_s007
                                 + 0.770511 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.33489 w_k1_ADM_s007
                                 + 1.29695 w_k1_AXP_s007
                                 + 1.24475 w_k1_DIS_s007
                                 + 1.18382 w_k1_FDX_s007
                                 + 1.01962 w_k1_IBM_s007
                                 + 1.28084 w_k1_INTC_s007
                                 + 1.26594 w_k1_KO_s007
                                 + 0.868767 w_k1_MAT_s007
                                 + 1.14402 w_k1_MSFT_s007
                                 + 1.18891 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.821373 w_k1_ADM_s010
                                 + 0.93671 w_k1_AXP_s010
                                 + 0.608812 w_k1_DIS_s010 + 0.737 w_k1_FDX_s010
                                 + 1.45724 w_k1_IBM_s010
                                 + 0.644818 w_k1_INTC_s010
                                 + 0.807592 w_k1_KO_s010
                                 + 1.58482 w_k1_MAT_s010
                                 + 0.622175 w_k1_MSFT_s010
                                 + 0.831973 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.07429 w_k1_ADM_s010
                                 + 1.18266 w_k1_AXP_s010
                                 + 1.24641 w_k1_DIS_s010
                                 + 0.973625 w_k1_FDX_s010
                                 + 1.76075 w_k1_IBM_s010
                                 + 0.855821 w_k1_INTC_s010
                                 + 1.27397 w_k1_KO_s010 + 1.88838 w_k1_MAT_s010
                                 + 0.9737 w_k1_MSFT_s010
                                 + 0.970329 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.987397 w_k1_ADM_s010
                                 + 1.15632 w_k1_AXP_s010
                                 + 1.06375 w_k1_DIS_s010
                                 + 0.979556 w_k1_FDX_s010
                                 + 1.30774 w_k1_IBM_s010
                                 + 1.14944 w_k1_INTC_s010
                                 + 0.987334 w_k1_KO_s010
                                 + 1.39527 w_k1_MAT_s010
                                 + 1.0869 w_k1_MSFT_s010
                                 + 0.884238 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.03108 w_k1_ADM_s010
                                 + 1.07214 w_k1_AXP_s010
                                 + 1.00552 w_k1_DIS_s010
                                 + 1.11155 w_k1_FDX_s010
                                 + 0.705728 w_k1_IBM_s010
                                 + 1.12535 w_k1_INTC_s010
                                 + 0.807108 w_k1_KO_s010 + 1.1593 w_k1_MAT_s010
                                 + 1.10522 w_k1_MSFT_s010
                                 + 0.917805 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.970351 w_k1_ADM_s010
                                 + 0.820448 w_k1_AXP_s010
                                 + 0.809143 w_k1_DIS_s010
                                 + 0.903096 w_k1_FDX_s010
                                 + 0.849132 w_k1_IBM_s010
                                 + 0.849313 w_k1_INTC_s010
                                 + 0.859977 w_k1_KO_s010
                                 + 0.68616 w_k1_MAT_s010
                                 + 0.892199 w_k1_MSFT_s010
                                 + 0.744779 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.18914 w_k1_ADM_s010
                                 + 1.2343 w_k1_AXP_s010 + 1.37829 w_k1_DIS_s010
                                 + 1.18767 w_k1_FDX_s010
                                 + 1.06222 w_k1_IBM_s010
                                 + 0.952375 w_k1_INTC_s010
                                 + 1.15895 w_k1_KO_s010 + 1.40827 w_k1_MAT_s010
                                 + 1.21443 w_k1_MSFT_s010
                                 + 1.13435 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.07774 w_k1_ADM_s010
                                 + 1.02433 w_k1_AXP_s010
                                 + 1.31061 w_k1_DIS_s010
                                 + 1.20694 w_k1_FDX_s010
                                 + 0.843318 w_k1_IBM_s010
                                 + 1.02492 w_k1_INTC_s010
                                 + 0.831022 w_k1_KO_s010
                                 + 0.773801 w_k1_MAT_s010
                                 + 1.05977 w_k1_MSFT_s010
                                 + 0.925636 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.847081 w_k1_ADM_s010
                                 + 0.647425 w_k1_AXP_s010
                                 + 0.662999 w_k1_DIS_s010
                                 + 0.789247 w_k1_FDX_s010
                                 + 0.784551 w_k1_IBM_s010
                                 + 0.837586 w_k1_INTC_s010
                                 + 0.774445 w_k1_KO_s010
                                 + 0.728081 w_k1_MAT_s010
                                 + 0.848974 w_k1_MSFT_s010
                                 + 0.809782 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.26995 w_k1_ADM_s011
                                 + 1.32844 w_k1_AXP_s011
                                 + 1.54131 w_k1_DIS_s011
                                 + 1.42254 w_k1_FDX_s011
                                 + 0.922817 w_k1_IBM_s011
                                 + 1.28769 w_k1_INTC_s011
                                 + 1.26122 w_k1_KO_s011 + 1.12065 w_k1_MAT_s011
                                 + 1.37351 w_k1_MSFT_s011 + 1.1869 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.81215 w_k1_ADM_s011
                                 + 1.44094 w_k1_AXP_s011
                                 + 1.33769 w_k1_DIS_s011
                                 + 1.19381 w_k1_FDX_s011
                                 + 1.68302 w_k1_IBM_s011
                                 + 1.44824 w_k1_INTC_s011
                                 + 1.43166 w_k1_KO_s011 + 1.88855 w_k1_MAT_s011
                                 + 1.18272 w_k1_MSFT_s011
                                 + 1.20972 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.667578 w_k1_ADM_s011
                                 + 0.631975 w_k1_AXP_s011
                                 + 0.462549 w_k1_DIS_s011
                                 + 0.68888 w_k1_FDX_s011
                                 + 0.796951 w_k1_IBM_s011
                                 + 0.684457 w_k1_INTC_s011
                                 + 0.590227 w_k1_KO_s011
                                 + 0.814468 w_k1_MAT_s011
                                 + 0.724532 w_k1_MSFT_s011
                                 + 0.65076 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.672883 w_k1_ADM_s011
                                 + 0.7794 w_k1_AXP_s011
                                 + 0.909845 w_k1_DIS_s011
                                 + 0.95412 w_k1_FDX_s011
                                 + 0.661896 w_k1_IBM_s011
                                 + 0.983088 w_k1_INTC_s011
                                 + 0.690303 w_k1_KO_s011
                                 + 0.963216 w_k1_MAT_s011
                                 + 1.09095 w_k1_MSFT_s011
                                 + 0.715524 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.00796 w_k1_ADM_s011
                                 + 1.36695 w_k1_AXP_s011
                                 + 1.11216 w_k1_DIS_s011
                                 + 1.14055 w_k1_FDX_s011
                                 + 1.18074 w_k1_IBM_s011
                                 + 1.11959 w_k1_INTC_s011
                                 + 0.967484 w_k1_KO_s011
                                 + 1.23693 w_k1_MAT_s011
                                 + 0.960212 w_k1_MSFT_s011
                                 + 1.10181 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.782988 w_k1_ADM_s011
                                 + 0.904435 w_k1_AXP_s011
                                 + 1.25777 w_k1_DIS_s011
                                 + 1.03688 w_k1_FDX_s011
                                 + 1.15224 w_k1_IBM_s011
                                 + 1.0364 w_k1_INTC_s011 + 1.08544 w_k1_KO_s011
                                 + 1.37998 w_k1_MAT_s011
                                 + 1.21034 w_k1_MSFT_s011
                                 + 1.03793 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.768491 w_k1_ADM_s011
                                 + 0.934611 w_k1_AXP_s011
                                 + 0.741368 w_k1_DIS_s011
                                 + 0.904625 w_k1_FDX_s011
                                 + 1.0357 w_k1_IBM_s011
                                 + 1.16007 w_k1_INTC_s011
                                 + 0.862572 w_k1_KO_s011
                                 + 1.09166 w_k1_MAT_s011
                                 + 0.941629 w_k1_MSFT_s011
                                 + 0.922999 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.669565 w_k1_ADM_s011
                                 + 0.708275 w_k1_AXP_s011
                                 + 0.678209 w_k1_DIS_s011
                                 + 0.739332 w_k1_FDX_s011
                                 + 0.958218 w_k1_IBM_s011
                                 + 0.885444 w_k1_INTC_s011
                                 + 0.844782 w_k1_KO_s011
                                 + 1.01946 w_k1_MAT_s011
                                 + 1.0077 w_k1_MSFT_s011
                                 + 0.794674 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.950375 w_k1_ADM_s012
                                 + 0.735862 w_k1_AXP_s012
                                 + 1.07291 w_k1_DIS_s012
                                 + 0.950106 w_k1_FDX_s012
                                 + 0.798647 w_k1_IBM_s012
                                 + 0.928478 w_k1_INTC_s012
                                 + 0.95362 w_k1_KO_s012 + 1.03734 w_k1_MAT_s012
                                 + 1.05364 w_k1_MSFT_s012
                                 + 0.958846 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.858953 w_k1_ADM_s012
                                 + 0.81318 w_k1_AXP_s012
                                 + 0.712652 w_k1_DIS_s012
                                 + 0.807432 w_k1_FDX_s012
                                 + 0.65659 w_k1_IBM_s012
                                 + 0.713097 w_k1_INTC_s012
                                 + 0.66982 w_k1_KO_s012
                                 + 0.902867 w_k1_MAT_s012
                                 + 0.830596 w_k1_MSFT_s012
                                 + 0.853556 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.851863 w_k1_ADM_s012
                                 + 0.720192 w_k1_AXP_s012
                                 + 0.785661 w_k1_DIS_s012
                                 + 0.834376 w_k1_FDX_s012
                                 + 1.01091 w_k1_IBM_s012
                                 + 0.902124 w_k1_INTC_s012
                                 + 0.958857 w_k1_KO_s012
                                 + 1.10674 w_k1_MAT_s012
                                 + 0.839192 w_k1_MSFT_s012
                                 + 0.818481 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.92151 w_k1_ADM_s012
                                 + 0.972799 w_k1_AXP_s012
                                 + 1.0624 w_k1_DIS_s012 + 1.02584 w_k1_FDX_s012
                                 + 1.28148 w_k1_IBM_s012
                                 + 1.12245 w_k1_INTC_s012
                                 + 1.11838 w_k1_KO_s012 + 1.44263 w_k1_MAT_s012
                                 + 1.12628 w_k1_MSFT_s012
                                 + 1.06913 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.746565 w_k1_ADM_s012
                                 + 1.02365 w_k1_AXP_s012
                                 + 0.962985 w_k1_DIS_s012
                                 + 0.959628 w_k1_FDX_s012
                                 + 1.16644 w_k1_IBM_s012
                                 + 1.0152 w_k1_INTC_s012 + 1.05251 w_k1_KO_s012
                                 + 1.7409 w_k1_MAT_s012
                                 + 1.07405 w_k1_MSFT_s012
                                 + 1.05815 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.941025 w_k1_ADM_s012
                                 + 1.26621 w_k1_AXP_s012
                                 + 1.55009 w_k1_DIS_s012
                                 + 1.17507 w_k1_FDX_s012
                                 + 1.52356 w_k1_IBM_s012
                                 + 1.25874 w_k1_INTC_s012
                                 + 1.34228 w_k1_KO_s012 + 2.36444 w_k1_MAT_s012
                                 + 1.28044 w_k1_MSFT_s012
                                 + 1.10573 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.912184 w_k1_ADM_s012
                                 + 0.975189 w_k1_AXP_s012
                                 + 0.910905 w_k1_DIS_s012
                                 + 1.07149 w_k1_FDX_s012
                                 + 0.720598 w_k1_IBM_s012
                                 + 1.12705 w_k1_INTC_s012
                                 + 0.749422 w_k1_KO_s012
                                 + 0.981945 w_k1_MAT_s012
                                 + 1.01069 w_k1_MSFT_s012
                                 + 0.931407 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.13426 w_k1_ADM_s012
                                 + 1.28923 w_k1_AXP_s012
                                 + 1.32919 w_k1_DIS_s012
                                 + 1.24532 w_k1_FDX_s012
                                 + 1.11754 w_k1_IBM_s012
                                 + 1.06464 w_k1_INTC_s012
                                 + 1.24576 w_k1_KO_s012 + 1.51883 w_k1_MAT_s012
                                 + 1.22256 w_k1_MSFT_s012
                                 + 1.25551 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.779436 w_k1_ADM_s013
                                 + 0.790141 w_k1_AXP_s013
                                 + 0.786646 w_k1_DIS_s013
                                 + 0.744859 w_k1_FDX_s013
                                 + 1.06188 w_k1_IBM_s013
                                 + 0.836043 w_k1_INTC_s013
                                 + 0.922695 w_k1_KO_s013
                                 + 0.897519 w_k1_MAT_s013
                                 + 0.701393 w_k1_MSFT_s013
                                 + 0.88736 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.11201 w_k1_ADM_s013
                                 + 1.18217 w_k1_AXP_s013
                                 + 1.04394 w_k1_DIS_s013
                                 + 1.10817 w_k1_FDX_s013
                                 + 0.59281 w_k1_IBM_s013
                                 + 1.07566 w_k1_INTC_s013
                                 + 0.771394 w_k1_KO_s013
                                 + 0.581547 w_k1_MAT_s013
                                 + 0.932917 w_k1_MSFT_s013
                                 + 0.933296 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.08658 w_k1_ADM_s013
                                 + 1.52032 w_k1_AXP_s013
                                 + 1.53613 w_k1_DIS_s013
                                 + 1.21681 w_k1_FDX_s013
                                 + 1.56834 w_k1_IBM_s013
                                 + 1.13911 w_k1_INTC_s013
                                 + 1.34214 w_k1_KO_s013 + 1.83353 w_k1_MAT_s013
                                 + 1.05211 w_k1_MSFT_s013
                                 + 1.38876 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.07749 w_k1_ADM_s013
                                 + 1.05991 w_k1_AXP_s013
                                 + 1.24519 w_k1_DIS_s013
                                 + 1.10662 w_k1_FDX_s013
                                 + 0.781909 w_k1_IBM_s013
                                 + 1.11241 w_k1_INTC_s013
                                 + 0.962762 w_k1_KO_s013
                                 + 0.601414 w_k1_MAT_s013
                                 + 0.951742 w_k1_MSFT_s013
                                 + 1.08327 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.00305 w_k1_ADM_s013
                                 + 0.858928 w_k1_AXP_s013
                                 + 0.721295 w_k1_DIS_s013
                                 + 0.871865 w_k1_FDX_s013
                                 + 0.684303 w_k1_IBM_s013
                                 + 0.926311 w_k1_INTC_s013
                                 + 0.676993 w_k1_KO_s013
                                 + 0.76614 w_k1_MAT_s013
                                 + 0.786853 w_k1_MSFT_s013
                                 + 0.863677 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.03301 w_k1_ADM_s013
                                 + 1.30026 w_k1_AXP_s013 + 1.0197 w_k1_DIS_s013
                                 + 0.954262 w_k1_FDX_s013
                                 + 1.12878 w_k1_IBM_s013
                                 + 0.938564 w_k1_INTC_s013
                                 + 0.992524 w_k1_KO_s013
                                 + 1.12383 w_k1_MAT_s013
                                 + 0.817498 w_k1_MSFT_s013
                                 + 1.08145 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.890224 w_k1_ADM_s013
                                 + 1.02152 w_k1_AXP_s013
                                 + 1.15117 w_k1_DIS_s013
                                 + 1.00892 w_k1_FDX_s013
                                 + 1.22439 w_k1_IBM_s013
                                 + 0.964969 w_k1_INTC_s013
                                 + 1.1511 w_k1_KO_s013 + 1.15801 w_k1_MAT_s013
                                 + 0.904215 w_k1_MSFT_s013
                                 + 1.0976 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.15287 w_k1_ADM_s013
                                 + 1.75703 w_k1_AXP_s013
                                 + 1.42909 w_k1_DIS_s013
                                 + 1.28922 w_k1_FDX_s013
                                 + 1.04608 w_k1_IBM_s013
                                 + 1.30334 w_k1_INTC_s013
                                 + 1.10202 w_k1_KO_s013 + 1.11213 w_k1_MAT_s013
                                 + 0.977327 w_k1_MSFT_s013
                                 + 1.19948 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.35491 w_k1_ADM_s014
                                 + 0.983078 w_k1_AXP_s014
                                 + 1.1387 w_k1_DIS_s014 + 1.13421 w_k1_FDX_s014
                                 + 0.789579 w_k1_IBM_s014
                                 + 1.0241 w_k1_INTC_s014
                                 + 0.922643 w_k1_KO_s014
                                 + 0.744343 w_k1_MAT_s014
                                 + 1.03308 w_k1_MSFT_s014
                                 + 1.02431 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.585674 w_k1_ADM_s014
                                 + 0.80176 w_k1_AXP_s014
                                 + 0.690676 w_k1_DIS_s014
                                 + 0.735786 w_k1_FDX_s014
                                 + 0.908965 w_k1_IBM_s014
                                 + 0.885456 w_k1_INTC_s014
                                 + 0.781156 w_k1_KO_s014
                                 + 0.994384 w_k1_MAT_s014
                                 + 0.766708 w_k1_MSFT_s014
                                 + 0.805954 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.35591 w_k1_ADM_s014
                                 + 1.40832 w_k1_AXP_s014 + 1.4629 w_k1_DIS_s014
                                 + 1.19851 w_k1_FDX_s014
                                 + 1.37804 w_k1_IBM_s014
                                 + 1.20877 w_k1_INTC_s014
                                 + 1.30655 w_k1_KO_s014 + 1.26716 w_k1_MAT_s014
                                 + 1.13631 w_k1_MSFT_s014
                                 + 1.24148 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20016 w_k1_ADM_s014
                                 + 1.05896 w_k1_AXP_s014
                                 + 0.932177 w_k1_DIS_s014
                                 + 0.878696 w_k1_FDX_s014
                                 + 1.34187 w_k1_IBM_s014
                                 + 0.805941 w_k1_INTC_s014
                                 + 1.13745 w_k1_KO_s014 + 1.80262 w_k1_MAT_s014
                                 + 0.785012 w_k1_MSFT_s014
                                 + 1.04164 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.762949 w_k1_ADM_s014
                                 + 0.744852 w_k1_AXP_s014
                                 + 0.938481 w_k1_DIS_s014
                                 + 0.949449 w_k1_FDX_s014
                                 + 0.666131 w_k1_IBM_s014
                                 + 1.13215 w_k1_INTC_s014
                                 + 0.808845 w_k1_KO_s014
                                 + 0.549569 w_k1_MAT_s014
                                 + 1.00258 w_k1_MSFT_s014 + 1.054 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.820179 w_k1_ADM_s014
                                 + 0.618918 w_k1_AXP_s014
                                 + 0.58213 w_k1_DIS_s014
                                 + 0.730229 w_k1_FDX_s014
                                 + 0.613878 w_k1_IBM_s014
                                 + 0.821081 w_k1_INTC_s014
                                 + 0.58912 w_k1_KO_s014
                                 + 0.556435 w_k1_MAT_s014
                                 + 0.700497 w_k1_MSFT_s014
                                 + 0.734452 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.675503 w_k1_ADM_s014
                                 + 1.11189 w_k1_AXP_s014
                                 + 0.770649 w_k1_DIS_s014
                                 + 0.756114 w_k1_FDX_s014
                                 + 1.38586 w_k1_IBM_s014
                                 + 1.07349 w_k1_INTC_s014
                                 + 0.945593 w_k1_KO_s014
                                 + 1.62596 w_k1_MAT_s014
                                 + 0.843547 w_k1_MSFT_s014
                                 + 0.874152 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.13452 w_k1_ADM_s014
                                 + 1.0149 w_k1_AXP_s014 + 1.11332 w_k1_DIS_s014
                                 + 1.02034 w_k1_FDX_s014
                                 + 1.02619 w_k1_IBM_s014
                                 + 1.07372 w_k1_INTC_s014
                                 + 1.02736 w_k1_KO_s014 + 1.20105 w_k1_MAT_s014
                                 + 0.927478 w_k1_MSFT_s014
                                 + 1.07098 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.843145 w_k1_ADM_s015
                                 + 0.877702 w_k1_AXP_s015
                                 + 1.03971 w_k1_DIS_s015
                                 + 1.06519 w_k1_FDX_s015
                                 + 0.693177 w_k1_IBM_s015
                                 + 1.02662 w_k1_INTC_s015
                                 + 0.914723 w_k1_KO_s015
                                 + 0.995918 w_k1_MAT_s015
                                 + 1.23165 w_k1_MSFT_s015
                                 + 0.993328 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.16741 w_k1_ADM_s015
                                 + 1.13685 w_k1_AXP_s015
                                 + 1.51446 w_k1_DIS_s015
                                 + 1.15173 w_k1_FDX_s015
                                 + 0.885312 w_k1_IBM_s015
                                 + 0.903923 w_k1_INTC_s015
                                 + 1.22645 w_k1_KO_s015 + 1.25159 w_k1_MAT_s015
                                 + 1.07954 w_k1_MSFT_s015 + 1.1735 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.73111 w_k1_ADM_s015
                                 + 0.694676 w_k1_AXP_s015
                                 + 0.530119 w_k1_DIS_s015
                                 + 0.679172 w_k1_FDX_s015
                                 + 0.928873 w_k1_IBM_s015
                                 + 0.701221 w_k1_INTC_s015
                                 + 0.695914 w_k1_KO_s015
                                 + 0.892214 w_k1_MAT_s015
                                 + 0.769594 w_k1_MSFT_s015
                                 + 0.76849 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.24381 w_k1_ADM_s015
                                 + 1.30332 w_k1_AXP_s015
                                 + 1.35171 w_k1_DIS_s015
                                 + 1.35145 w_k1_FDX_s015
                                 + 1.13385 w_k1_IBM_s015
                                 + 1.21623 w_k1_INTC_s015
                                 + 1.07609 w_k1_KO_s015 + 1.21008 w_k1_MAT_s015
                                 + 1.24159 w_k1_MSFT_s015
                                 + 1.19378 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.811479 w_k1_ADM_s015
                                 + 0.874257 w_k1_AXP_s015
                                 + 0.771232 w_k1_DIS_s015
                                 + 0.87576 w_k1_FDX_s015
                                 + 0.723362 w_k1_IBM_s015
                                 + 0.898125 w_k1_INTC_s015
                                 + 0.793041 w_k1_KO_s015
                                 + 0.710958 w_k1_MAT_s015
                                 + 0.835008 w_k1_MSFT_s015
                                 + 0.936977 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.905413 w_k1_ADM_s015
                                 + 0.992425 w_k1_AXP_s015
                                 + 1.02062 w_k1_DIS_s015
                                 + 0.930176 w_k1_FDX_s015
                                 + 1.54722 w_k1_IBM_s015
                                 + 1.03623 w_k1_INTC_s015
                                 + 1.22273 w_k1_KO_s015 + 1.44683 w_k1_MAT_s015
                                 + 1.0357 w_k1_MSFT_s015 + 1.11024 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.840795 w_k1_ADM_s015
                                 + 0.787088 w_k1_AXP_s015
                                 + 0.890534 w_k1_DIS_s015
                                 + 0.932792 w_k1_FDX_s015
                                 + 0.929704 w_k1_IBM_s015
                                 + 0.713405 w_k1_INTC_s015
                                 + 0.934783 w_k1_KO_s015
                                 + 0.957927 w_k1_MAT_s015
                                 + 0.905205 w_k1_MSFT_s015
                                 + 1.07624 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.886531 w_k1_ADM_s015
                                 + 0.883608 w_k1_AXP_s015
                                 + 0.823382 w_k1_DIS_s015
                                 + 0.913064 w_k1_FDX_s015
                                 + 1.13414 w_k1_IBM_s015
                                 + 0.857533 w_k1_INTC_s015
                                 + 1.11575 w_k1_KO_s015 + 1.04607 w_k1_MAT_s015
                                 + 0.974249 w_k1_MSFT_s015
                                 + 1.00153 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.12148 w_k1_ADM_s016
                                 + 0.911855 w_k1_AXP_s016
                                 + 0.714796 w_k1_DIS_s016
                                 + 0.885155 w_k1_FDX_s016
                                 + 0.914585 w_k1_IBM_s016
                                 + 0.697092 w_k1_INTC_s016
                                 + 0.754489 w_k1_KO_s016
                                 + 0.714857 w_k1_MAT_s016
                                 + 0.694356 w_k1_MSFT_s016
                                 + 0.822291 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.06642 w_k1_ADM_s016
                                 + 1.03424 w_k1_AXP_s016
                                 + 1.07956 w_k1_DIS_s016
                                 + 0.975597 w_k1_FDX_s016
                                 + 1.45055 w_k1_IBM_s016
                                 + 1.1313 w_k1_INTC_s016 + 1.28768 w_k1_KO_s016
                                 + 1.26133 w_k1_MAT_s016
                                 + 1.02708 w_k1_MSFT_s016
                                 + 1.12559 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.15022 w_k1_ADM_s016
                                 + 0.76431 w_k1_AXP_s016
                                 + 0.956495 w_k1_DIS_s016
                                 + 0.964234 w_k1_FDX_s016
                                 + 0.604748 w_k1_IBM_s016
                                 + 0.9537 w_k1_INTC_s016
                                 + 0.755375 w_k1_KO_s016
                                 + 0.406667 w_k1_MAT_s016
                                 + 1.00358 w_k1_MSFT_s016
                                 + 0.922928 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.828374 w_k1_ADM_s016
                                 + 0.723041 w_k1_AXP_s016
                                 + 0.914347 w_k1_DIS_s016
                                 + 0.869206 w_k1_FDX_s016
                                 + 0.905988 w_k1_IBM_s016
                                 + 0.967741 w_k1_INTC_s016
                                 + 0.889329 w_k1_KO_s016
                                 + 0.525139 w_k1_MAT_s016
                                 + 0.96086 w_k1_MSFT_s016
                                 + 0.877928 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.32327 w_k1_ADM_s016
                                 + 0.85248 w_k1_AXP_s016
                                 + 1.22656 w_k1_DIS_s016
                                 + 1.04407 w_k1_FDX_s016
                                 + 1.14588 w_k1_IBM_s016
                                 + 0.818854 w_k1_INTC_s016
                                 + 1.15826 w_k1_KO_s016
                                 + 0.715221 w_k1_MAT_s016
                                 + 0.982337 w_k1_MSFT_s016
                                 + 0.911934 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.716954 w_k1_ADM_s016
                                 + 0.684441 w_k1_AXP_s016
                                 + 0.695413 w_k1_DIS_s016
                                 + 0.825828 w_k1_FDX_s016
                                 + 0.594285 w_k1_IBM_s016
                                 + 0.694776 w_k1_INTC_s016
                                 + 0.655974 w_k1_KO_s016
                                 + 0.720601 w_k1_MAT_s016
                                 + 0.709785 w_k1_MSFT_s016
                                 + 0.803776 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.33581 w_k1_ADM_s016
                                 + 1.22207 w_k1_AXP_s016
                                 + 1.25454 w_k1_DIS_s016 + 1.194 w_k1_FDX_s016
                                 + 1.02665 w_k1_IBM_s016
                                 + 1.0871 w_k1_INTC_s016
                                 + 0.907628 w_k1_KO_s016
                                 + 0.953403 w_k1_MAT_s016
                                 + 1.08641 w_k1_MSFT_s016
                                 + 0.940318 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.796388 w_k1_ADM_s016
                                 + 0.618663 w_k1_AXP_s016
                                 + 0.86559 w_k1_DIS_s016
                                 + 0.763035 w_k1_FDX_s016
                                 + 1.25127 w_k1_IBM_s016
                                 + 1.03718 w_k1_INTC_s016
                                 + 1.04212 w_k1_KO_s016
                                 + 0.901237 w_k1_MAT_s016
                                 + 0.947952 w_k1_MSFT_s016
                                 + 0.884935 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.07849 w_k1_ADM_s017
                                 + 1.46194 w_k1_AXP_s017
                                 + 1.76673 w_k1_DIS_s017
                                 + 1.49755 w_k1_FDX_s017
                                 + 0.994227 w_k1_IBM_s017
                                 + 1.73017 w_k1_INTC_s017
                                 + 1.50047 w_k1_KO_s017 + 1.20148 w_k1_MAT_s017
                                 + 1.49343 w_k1_MSFT_s017
                                 + 1.22687 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.939894 w_k1_ADM_s017
                                 + 0.942351 w_k1_AXP_s017
                                 + 0.906919 w_k1_DIS_s017
                                 + 1.04074 w_k1_FDX_s017
                                 + 0.823326 w_k1_IBM_s017
                                 + 1.12963 w_k1_INTC_s017
                                 + 1.04218 w_k1_KO_s017 + 1.08808 w_k1_MAT_s017
                                 + 1.18577 w_k1_MSFT_s017
                                 + 1.05059 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.912182 w_k1_ADM_s017
                                 + 1.09602 w_k1_AXP_s017
                                 + 1.01733 w_k1_DIS_s017
                                 + 0.909005 w_k1_FDX_s017
                                 + 1.08987 w_k1_IBM_s017
                                 + 1.11096 w_k1_INTC_s017
                                 + 1.32646 w_k1_KO_s017 + 1.44106 w_k1_MAT_s017
                                 + 1.1089 w_k1_MSFT_s017
                                 + 0.972628 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.861133 w_k1_ADM_s017
                                 + 0.81854 w_k1_AXP_s017
                                 + 0.860951 w_k1_DIS_s017
                                 + 0.878179 w_k1_FDX_s017
                                 + 0.798794 w_k1_IBM_s017
                                 + 0.857384 w_k1_INTC_s017
                                 + 0.896183 w_k1_KO_s017
                                 + 0.913991 w_k1_MAT_s017
                                 + 0.947794 w_k1_MSFT_s017
                                 + 0.871529 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.861848 w_k1_ADM_s017
                                 + 0.827469 w_k1_AXP_s017
                                 + 0.613402 w_k1_DIS_s017
                                 + 0.68509 w_k1_FDX_s017
                                 + 0.930653 w_k1_IBM_s017
                                 + 0.922701 w_k1_INTC_s017
                                 + 0.806115 w_k1_KO_s017
                                 + 0.880083 w_k1_MAT_s017
                                 + 0.884245 w_k1_MSFT_s017
                                 + 0.856421 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.09512 w_k1_ADM_s017
                                 + 1.64571 w_k1_AXP_s017
                                 + 1.76812 w_k1_DIS_s017 + 1.4343 w_k1_FDX_s017
                                 + 1.44304 w_k1_IBM_s017
                                 + 1.58091 w_k1_INTC_s017
                                 + 1.48519 w_k1_KO_s017 + 2.00498 w_k1_MAT_s017
                                 + 1.45471 w_k1_MSFT_s017
                                 + 1.23965 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.05548 w_k1_ADM_s017
                                 + 1.37626 w_k1_AXP_s017
                                 + 1.27475 w_k1_DIS_s017
                                 + 1.14035 w_k1_FDX_s017
                                 + 1.30375 w_k1_IBM_s017
                                 + 1.17461 w_k1_INTC_s017
                                 + 1.17946 w_k1_KO_s017 + 1.31783 w_k1_MAT_s017
                                 + 1.15651 w_k1_MSFT_s017 + 1.1104 w_k1_PG_s017
                                 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.97729 w_k1_ADM_s017
                                 + 0.978773 w_k1_AXP_s017
                                 + 1.18445 w_k1_DIS_s017
                                 + 0.978121 w_k1_FDX_s017
                                 + 1.17262 w_k1_IBM_s017
                                 + 1.10119 w_k1_INTC_s017
                                 + 1.23162 w_k1_KO_s017 + 1.02899 w_k1_MAT_s017
                                 + 1.16719 w_k1_MSFT_s017
                                 + 1.05763 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.04089 w_k1_ADM_s100
                                 + 1.02435 w_k1_AXP_s100
                                 + 1.00316 w_k1_DIS_s100
                                 + 0.91181 w_k1_FDX_s100
                                 + 1.22688 w_k1_IBM_s100
                                 + 1.27877 w_k1_INTC_s100
                                 + 1.13008 w_k1_KO_s100 + 1.38836 w_k1_MAT_s100
                                 + 1.12253 w_k1_MSFT_s100
                                 + 1.02713 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.15243 w_k1_ADM_s100
                                 + 1.33605 w_k1_AXP_s100
                                 + 0.805059 w_k1_DIS_s100
                                 + 1.0612 w_k1_FDX_s100 + 0.70584 w_k1_IBM_s100
                                 + 1.03069 w_k1_INTC_s100
                                 + 0.771256 w_k1_KO_s100
                                 + 1.39288 w_k1_MAT_s100
                                 + 1.0628 w_k1_MSFT_s100
                                 + 0.907311 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.85739 w_k1_ADM_s100
                                 + 1.50179 w_k1_AXP_s100 + 1.6506 w_k1_DIS_s100
                                 + 1.39293 w_k1_FDX_s100
                                 + 1.32305 w_k1_IBM_s100
                                 + 1.51915 w_k1_INTC_s100
                                 + 1.28599 w_k1_KO_s100 + 1.38489 w_k1_MAT_s100
                                 + 1.63029 w_k1_MSFT_s100
                                 + 1.38515 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.19392 w_k1_ADM_s100
                                 + 0.826235 w_k1_AXP_s100
                                 + 0.902939 w_k1_DIS_s100
                                 + 0.88631 w_k1_FDX_s100
                                 + 0.933144 w_k1_IBM_s100
                                 + 1.08253 w_k1_INTC_s100 + 1.0123 w_k1_KO_s100
                                 + 0.841126 w_k1_MAT_s100
                                 + 1.10115 w_k1_MSFT_s100
                                 + 0.947348 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.840342 w_k1_ADM_s100
                                 + 0.599696 w_k1_AXP_s100
                                 + 0.549469 w_k1_DIS_s100
                                 + 0.710149 w_k1_FDX_s100
                                 + 0.895492 w_k1_IBM_s100
                                 + 0.821617 w_k1_INTC_s100
                                 + 0.744342 w_k1_KO_s100
                                 + 0.917875 w_k1_MAT_s100
                                 + 0.78503 w_k1_MSFT_s100
                                 + 0.948463 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.50245 w_k1_ADM_s100
                                 + 1.37282 w_k1_AXP_s100 + 1.4695 w_k1_DIS_s100
                                 + 1.04431 w_k1_FDX_s100
                                 + 1.75349 w_k1_IBM_s100
                                 + 1.32988 w_k1_INTC_s100
                                 + 1.54677 w_k1_KO_s100 + 1.91377 w_k1_MAT_s100
                                 + 1.28901 w_k1_MSFT_s100
                                 + 1.18003 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.781787 w_k1_ADM_s100
                                 + 0.626187 w_k1_AXP_s100
                                 + 0.954097 w_k1_DIS_s100
                                 + 0.93485 w_k1_FDX_s100
                                 + 0.671287 w_k1_IBM_s100
                                 + 0.912911 w_k1_INTC_s100
                                 + 0.87912 w_k1_KO_s100
                                 + 0.728892 w_k1_MAT_s100
                                 + 1.33486 w_k1_MSFT_s100
                                 + 1.04108 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.993201 w_k1_ADM_s100
                                 + 1.19732 w_k1_AXP_s100
                                 + 1.33139 w_k1_DIS_s100
                                 + 1.15087 w_k1_FDX_s100
                                 + 0.943654 w_k1_IBM_s100
                                 + 1.30264 w_k1_INTC_s100
                                 + 1.06856 w_k1_KO_s100 + 1.33641 w_k1_MAT_s100
                                 + 1.38731 w_k1_MSFT_s100
                                 + 0.974621 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.521673 w_k1_ADM_s101
                                 + 0.798908 w_k1_AXP_s101
                                 + 0.774844 w_k1_DIS_s101
                                 + 0.715862 w_k1_FDX_s101 + 1.146 w_k1_IBM_s101
                                 + 1.00221 w_k1_INTC_s101
                                 + 0.971142 w_k1_KO_s101
                                 + 0.989082 w_k1_MAT_s101
                                 + 0.820444 w_k1_MSFT_s101
                                 + 0.96259 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.80928 w_k1_ADM_s101
                                 + 1.00039 w_k1_AXP_s101 + 1.007 w_k1_DIS_s101
                                 + 0.935463 w_k1_FDX_s101
                                 + 1.28813 w_k1_IBM_s101
                                 + 1.05225 w_k1_INTC_s101
                                 + 1.08686 w_k1_KO_s101 + 1.13287 w_k1_MAT_s101
                                 + 1.0593 w_k1_MSFT_s101 + 1.01682 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.795862 w_k1_ADM_s101
                                 + 0.687168 w_k1_AXP_s101
                                 + 0.841326 w_k1_DIS_s101
                                 + 0.786009 w_k1_FDX_s101
                                 + 0.950691 w_k1_IBM_s101
                                 + 1.06875 w_k1_INTC_s101
                                 + 0.878963 w_k1_KO_s101
                                 + 0.646477 w_k1_MAT_s101
                                 + 0.895649 w_k1_MSFT_s101
                                 + 0.969809 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.26241 w_k1_ADM_s101
                                 + 1.19742 w_k1_AXP_s101
                                 + 1.11295 w_k1_DIS_s101 + 1.0808 w_k1_FDX_s101
                                 + 1.57716 w_k1_IBM_s101
                                 + 1.03779 w_k1_INTC_s101
                                 + 1.27093 w_k1_KO_s101 + 1.50624 w_k1_MAT_s101
                                 + 1.05435 w_k1_MSFT_s101
                                 + 1.18922 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.02464 w_k1_ADM_s101
                                 + 1.09483 w_k1_AXP_s101
                                 + 1.23399 w_k1_DIS_s101
                                 + 1.12513 w_k1_FDX_s101
                                 + 1.05765 w_k1_IBM_s101
                                 + 1.21089 w_k1_INTC_s101
                                 + 1.16576 w_k1_KO_s101
                                 + 0.813631 w_k1_MAT_s101
                                 + 1.05135 w_k1_MSFT_s101
                                 + 1.15848 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.08505 w_k1_ADM_s101
                                 + 1.02738 w_k1_AXP_s101
                                 + 1.04647 w_k1_DIS_s101
                                 + 1.06368 w_k1_FDX_s101
                                 + 0.93212 w_k1_IBM_s101
                                 + 1.04029 w_k1_INTC_s101
                                 + 1.00945 w_k1_KO_s101
                                 + 0.862258 w_k1_MAT_s101
                                 + 1.15121 w_k1_MSFT_s101
                                 + 1.01878 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.17382 w_k1_ADM_s101
                                 + 1.39769 w_k1_AXP_s101
                                 + 1.59916 w_k1_DIS_s101
                                 + 1.37854 w_k1_FDX_s101
                                 + 1.30745 w_k1_IBM_s101
                                 + 1.24068 w_k1_INTC_s101
                                 + 1.37091 w_k1_KO_s101 + 1.22721 w_k1_MAT_s101
                                 + 1.27862 w_k1_MSFT_s101
                                 + 1.11817 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.762951 w_k1_ADM_s101
                                 + 0.748834 w_k1_AXP_s101
                                 + 0.746598 w_k1_DIS_s101
                                 + 0.838337 w_k1_FDX_s101
                                 + 0.584098 w_k1_IBM_s101
                                 + 0.883015 w_k1_INTC_s101
                                 + 0.685281 w_k1_KO_s101
                                 + 0.701701 w_k1_MAT_s101
                                 + 0.907662 w_k1_MSFT_s101
                                 + 0.966909 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.35718 w_k1_ADM_s102
                                 + 1.30163 w_k1_AXP_s102 + 1.0958 w_k1_DIS_s102
                                 + 1.10511 w_k1_FDX_s102
                                 + 1.21331 w_k1_IBM_s102
                                 + 0.929934 w_k1_INTC_s102
                                 + 1.06281 w_k1_KO_s102 + 1.20867 w_k1_MAT_s102
                                 + 1.1826 w_k1_MSFT_s102 + 1.11115 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.19945 w_k1_ADM_s102
                                 + 1.16181 w_k1_AXP_s102
                                 + 1.17792 w_k1_DIS_s102 + 1.3114 w_k1_FDX_s102
                                 + 0.926281 w_k1_IBM_s102
                                 + 1.01807 w_k1_INTC_s102
                                 + 1.11445 w_k1_KO_s102 + 1.01324 w_k1_MAT_s102
                                 + 1.32221 w_k1_MSFT_s102
                                 + 0.975509 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.742398 w_k1_ADM_s102
                                 + 0.72156 w_k1_AXP_s102
                                 + 0.660043 w_k1_DIS_s102
                                 + 0.806421 w_k1_FDX_s102
                                 + 0.783451 w_k1_IBM_s102
                                 + 0.652643 w_k1_INTC_s102
                                 + 0.73124 w_k1_KO_s102
                                 + 0.897319 w_k1_MAT_s102
                                 + 0.754407 w_k1_MSFT_s102
                                 + 0.82977 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.688552 w_k1_ADM_s102
                                 + 0.676794 w_k1_AXP_s102
                                 + 0.617976 w_k1_DIS_s102
                                 + 0.710639 w_k1_FDX_s102
                                 + 1.11171 w_k1_IBM_s102
                                 + 0.702135 w_k1_INTC_s102
                                 + 1.06834 w_k1_KO_s102 + 1.32748 w_k1_MAT_s102
                                 + 0.853492 w_k1_MSFT_s102
                                 + 0.866007 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.06024 w_k1_ADM_s102
                                 + 0.881258 w_k1_AXP_s102
                                 + 1.00616 w_k1_DIS_s102
                                 + 0.978871 w_k1_FDX_s102
                                 + 1.04093 w_k1_IBM_s102
                                 + 0.817646 w_k1_INTC_s102
                                 + 1.11478 w_k1_KO_s102
                                 + 0.926523 w_k1_MAT_s102
                                 + 1.06735 w_k1_MSFT_s102
                                 + 0.926237 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.31776 w_k1_ADM_s102
                                 + 1.0661 w_k1_AXP_s102 + 1.47295 w_k1_DIS_s102
                                 + 1.28125 w_k1_FDX_s102
                                 + 0.86123 w_k1_IBM_s102
                                 + 0.862098 w_k1_INTC_s102
                                 + 1.24606 w_k1_KO_s102 + 0.79172 w_k1_MAT_s102
                                 + 1.19412 w_k1_MSFT_s102
                                 + 1.12038 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.07576 w_k1_ADM_s102
                                 + 1.4579 w_k1_AXP_s102 + 1.05552 w_k1_DIS_s102
                                 + 1.08948 w_k1_FDX_s102
                                 + 0.93581 w_k1_IBM_s102
                                 + 0.937793 w_k1_INTC_s102
                                 + 1.03065 w_k1_KO_s102 + 1.13037 w_k1_MAT_s102
                                 + 1.06343 w_k1_MSFT_s102
                                 + 0.958893 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.06731 w_k1_ADM_s102
                                 + 1.02016 w_k1_AXP_s102 + 1.0043 w_k1_DIS_s102
                                 + 0.9587 w_k1_FDX_s102 + 1.43882 w_k1_IBM_s102
                                 + 0.732042 w_k1_INTC_s102
                                 + 1.31488 w_k1_KO_s102 + 1.91939 w_k1_MAT_s102
                                 + 1.00037 w_k1_MSFT_s102 + 1.1239 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.835532 w_k1_ADM_s103
                                 + 0.642785 w_k1_AXP_s103
                                 + 0.519336 w_k1_DIS_s103
                                 + 0.723794 w_k1_FDX_s103
                                 + 0.822915 w_k1_IBM_s103
                                 + 0.75825 w_k1_INTC_s103
                                 + 0.674603 w_k1_KO_s103
                                 + 0.68749 w_k1_MAT_s103
                                 + 0.734212 w_k1_MSFT_s103
                                 + 0.87615 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.14307 w_k1_ADM_s103
                                 + 1.34909 w_k1_AXP_s103
                                 + 1.33174 w_k1_DIS_s103
                                 + 1.31473 w_k1_FDX_s103
                                 + 1.33039 w_k1_IBM_s103
                                 + 1.48573 w_k1_INTC_s103
                                 + 1.59248 w_k1_KO_s103 + 1.34038 w_k1_MAT_s103
                                 + 1.28349 w_k1_MSFT_s103
                                 + 1.13859 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.758723 w_k1_ADM_s103
                                 + 1.04833 w_k1_AXP_s103
                                 + 0.863996 w_k1_DIS_s103
                                 + 0.872904 w_k1_FDX_s103
                                 + 1.39234 w_k1_IBM_s103
                                 + 0.974009 w_k1_INTC_s103
                                 + 1.12168 w_k1_KO_s103 + 1.59977 w_k1_MAT_s103
                                 + 0.893779 w_k1_MSFT_s103
                                 + 0.929742 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.857576 w_k1_ADM_s103
                                 + 1.23799 w_k1_AXP_s103
                                 + 0.752809 w_k1_DIS_s103
                                 + 1.01283 w_k1_FDX_s103 + 0.9613 w_k1_IBM_s103
                                 + 0.971871 w_k1_INTC_s103
                                 + 0.849412 w_k1_KO_s103
                                 + 0.851351 w_k1_MAT_s103
                                 + 0.799955 w_k1_MSFT_s103
                                 + 0.907643 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.846435 w_k1_ADM_s103
                                 + 0.863446 w_k1_AXP_s103
                                 + 0.738831 w_k1_DIS_s103
                                 + 0.834114 w_k1_FDX_s103
                                 + 1.10452 w_k1_IBM_s103
                                 + 0.879932 w_k1_INTC_s103
                                 + 1.0298 w_k1_KO_s103 + 0.841094 w_k1_MAT_s103
                                 + 0.850396 w_k1_MSFT_s103
                                 + 0.951193 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.847771 w_k1_ADM_s103
                                 + 0.982539 w_k1_AXP_s103
                                 + 0.690259 w_k1_DIS_s103
                                 + 0.884387 w_k1_FDX_s103
                                 + 0.732151 w_k1_IBM_s103
                                 + 0.853579 w_k1_INTC_s103
                                 + 0.86749 w_k1_KO_s103
                                 + 0.846537 w_k1_MAT_s103
                                 + 0.782014 w_k1_MSFT_s103
                                 + 0.818699 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.879345 w_k1_ADM_s103
                                 + 0.738736 w_k1_AXP_s103
                                 + 0.89151 w_k1_DIS_s103
                                 + 0.938796 w_k1_FDX_s103
                                 + 0.709448 w_k1_IBM_s103
                                 + 0.919064 w_k1_INTC_s103
                                 + 0.962385 w_k1_KO_s103
                                 + 0.593527 w_k1_MAT_s103
                                 + 0.972196 w_k1_MSFT_s103
                                 + 0.87152 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.24996 w_k1_ADM_s103
                                 + 1.08448 w_k1_AXP_s103
                                 + 1.01698 w_k1_DIS_s103
                                 + 1.17125 w_k1_FDX_s103
                                 + 0.876433 w_k1_IBM_s103
                                 + 1.14715 w_k1_INTC_s103
                                 + 1.06777 w_k1_KO_s103 + 1.0084 w_k1_MAT_s103
                                 + 1.11107 w_k1_MSFT_s103
                                 + 1.04684 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.857393 w_k1_ADM_s104
                                 + 1.0899 w_k1_AXP_s104 + 1.23991 w_k1_DIS_s104
                                 + 1.05113 w_k1_FDX_s104
                                 + 1.38289 w_k1_IBM_s104
                                 + 1.11703 w_k1_INTC_s104
                                 + 1.30603 w_k1_KO_s104 + 1.57261 w_k1_MAT_s104
                                 + 1.09082 w_k1_MSFT_s104
                                 + 1.13359 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.24464 w_k1_ADM_s104
                                 + 1.42997 w_k1_AXP_s104
                                 + 1.36686 w_k1_DIS_s104
                                 + 1.22654 w_k1_FDX_s104
                                 + 1.18533 w_k1_IBM_s104
                                 + 1.21733 w_k1_INTC_s104
                                 + 1.22256 w_k1_KO_s104 + 1.86178 w_k1_MAT_s104
                                 + 1.11309 w_k1_MSFT_s104
                                 + 1.13158 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.01276 w_k1_ADM_s104
                                 + 1.26305 w_k1_AXP_s104
                                 + 0.961438 w_k1_DIS_s104
                                 + 1.00296 w_k1_FDX_s104
                                 + 1.20985 w_k1_IBM_s104
                                 + 0.867832 w_k1_INTC_s104
                                 + 1.01673 w_k1_KO_s104 + 1.88641 w_k1_MAT_s104
                                 + 0.871182 w_k1_MSFT_s104
                                 + 1.08062 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.17618 w_k1_ADM_s104
                                 + 1.12785 w_k1_AXP_s104 + 1.662 w_k1_DIS_s104
                                 + 1.31751 w_k1_FDX_s104
                                 + 1.11756 w_k1_IBM_s104
                                 + 1.2411 w_k1_INTC_s104 + 1.25968 w_k1_KO_s104
                                 + 1.26838 w_k1_MAT_s104
                                 + 1.34201 w_k1_MSFT_s104
                                 + 1.04517 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.745689 w_k1_ADM_s104
                                 + 0.731687 w_k1_AXP_s104
                                 + 0.83523 w_k1_DIS_s104
                                 + 0.832463 w_k1_FDX_s104
                                 + 1.25399 w_k1_IBM_s104
                                 + 1.02027 w_k1_INTC_s104
                                 + 1.01639 w_k1_KO_s104 + 1.38129 w_k1_MAT_s104
                                 + 0.913723 w_k1_MSFT_s104
                                 + 0.909469 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.1828 w_k1_ADM_s104
                                 + 0.88134 w_k1_AXP_s104
                                 + 1.11937 w_k1_DIS_s104
                                 + 1.04046 w_k1_FDX_s104
                                 + 1.02068 w_k1_IBM_s104
                                 + 1.13853 w_k1_INTC_s104 + 1.1545 w_k1_KO_s104
                                 + 1.08319 w_k1_MAT_s104
                                 + 0.953708 w_k1_MSFT_s104
                                 + 0.95574 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.22413 w_k1_ADM_s104
                                 + 1.07668 w_k1_AXP_s104
                                 + 0.954469 w_k1_DIS_s104
                                 + 1.12571 w_k1_FDX_s104
                                 + 0.734526 w_k1_IBM_s104
                                 + 1.18201 w_k1_INTC_s104
                                 + 0.891592 w_k1_KO_s104
                                 + 1.16287 w_k1_MAT_s104
                                 + 1.12338 w_k1_MSFT_s104
                                 + 0.978102 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.637486 w_k1_ADM_s104
                                 + 0.695889 w_k1_AXP_s104
                                 + 0.738573 w_k1_DIS_s104
                                 + 0.801446 w_k1_FDX_s104
                                 + 0.806464 w_k1_IBM_s104
                                 + 0.942495 w_k1_INTC_s104
                                 + 0.804763 w_k1_KO_s104
                                 + 0.755415 w_k1_MAT_s104
                                 + 0.731621 w_k1_MSFT_s104
                                 + 0.866418 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.863934 w_k1_ADM_s105
                                 + 1.49792 w_k1_AXP_s105
                                 + 0.961782 w_k1_DIS_s105
                                 + 1.05756 w_k1_FDX_s105
                                 + 1.21443 w_k1_IBM_s105
                                 + 1.40679 w_k1_INTC_s105
                                 + 1.07917 w_k1_KO_s105 + 1.59026 w_k1_MAT_s105
                                 + 0.89388 w_k1_MSFT_s105
                                 + 1.26446 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.84242 w_k1_ADM_s105
                                 + 1.00193 w_k1_AXP_s105
                                 + 0.954903 w_k1_DIS_s105
                                 + 0.849059 w_k1_FDX_s105
                                 + 0.880339 w_k1_IBM_s105
                                 + 1.17696 w_k1_INTC_s105
                                 + 1.01317 w_k1_KO_s105
                                 + 0.908165 w_k1_MAT_s105
                                 + 0.893515 w_k1_MSFT_s105
                                 + 1.0668 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.842244 w_k1_ADM_s105
                                 + 1.02132 w_k1_AXP_s105
                                 + 0.772829 w_k1_DIS_s105
                                 + 0.753156 w_k1_FDX_s105
                                 + 1.18403 w_k1_IBM_s105
                                 + 1.04031 w_k1_INTC_s105
                                 + 0.97031 w_k1_KO_s105 + 1.00908 w_k1_MAT_s105
                                 + 0.830328 w_k1_MSFT_s105
                                 + 0.938091 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.2493 w_k1_ADM_s105
                                 + 0.985066 w_k1_AXP_s105
                                 + 1.32834 w_k1_DIS_s105
                                 + 0.989542 w_k1_FDX_s105
                                 + 1.24553 w_k1_IBM_s105
                                 + 1.40402 w_k1_INTC_s105
                                 + 1.25367 w_k1_KO_s105
                                 + 0.793297 w_k1_MAT_s105
                                 + 1.10597 w_k1_MSFT_s105
                                 + 1.05994 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.902355 w_k1_ADM_s105
                                 + 1.48458 w_k1_AXP_s105
                                 + 1.59543 w_k1_DIS_s105
                                 + 1.24177 w_k1_FDX_s105
                                 + 1.63208 w_k1_IBM_s105
                                 + 1.56327 w_k1_INTC_s105
                                 + 1.39214 w_k1_KO_s105 + 1.4452 w_k1_MAT_s105
                                 + 1.21237 w_k1_MSFT_s105 + 1.1865 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.854263 w_k1_ADM_s105
                                 + 1.12996 w_k1_AXP_s105 + 1.0658 w_k1_DIS_s105
                                 + 1.10869 w_k1_FDX_s105
                                 + 0.850211 w_k1_IBM_s105
                                 + 1.26561 w_k1_INTC_s105
                                 + 0.920542 w_k1_KO_s105
                                 + 0.968592 w_k1_MAT_s105
                                 + 1.02448 w_k1_MSFT_s105
                                 + 1.08226 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.20916 w_k1_ADM_s105
                                 + 1.35498 w_k1_AXP_s105
                                 + 1.42928 w_k1_DIS_s105 + 1.2877 w_k1_FDX_s105
                                 + 1.0471 w_k1_IBM_s105
                                 + 1.41965 w_k1_INTC_s105 + 1.0887 w_k1_KO_s105
                                 + 1.24505 w_k1_MAT_s105
                                 + 1.26443 w_k1_MSFT_s105
                                 + 1.28441 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.859229 w_k1_ADM_s105
                                 + 0.694721 w_k1_AXP_s105
                                 + 0.68323 w_k1_DIS_s105
                                 + 0.802167 w_k1_FDX_s105
                                 + 0.747058 w_k1_IBM_s105
                                 + 0.79103 w_k1_INTC_s105
                                 + 0.680357 w_k1_KO_s105
                                 + 0.568524 w_k1_MAT_s105
                                 + 0.718866 w_k1_MSFT_s105
                                 + 0.931707 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.82674 w_k1_ADM_s106
                                 + 1.09082 w_k1_AXP_s106
                                 + 0.893397 w_k1_DIS_s106
                                 + 0.940114 w_k1_FDX_s106
                                 + 0.949043 w_k1_IBM_s106
                                 + 1.09437 w_k1_INTC_s106
                                 + 1.05925 w_k1_KO_s106
                                 + 0.905424 w_k1_MAT_s106
                                 + 1.03126 w_k1_MSFT_s106
                                 + 1.00747 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.887042 w_k1_ADM_s106
                                 + 0.93604 w_k1_AXP_s106
                                 + 0.641999 w_k1_DIS_s106
                                 + 0.754236 w_k1_FDX_s106
                                 + 0.93747 w_k1_IBM_s106
                                 + 0.775563 w_k1_INTC_s106
                                 + 0.801417 w_k1_KO_s106
                                 + 0.992041 w_k1_MAT_s106
                                 + 0.728087 w_k1_MSFT_s106
                                 + 0.820509 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.673789 w_k1_ADM_s106
                                 + 0.600386 w_k1_AXP_s106
                                 + 0.739816 w_k1_DIS_s106
                                 + 0.860258 w_k1_FDX_s106
                                 + 0.622015 w_k1_IBM_s106
                                 + 1.00639 w_k1_INTC_s106
                                 + 0.697122 w_k1_KO_s106
                                 + 0.529518 w_k1_MAT_s106
                                 + 0.819951 w_k1_MSFT_s106
                                 + 0.805507 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.10105 w_k1_ADM_s106
                                 + 1.6187 w_k1_AXP_s106 + 1.69413 w_k1_DIS_s106
                                 + 1.37733 w_k1_FDX_s106
                                 + 1.27186 w_k1_IBM_s106
                                 + 1.38052 w_k1_INTC_s106
                                 + 1.18433 w_k1_KO_s106 + 1.19269 w_k1_MAT_s106
                                 + 1.2969 w_k1_MSFT_s106 + 1.09584 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.997102 w_k1_ADM_s106
                                 + 1.31423 w_k1_AXP_s106 + 1.0676 w_k1_DIS_s106
                                 + 0.943565 w_k1_FDX_s106
                                 + 1.43637 w_k1_IBM_s106
                                 + 1.33882 w_k1_INTC_s106
                                 + 1.47622 w_k1_KO_s106 + 1.20451 w_k1_MAT_s106
                                 + 1.04772 w_k1_MSFT_s106
                                 + 1.11262 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.936804 w_k1_ADM_s106
                                 + 1.01972 w_k1_AXP_s106 + 1.3349 w_k1_DIS_s106
                                 + 1.13618 w_k1_FDX_s106
                                 + 0.739758 w_k1_IBM_s106
                                 + 1.2068 w_k1_INTC_s106 + 1.07616 w_k1_KO_s106
                                 + 0.572399 w_k1_MAT_s106
                                 + 1.14934 w_k1_MSFT_s106
                                 + 0.996824 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.900746 w_k1_ADM_s106
                                 + 1.11981 w_k1_AXP_s106
                                 + 0.862747 w_k1_DIS_s106
                                 + 1.10225 w_k1_FDX_s106
                                 + 0.663596 w_k1_IBM_s106
                                 + 1.20599 w_k1_INTC_s106
                                 + 0.693165 w_k1_KO_s106
                                 + 0.650248 w_k1_MAT_s106
                                 + 0.988778 w_k1_MSFT_s106
                                 + 0.844574 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.909986 w_k1_ADM_s106
                                 + 1.37776 w_k1_AXP_s106
                                 + 1.25946 w_k1_DIS_s106 + 1.1493 w_k1_FDX_s106
                                 + 1.0204 w_k1_IBM_s106
                                 + 1.28991 w_k1_INTC_s106
                                 + 1.04615 w_k1_KO_s106
                                 + 0.818228 w_k1_MAT_s106
                                 + 1.15778 w_k1_MSFT_s106
                                 + 1.07788 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.823711 w_k1_ADM_s107
                                 + 1.05063 w_k1_AXP_s107
                                 + 0.816197 w_k1_DIS_s107
                                 + 0.972582 w_k1_FDX_s107
                                 + 0.732363 w_k1_IBM_s107
                                 + 0.99087 w_k1_INTC_s107
                                 + 0.827331 w_k1_KO_s107
                                 + 0.945083 w_k1_MAT_s107
                                 + 1.03387 w_k1_MSFT_s107
                                 + 1.03006 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.05215 w_k1_ADM_s107
                                 + 1.08863 w_k1_AXP_s107
                                 + 0.956993 w_k1_DIS_s107
                                 + 0.929771 w_k1_FDX_s107
                                 + 0.761068 w_k1_IBM_s107
                                 + 0.733317 w_k1_INTC_s107
                                 + 0.795388 w_k1_KO_s107
                                 + 0.703138 w_k1_MAT_s107
                                 + 0.748855 w_k1_MSFT_s107
                                 + 0.948936 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.808116 w_k1_ADM_s107
                                 + 0.960419 w_k1_AXP_s107
                                 + 0.871313 w_k1_DIS_s107
                                 + 0.949086 w_k1_FDX_s107
                                 + 1.06758 w_k1_IBM_s107
                                 + 0.919508 w_k1_INTC_s107
                                 + 0.956106 w_k1_KO_s107
                                 + 1.05506 w_k1_MAT_s107
                                 + 0.850308 w_k1_MSFT_s107
                                 + 0.98134 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.81072 w_k1_ADM_s107
                                 + 0.726574 w_k1_AXP_s107
                                 + 0.644723 w_k1_DIS_s107
                                 + 0.709215 w_k1_FDX_s107
                                 + 0.726303 w_k1_IBM_s107
                                 + 0.724659 w_k1_INTC_s107
                                 + 0.789132 w_k1_KO_s107
                                 + 0.753256 w_k1_MAT_s107
                                 + 0.68547 w_k1_MSFT_s107
                                 + 0.824673 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.955593 w_k1_ADM_s107
                                 + 1.20732 w_k1_AXP_s107
                                 + 1.24753 w_k1_DIS_s107
                                 + 1.19313 w_k1_FDX_s107
                                 + 0.980449 w_k1_IBM_s107
                                 + 1.17331 w_k1_INTC_s107
                                 + 1.21208 w_k1_KO_s107 + 1.05628 w_k1_MAT_s107
                                 + 1.09388 w_k1_MSFT_s107
                                 + 1.25697 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.12795 w_k1_ADM_s107
                                 + 0.912803 w_k1_AXP_s107
                                 + 0.958822 w_k1_DIS_s107
                                 + 1.03821 w_k1_FDX_s107
                                 + 0.625635 w_k1_IBM_s107
                                 + 0.842323 w_k1_INTC_s107
                                 + 0.850097 w_k1_KO_s107
                                 + 0.620892 w_k1_MAT_s107
                                 + 0.922718 w_k1_MSFT_s107
                                 + 0.965561 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.1764 w_k1_ADM_s107
                                 + 1.46629 w_k1_AXP_s107 + 1.431 w_k1_DIS_s107
                                 + 1.26921 w_k1_FDX_s107
                                 + 1.35166 w_k1_IBM_s107
                                 + 1.01375 w_k1_INTC_s107 + 1.3848 w_k1_KO_s107
                                 + 1.82364 w_k1_MAT_s107
                                 + 1.30231 w_k1_MSFT_s107
                                 + 1.46414 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.827028 w_k1_ADM_s107
                                 + 0.859504 w_k1_AXP_s107
                                 + 0.816167 w_k1_DIS_s107
                                 + 0.74718 w_k1_FDX_s107
                                 + 1.12102 w_k1_IBM_s107
                                 + 0.916588 w_k1_INTC_s107
                                 + 0.945394 w_k1_KO_s107
                                 + 0.808794 w_k1_MAT_s107
                                 + 0.742987 w_k1_MSFT_s107
                                 + 0.803968 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.12602 w_k1_ADM_s110
                                 + 1.16401 w_k1_AXP_s110
                                 + 1.30796 w_k1_DIS_s110
                                 + 1.11274 w_k1_FDX_s110
                                 + 1.23353 w_k1_IBM_s110
                                 + 0.787711 w_k1_INTC_s110
                                 + 1.09192 w_k1_KO_s110 + 1.36652 w_k1_MAT_s110
                                 + 1.05319 w_k1_MSFT_s110
                                 + 1.08344 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.69425 w_k1_ADM_s110
                                 + 0.904955 w_k1_AXP_s110
                                 + 0.864388 w_k1_DIS_s110
                                 + 0.881075 w_k1_FDX_s110
                                 + 1.0574 w_k1_IBM_s110
                                 + 0.878037 w_k1_INTC_s110
                                 + 0.816848 w_k1_KO_s110
                                 + 0.98176 w_k1_MAT_s110
                                 + 0.903863 w_k1_MSFT_s110
                                 + 0.85578 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.961795 w_k1_ADM_s110
                                 + 1.35249 w_k1_AXP_s110
                                 + 1.35338 w_k1_DIS_s110
                                 + 1.18529 w_k1_FDX_s110
                                 + 1.71365 w_k1_IBM_s110
                                 + 1.19003 w_k1_INTC_s110 + 1.2353 w_k1_KO_s110
                                 + 1.75329 w_k1_MAT_s110
                                 + 1.09817 w_k1_MSFT_s110
                                 + 1.13115 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.668408 w_k1_ADM_s110
                                 + 0.577145 w_k1_AXP_s110
                                 + 0.599923 w_k1_DIS_s110
                                 + 0.715391 w_k1_FDX_s110
                                 + 0.874712 w_k1_IBM_s110
                                 + 0.677638 w_k1_INTC_s110
                                 + 0.757216 w_k1_KO_s110
                                 + 0.75347 w_k1_MAT_s110
                                 + 0.79529 w_k1_MSFT_s110
                                 + 0.71178 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.13159 w_k1_ADM_s110
                                 + 1.12927 w_k1_AXP_s110
                                 + 1.00648 w_k1_DIS_s110
                                 + 1.10155 w_k1_FDX_s110
                                 + 0.822643 w_k1_IBM_s110
                                 + 0.874727 w_k1_INTC_s110
                                 + 0.962603 w_k1_KO_s110 + 0.8563 w_k1_MAT_s110
                                 + 0.944509 w_k1_MSFT_s110
                                 + 0.974916 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.826515 w_k1_ADM_s110
                                 + 0.807569 w_k1_AXP_s110
                                 + 0.71061 w_k1_DIS_s110
                                 + 0.898352 w_k1_FDX_s110
                                 + 0.616164 w_k1_IBM_s110
                                 + 0.78732 w_k1_INTC_s110
                                 + 0.679493 w_k1_KO_s110
                                 + 0.851267 w_k1_MAT_s110
                                 + 0.827808 w_k1_MSFT_s110
                                 + 0.770155 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.668562 w_k1_ADM_s110
                                 + 0.60565 w_k1_AXP_s110
                                 + 0.517227 w_k1_DIS_s110
                                 + 0.633413 w_k1_FDX_s110
                                 + 0.705825 w_k1_IBM_s110
                                 + 0.567088 w_k1_INTC_s110
                                 + 0.583096 w_k1_KO_s110
                                 + 0.546175 w_k1_MAT_s110
                                 + 0.720329 w_k1_MSFT_s110
                                 + 0.694288 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.21247 w_k1_ADM_s110
                                 + 1.55929 w_k1_AXP_s110
                                 + 1.54275 w_k1_DIS_s110
                                 + 1.49536 w_k1_FDX_s110
                                 + 0.954266 w_k1_IBM_s110
                                 + 1.21169 w_k1_INTC_s110
                                 + 0.929551 w_k1_KO_s110 + 0.9949 w_k1_MAT_s110
                                 + 1.27172 w_k1_MSFT_s110
                                 + 1.03871 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.17202 w_k1_ADM_s111
                                 + 1.15549 w_k1_AXP_s111
                                 + 0.91408 w_k1_DIS_s111
                                 + 0.848058 w_k1_FDX_s111
                                 + 1.38047 w_k1_IBM_s111
                                 + 0.909622 w_k1_INTC_s111
                                 + 1.06944 w_k1_KO_s111 + 1.31511 w_k1_MAT_s111
                                 + 0.836306 w_k1_MSFT_s111
                                 + 0.942946 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.31696 w_k1_ADM_s111
                                 + 1.20023 w_k1_AXP_s111
                                 + 1.12797 w_k1_DIS_s111
                                 + 1.16476 w_k1_FDX_s111
                                 + 0.906761 w_k1_IBM_s111
                                 + 0.996163 w_k1_INTC_s111
                                 + 0.988751 w_k1_KO_s111
                                 + 0.998768 w_k1_MAT_s111
                                 + 0.97954 w_k1_MSFT_s111
                                 + 1.08521 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.16421 w_k1_ADM_s111
                                 + 1.18831 w_k1_AXP_s111
                                 + 1.49783 w_k1_DIS_s111
                                 + 1.25505 w_k1_FDX_s111
                                 + 1.13374 w_k1_IBM_s111
                                 + 1.18311 w_k1_INTC_s111
                                 + 1.28515 w_k1_KO_s111 + 1.19192 w_k1_MAT_s111
                                 + 1.26094 w_k1_MSFT_s111
                                 + 1.09609 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.28069 w_k1_ADM_s111
                                 + 1.27788 w_k1_AXP_s111 + 1.2117 w_k1_DIS_s111
                                 + 1.04319 w_k1_FDX_s111
                                 + 1.36022 w_k1_IBM_s111
                                 + 1.03214 w_k1_INTC_s111
                                 + 1.22377 w_k1_KO_s111 + 1.40977 w_k1_MAT_s111
                                 + 1.12418 w_k1_MSFT_s111
                                 + 1.12717 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.830834 w_k1_ADM_s111
                                 + 0.710587 w_k1_AXP_s111
                                 + 0.673899 w_k1_DIS_s111
                                 + 0.744264 w_k1_FDX_s111
                                 + 0.836117 w_k1_IBM_s111
                                 + 0.706381 w_k1_INTC_s111
                                 + 0.725137 w_k1_KO_s111
                                 + 0.603703 w_k1_MAT_s111
                                 + 0.693385 w_k1_MSFT_s111
                                 + 0.786332 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.45494 w_k1_ADM_s111
                                 + 1.38497 w_k1_AXP_s111 + 1.4751 w_k1_DIS_s111
                                 + 1.25536 w_k1_FDX_s111
                                 + 1.64884 w_k1_IBM_s111
                                 + 1.08941 w_k1_INTC_s111
                                 + 1.50709 w_k1_KO_s111 + 1.66244 w_k1_MAT_s111
                                 + 1.33844 w_k1_MSFT_s111
                                 + 1.20239 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.15421 w_k1_ADM_s111
                                 + 0.886183 w_k1_AXP_s111
                                 + 0.917382 w_k1_DIS_s111
                                 + 0.928324 w_k1_FDX_s111
                                 + 0.892605 w_k1_IBM_s111
                                 + 0.760391 w_k1_INTC_s111
                                 + 0.866166 w_k1_KO_s111
                                 + 0.921502 w_k1_MAT_s111
                                 + 0.925873 w_k1_MSFT_s111
                                 + 0.9213 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.20104 w_k1_ADM_s111
                                 + 0.814858 w_k1_AXP_s111
                                 + 1.2124 w_k1_DIS_s111
                                 + 0.933485 w_k1_FDX_s111
                                 + 1.01532 w_k1_IBM_s111
                                 + 0.9118 w_k1_INTC_s111 + 1.07113 w_k1_KO_s111
                                 + 0.842394 w_k1_MAT_s111
                                 + 1.07575 w_k1_MSFT_s111
                                 + 1.00197 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.42635 w_k1_ADM_s112
                                 + 1.32275 w_k1_AXP_s112 + 1.7523 w_k1_DIS_s112
                                 + 1.4855 w_k1_FDX_s112 + 1.01248 w_k1_IBM_s112
                                 + 1.25561 w_k1_INTC_s112
                                 + 1.06445 w_k1_KO_s112
                                 + 0.845536 w_k1_MAT_s112
                                 + 1.20584 w_k1_MSFT_s112
                                 + 1.11319 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.12637 w_k1_ADM_s112
                                 + 1.44486 w_k1_AXP_s112
                                 + 1.71569 w_k1_DIS_s112
                                 + 1.34848 w_k1_FDX_s112
                                 + 1.55015 w_k1_IBM_s112
                                 + 1.50339 w_k1_INTC_s112
                                 + 1.30544 w_k1_KO_s112 + 1.2935 w_k1_MAT_s112
                                 + 1.16511 w_k1_MSFT_s112 + 1.0777 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.03518 w_k1_ADM_s112
                                 + 0.910172 w_k1_AXP_s112
                                 + 1.09152 w_k1_DIS_s112
                                 + 0.960398 w_k1_FDX_s112
                                 + 1.18708 w_k1_IBM_s112
                                 + 1.11564 w_k1_INTC_s112
                                 + 0.925636 w_k1_KO_s112
                                 + 0.901621 w_k1_MAT_s112
                                 + 0.995544 w_k1_MSFT_s112
                                 + 0.888735 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.864824 w_k1_ADM_s112
                                 + 0.747871 w_k1_AXP_s112
                                 + 0.607713 w_k1_DIS_s112
                                 + 0.685832 w_k1_FDX_s112
                                 + 0.889053 w_k1_IBM_s112
                                 + 0.804664 w_k1_INTC_s112
                                 + 0.67085 w_k1_KO_s112
                                 + 0.906957 w_k1_MAT_s112
                                 + 0.769026 w_k1_MSFT_s112
                                 + 0.805957 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.02997 w_k1_ADM_s112
                                 + 1.28667 w_k1_AXP_s112
                                 + 1.20634 w_k1_DIS_s112
                                 + 1.18096 w_k1_FDX_s112
                                 + 0.887312 w_k1_IBM_s112
                                 + 1.20644 w_k1_INTC_s112
                                 + 0.969977 w_k1_KO_s112
                                 + 0.930168 w_k1_MAT_s112
                                 + 1.09076 w_k1_MSFT_s112
                                 + 1.03303 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.08352 w_k1_ADM_s112
                                 + 0.820118 w_k1_AXP_s112
                                 + 0.85451 w_k1_DIS_s112
                                 + 0.897196 w_k1_FDX_s112
                                 + 0.703499 w_k1_IBM_s112
                                 + 0.952724 w_k1_INTC_s112
                                 + 0.741968 w_k1_KO_s112
                                 + 0.430228 w_k1_MAT_s112
                                 + 0.90194 w_k1_MSFT_s112
                                 + 0.911789 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.854558 w_k1_ADM_s112
                                 + 1.02286 w_k1_AXP_s112
                                 + 1.07289 w_k1_DIS_s112
                                 + 0.881737 w_k1_FDX_s112
                                 + 1.45558 w_k1_IBM_s112
                                 + 0.997461 w_k1_INTC_s112
                                 + 0.985703 w_k1_KO_s112
                                 + 1.12616 w_k1_MAT_s112
                                 + 0.931358 w_k1_MSFT_s112
                                 + 0.921422 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.30671 w_k1_ADM_s112
                                 + 1.06873 w_k1_AXP_s112
                                 + 1.30409 w_k1_DIS_s112
                                 + 1.08922 w_k1_FDX_s112
                                 + 1.08047 w_k1_IBM_s112
                                 + 1.00254 w_k1_INTC_s112
                                 + 0.962254 w_k1_KO_s112
                                 + 0.864356 w_k1_MAT_s112
                                 + 1.06737 w_k1_MSFT_s112
                                 + 0.992833 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.71386 w_k1_ADM_s113
                                 + 0.656147 w_k1_AXP_s113
                                 + 0.547871 w_k1_DIS_s113
                                 + 0.675317 w_k1_FDX_s113
                                 + 0.946839 w_k1_IBM_s113
                                 + 0.677282 w_k1_INTC_s113
                                 + 0.679275 w_k1_KO_s113
                                 + 0.869748 w_k1_MAT_s113
                                 + 0.637274 w_k1_MSFT_s113
                                 + 0.854598 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.36717 w_k1_ADM_s113
                                 + 1.27344 w_k1_AXP_s113
                                 + 1.75055 w_k1_DIS_s113
                                 + 1.38768 w_k1_FDX_s113
                                 + 1.54074 w_k1_IBM_s113
                                 + 1.44361 w_k1_INTC_s113
                                 + 1.40381 w_k1_KO_s113 + 1.41894 w_k1_MAT_s113
                                 + 1.34891 w_k1_MSFT_s113
                                 + 1.28417 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.11257 w_k1_ADM_s113
                                 + 1.37191 w_k1_AXP_s113
                                 + 0.969046 w_k1_DIS_s113
                                 + 1.17836 w_k1_FDX_s113
                                 + 1.02229 w_k1_IBM_s113
                                 + 1.03913 w_k1_INTC_s113
                                 + 0.761179 w_k1_KO_s113
                                 + 0.80126 w_k1_MAT_s113
                                 + 0.840825 w_k1_MSFT_s113
                                 + 1.04263 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.17979 w_k1_ADM_s113
                                 + 1.084 w_k1_AXP_s113 + 1.19837 w_k1_DIS_s113
                                 + 1.01719 w_k1_FDX_s113
                                 + 1.66739 w_k1_IBM_s113
                                 + 0.809829 w_k1_INTC_s113
                                 + 1.22759 w_k1_KO_s113 + 1.86406 w_k1_MAT_s113
                                 + 1.09705 w_k1_MSFT_s113
                                 + 1.13616 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.09664 w_k1_ADM_s113
                                 + 0.986466 w_k1_AXP_s113
                                 + 1.02183 w_k1_DIS_s113
                                 + 0.945385 w_k1_FDX_s113
                                 + 1.19799 w_k1_IBM_s113
                                 + 1.06813 w_k1_INTC_s113
                                 + 1.00943 w_k1_KO_s113
                                 + 0.984962 w_k1_MAT_s113
                                 + 0.837716 w_k1_MSFT_s113
                                 + 1.04881 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.17898 w_k1_ADM_s113
                                 + 0.738601 w_k1_AXP_s113
                                 + 1.03921 w_k1_DIS_s113
                                 + 1.10213 w_k1_FDX_s113
                                 + 0.784062 w_k1_IBM_s113
                                 + 1.23701 w_k1_INTC_s113
                                 + 0.765337 w_k1_KO_s113
                                 + 0.502361 w_k1_MAT_s113
                                 + 1.08695 w_k1_MSFT_s113
                                 + 1.03631 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.20008 w_k1_ADM_s113
                                 + 1.08999 w_k1_AXP_s113 + 1.3508 w_k1_DIS_s113
                                 + 1.26948 w_k1_FDX_s113
                                 + 1.15669 w_k1_IBM_s113
                                 + 1.2909 w_k1_INTC_s113 + 1.04993 w_k1_KO_s113
                                 + 1.18016 w_k1_MAT_s113
                                 + 1.27026 w_k1_MSFT_s113
                                 + 1.16276 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.896527 w_k1_ADM_s113
                                 + 0.771702 w_k1_AXP_s113
                                 + 0.761069 w_k1_DIS_s113
                                 + 0.902984 w_k1_FDX_s113
                                 + 0.84224 w_k1_IBM_s113
                                 + 0.89796 w_k1_INTC_s113
                                 + 0.748905 w_k1_KO_s113
                                 + 0.566931 w_k1_MAT_s113
                                 + 0.88206 w_k1_MSFT_s113
                                 + 0.911127 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.23984 w_k1_ADM_s114
                                 + 1.1391 w_k1_AXP_s114 + 1.36738 w_k1_DIS_s114
                                 + 1.22685 w_k1_FDX_s114
                                 + 0.838779 w_k1_IBM_s114
                                 + 0.972328 w_k1_INTC_s114
                                 + 1.04667 w_k1_KO_s114
                                 + 0.955465 w_k1_MAT_s114
                                 + 1.21044 w_k1_MSFT_s114
                                 + 1.01515 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.06152 w_k1_ADM_s114
                                 + 1.40696 w_k1_AXP_s114
                                 + 1.34845 w_k1_DIS_s114
                                 + 1.18962 w_k1_FDX_s114
                                 + 1.72005 w_k1_IBM_s114
                                 + 0.98534 w_k1_INTC_s114
                                 + 1.26348 w_k1_KO_s114 + 1.93097 w_k1_MAT_s114
                                 + 1.14437 w_k1_MSFT_s114
                                 + 1.08816 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.794243 w_k1_ADM_s114
                                 + 0.910638 w_k1_AXP_s114
                                 + 0.868148 w_k1_DIS_s114
                                 + 0.867084 w_k1_FDX_s114
                                 + 1.07273 w_k1_IBM_s114
                                 + 1.04163 w_k1_INTC_s114
                                 + 1.03103 w_k1_KO_s114
                                 + 0.929626 w_k1_MAT_s114
                                 + 0.949311 w_k1_MSFT_s114
                                 + 0.88119 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.04483 w_k1_ADM_s114
                                 + 1.23659 w_k1_AXP_s114 + 1.0946 w_k1_DIS_s114
                                 + 1.07021 w_k1_FDX_s114
                                 + 1.26724 w_k1_IBM_s114
                                 + 1.02214 w_k1_INTC_s114
                                 + 1.07117 w_k1_KO_s114 + 1.59056 w_k1_MAT_s114
                                 + 1.1277 w_k1_MSFT_s114
                                 + 0.921202 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.914606 w_k1_ADM_s114
                                 + 0.813123 w_k1_AXP_s114
                                 + 0.726349 w_k1_DIS_s114
                                 + 0.884966 w_k1_FDX_s114
                                 + 0.685716 w_k1_IBM_s114
                                 + 0.944509 w_k1_INTC_s114
                                 + 0.778248 w_k1_KO_s114
                                 + 0.641234 w_k1_MAT_s114
                                 + 0.980071 w_k1_MSFT_s114
                                 + 0.935086 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.807054 w_k1_ADM_s114
                                 + 0.821433 w_k1_AXP_s114
                                 + 0.566112 w_k1_DIS_s114
                                 + 0.735219 w_k1_FDX_s114
                                 + 0.963118 w_k1_IBM_s114
                                 + 0.804714 w_k1_INTC_s114
                                 + 0.767343 w_k1_KO_s114
                                 + 0.993957 w_k1_MAT_s114
                                 + 0.747873 w_k1_MSFT_s114
                                 + 0.840255 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.32664 w_k1_ADM_s114
                                 + 1.61731 w_k1_AXP_s114
                                 + 1.56443 w_k1_DIS_s114
                                 + 1.46389 w_k1_FDX_s114
                                 + 1.14541 w_k1_IBM_s114
                                 + 1.37572 w_k1_INTC_s114
                                 + 1.22462 w_k1_KO_s114 + 1.36722 w_k1_MAT_s114
                                 + 1.26131 w_k1_MSFT_s114
                                 + 1.27884 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.13961 w_k1_ADM_s114
                                 + 1.29274 w_k1_AXP_s114
                                 + 0.859116 w_k1_DIS_s114
                                 + 1.09839 w_k1_FDX_s114
                                 + 0.94205 w_k1_IBM_s114
                                 + 1.07998 w_k1_INTC_s114
                                 + 0.843868 w_k1_KO_s114
                                 + 1.15493 w_k1_MAT_s114
                                 + 0.983047 w_k1_MSFT_s114
                                 + 1.05838 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.733895 w_k1_ADM_s115
                                 + 0.883602 w_k1_AXP_s115
                                 + 0.886621 w_k1_DIS_s115
                                 + 0.830579 w_k1_FDX_s115
                                 + 1.07259 w_k1_IBM_s115
                                 + 0.751832 w_k1_INTC_s115
                                 + 1.06669 w_k1_KO_s115 + 1.08599 w_k1_MAT_s115
                                 + 0.930641 w_k1_MSFT_s115
                                 + 0.9492 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.1801 w_k1_ADM_s115
                                 + 1.17641 w_k1_AXP_s115
                                 + 1.42239 w_k1_DIS_s115
                                 + 1.26793 w_k1_FDX_s115
                                 + 0.898524 w_k1_IBM_s115
                                 + 1.18749 w_k1_INTC_s115
                                 + 1.30821 w_k1_KO_s115
                                 + 0.932054 w_k1_MAT_s115
                                 + 1.25309 w_k1_MSFT_s115
                                 + 1.06557 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.03462 w_k1_ADM_s115
                                 + 1.15833 w_k1_AXP_s115
                                 + 1.40179 w_k1_DIS_s115
                                 + 1.02403 w_k1_FDX_s115
                                 + 1.59955 w_k1_IBM_s115
                                 + 1.16587 w_k1_INTC_s115
                                 + 1.52138 w_k1_KO_s115 + 1.37167 w_k1_MAT_s115
                                 + 1.14704 w_k1_MSFT_s115
                                 + 1.11303 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.934169 w_k1_ADM_s115
                                 + 0.892602 w_k1_AXP_s115
                                 + 1.14022 w_k1_DIS_s115
                                 + 1.04848 w_k1_FDX_s115
                                 + 0.825579 w_k1_IBM_s115
                                 + 0.871614 w_k1_INTC_s115
                                 + 1.0801 w_k1_KO_s115 + 0.790268 w_k1_MAT_s115
                                 + 1.09062 w_k1_MSFT_s115
                                 + 0.954154 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.96511 w_k1_ADM_s115
                                 + 1.04872 w_k1_AXP_s115
                                 + 0.859572 w_k1_DIS_s115
                                 + 1.05966 w_k1_FDX_s115
                                 + 1.04833 w_k1_IBM_s115
                                 + 0.919132 w_k1_INTC_s115
                                 + 0.919048 w_k1_KO_s115
                                 + 1.00551 w_k1_MAT_s115
                                 + 0.935227 w_k1_MSFT_s115
                                 + 1.00565 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.16981 w_k1_ADM_s115
                                 + 1.42005 w_k1_AXP_s115
                                 + 1.95828 w_k1_DIS_s115
                                 + 1.41478 w_k1_FDX_s115
                                 + 1.40376 w_k1_IBM_s115
                                 + 1.2885 w_k1_INTC_s115 + 1.68944 w_k1_KO_s115
                                 + 1.1034 w_k1_MAT_s115
                                 + 1.27194 w_k1_MSFT_s115
                                 + 1.31564 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.729265 w_k1_ADM_s115
                                 + 0.864534 w_k1_AXP_s115
                                 + 0.705359 w_k1_DIS_s115
                                 + 0.904727 w_k1_FDX_s115
                                 + 0.675457 w_k1_IBM_s115
                                 + 0.755087 w_k1_INTC_s115
                                 + 0.770938 w_k1_KO_s115
                                 + 0.742956 w_k1_MAT_s115
                                 + 0.908048 w_k1_MSFT_s115
                                 + 0.849961 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.04679 w_k1_ADM_s115
                                 + 1.56331 w_k1_AXP_s115
                                 + 1.27585 w_k1_DIS_s115
                                 + 1.20765 w_k1_FDX_s115
                                 + 1.21966 w_k1_IBM_s115
                                 + 1.02342 w_k1_INTC_s115
                                 + 1.33348 w_k1_KO_s115 + 1.5257 w_k1_MAT_s115
                                 + 1.20321 w_k1_MSFT_s115
                                 + 1.11565 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.15572 w_k1_ADM_s116
                                 + 1.38652 w_k1_AXP_s116
                                 + 1.14413 w_k1_DIS_s116
                                 + 1.21021 w_k1_FDX_s116 + 1.0464 w_k1_IBM_s116
                                 + 1.04165 w_k1_INTC_s116
                                 + 0.963721 w_k1_KO_s116
                                 + 1.35211 w_k1_MAT_s116
                                 + 1.09253 w_k1_MSFT_s116
                                 + 0.972243 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.18076 w_k1_ADM_s116
                                 + 1.44388 w_k1_AXP_s116
                                 + 1.51129 w_k1_DIS_s116
                                 + 1.20833 w_k1_FDX_s116
                                 + 1.61804 w_k1_IBM_s116
                                 + 1.2797 w_k1_INTC_s116 + 1.42482 w_k1_KO_s116
                                 + 1.65852 w_k1_MAT_s116
                                 + 1.10921 w_k1_MSFT_s116
                                 + 1.33352 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.764587 w_k1_ADM_s116
                                 + 0.813141 w_k1_AXP_s116
                                 + 0.691121 w_k1_DIS_s116
                                 + 0.744809 w_k1_FDX_s116
                                 + 0.821412 w_k1_IBM_s116
                                 + 0.737888 w_k1_INTC_s116
                                 + 0.756809 w_k1_KO_s116
                                 + 0.762713 w_k1_MAT_s116
                                 + 0.791323 w_k1_MSFT_s116
                                 + 0.860578 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.961929 w_k1_ADM_s116
                                 + 1.07599 w_k1_AXP_s116
                                 + 1.04915 w_k1_DIS_s116
                                 + 0.94911 w_k1_FDX_s116
                                 + 1.48697 w_k1_IBM_s116
                                 + 1.09526 w_k1_INTC_s116
                                 + 1.11222 w_k1_KO_s116 + 1.22275 w_k1_MAT_s116
                                 + 1.07797 w_k1_MSFT_s116
                                 + 0.915267 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.00192 w_k1_ADM_s116
                                 + 1.17419 w_k1_AXP_s116
                                 + 1.78372 w_k1_DIS_s116
                                 + 1.33184 w_k1_FDX_s116 + 1.1439 w_k1_IBM_s116
                                 + 1.39435 w_k1_INTC_s116
                                 + 1.20382 w_k1_KO_s116
                                 + 0.944758 w_k1_MAT_s116
                                 + 1.42929 w_k1_MSFT_s116
                                 + 1.12852 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.861182 w_k1_ADM_s116
                                 + 0.988517 w_k1_AXP_s116
                                 + 0.937218 w_k1_DIS_s116
                                 + 0.905883 w_k1_FDX_s116
                                 + 0.970976 w_k1_IBM_s116
                                 + 0.832937 w_k1_INTC_s116
                                 + 0.837486 w_k1_KO_s116
                                 + 1.35803 w_k1_MAT_s116
                                 + 0.983419 w_k1_MSFT_s116
                                 + 0.967822 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.09154 w_k1_ADM_s116
                                 + 1.17342 w_k1_AXP_s116
                                 + 1.26638 w_k1_DIS_s116 + 1.1981 w_k1_FDX_s116
                                 + 0.762087 w_k1_IBM_s116
                                 + 1.12475 w_k1_INTC_s116
                                 + 0.891819 w_k1_KO_s116
                                 + 0.711989 w_k1_MAT_s116
                                 + 1.14991 w_k1_MSFT_s116
                                 + 1.06676 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.04845 w_k1_ADM_s116
                                 + 1.01963 w_k1_AXP_s116
                                 + 1.20784 w_k1_DIS_s116
                                 + 1.03051 w_k1_FDX_s116
                                 + 1.04734 w_k1_IBM_s116
                                 + 1.09343 w_k1_INTC_s116
                                 + 1.12309 w_k1_KO_s116 + 1.04724 w_k1_MAT_s116
                                 + 1.13844 w_k1_MSFT_s116
                                 + 0.993829 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.02929 w_k1_ADM_s117
                                 + 0.81797 w_k1_AXP_s117
                                 + 1.10356 w_k1_DIS_s117
                                 + 0.929178 w_k1_FDX_s117
                                 + 1.59424 w_k1_IBM_s117
                                 + 0.794804 w_k1_INTC_s117
                                 + 1.27562 w_k1_KO_s117 + 1.37689 w_k1_MAT_s117
                                 + 0.943553 w_k1_MSFT_s117
                                 + 1.22163 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.858091 w_k1_ADM_s117
                                 + 0.622327 w_k1_AXP_s117
                                 + 0.785205 w_k1_DIS_s117
                                 + 0.752601 w_k1_FDX_s117
                                 + 0.726632 w_k1_IBM_s117
                                 + 0.880969 w_k1_INTC_s117
                                 + 0.807223 w_k1_KO_s117
                                 + 0.559733 w_k1_MAT_s117
                                 + 0.73779 w_k1_MSFT_s117
                                 + 0.859151 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.79583 w_k1_ADM_s117
                                 + 0.780528 w_k1_AXP_s117
                                 + 0.733546 w_k1_DIS_s117
                                 + 0.759642 w_k1_FDX_s117
                                 + 1.01445 w_k1_IBM_s117
                                 + 0.940026 w_k1_INTC_s117
                                 + 0.964612 w_k1_KO_s117
                                 + 0.847622 w_k1_MAT_s117
                                 + 0.734105 w_k1_MSFT_s117
                                 + 0.918475 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.03343 w_k1_ADM_s117
                                 + 0.93961 w_k1_AXP_s117
                                 + 0.949268 w_k1_DIS_s117
                                 + 1.02626 w_k1_FDX_s117
                                 + 0.854695 w_k1_IBM_s117
                                 + 1.20259 w_k1_INTC_s117 + 1.0738 w_k1_KO_s117
                                 + 0.638342 w_k1_MAT_s117
                                 + 0.937546 w_k1_MSFT_s117
                                 + 1.07824 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.10825 w_k1_ADM_s117
                                 + 1.34871 w_k1_AXP_s117
                                 + 1.50725 w_k1_DIS_s117
                                 + 1.23425 w_k1_FDX_s117
                                 + 1.50771 w_k1_IBM_s117
                                 + 1.14448 w_k1_INTC_s117
                                 + 1.41102 w_k1_KO_s117 + 2.05758 w_k1_MAT_s117
                                 + 1.09544 w_k1_MSFT_s117
                                 + 1.32578 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.18881 w_k1_ADM_s117
                                 + 1.19502 w_k1_AXP_s117
                                 + 1.54706 w_k1_DIS_s117
                                 + 1.31486 w_k1_FDX_s117
                                 + 1.03527 w_k1_IBM_s117
                                 + 1.34795 w_k1_INTC_s117
                                 + 1.27635 w_k1_KO_s117 + 1.312 w_k1_MAT_s117
                                 + 1.29608 w_k1_MSFT_s117 + 1.2365 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.219 w_k1_ADM_s117
                                 + 1.34382 w_k1_AXP_s117
                                 + 1.01643 w_k1_DIS_s117
                                 + 1.10173 w_k1_FDX_s117
                                 + 1.18161 w_k1_IBM_s117
                                 + 1.10973 w_k1_INTC_s117
                                 + 1.15916 w_k1_KO_s117 + 1.54981 w_k1_MAT_s117
                                 + 0.942804 w_k1_MSFT_s117
                                 + 1.03282 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.809215 w_k1_ADM_s117
                                 + 0.91452 w_k1_AXP_s117
                                 + 0.998487 w_k1_DIS_s117
                                 + 0.935129 w_k1_FDX_s117
                                 + 1.11901 w_k1_IBM_s117
                                 + 1.06458 w_k1_INTC_s117
                                 + 1.17489 w_k1_KO_s117 + 1.03337 w_k1_MAT_s117
                                 + 0.983824 w_k1_MSFT_s117
                                 + 1.14643 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.020408276587381 w_k1_ADM_s000
                                 + 0.0291398286506847 w_k1_ADM_s001
                                 + 0.0123483759088942 w_k1_ADM_s002
                                 + 0.0131337229783326 w_k1_ADM_s003
                                 + 0.0348205190247239 w_k1_ADM_s004
                                 + 0.0124516786002522 w_k1_ADM_s005
                                 + 0.0305897847408385 w_k1_ADM_s006
                                 + 0.0425066357775445 w_k1_ADM_s007
                                 + 0.0234379128796647 w_k1_ADM_s010
                                 + 0.0148584418833911 w_k1_ADM_s011
                                 + 0.0227768512419567 w_k1_ADM_s012
                                 + 0.0219619719528438 w_k1_ADM_s013
                                 + 0.011785423340722 w_k1_ADM_s014
                                 + 0.0281068425508994 w_k1_ADM_s015
                                 + 0.0283523891767026 w_k1_ADM_s016
                                 + 0.0240395680092261 w_k1_ADM_s017
                                 + 0.0130802653500182 w_k1_ADM_s100
                                 + 0.0149291178567532 w_k1_ADM_s101
                                 + 0.0172074821358111 w_k1_ADM_s102
                                 + 0.0279420013659569 w_k1_ADM_s103
                                 + 0.0194812300862521 w_k1_ADM_s104
                                 + 0.0103657993684883 w_k1_ADM_s105
                                 + 0.0148736704877542 w_k1_ADM_s106
                                 + 0.00922458294018432 w_k1_ADM_s107
                                 + 0.0965864878770987 w_k1_ADM_s110
                                 + 0.0404150196145064 w_k1_ADM_s111
                                 + 0.0663577673245247 w_k1_ADM_s112
                                 + 0.0292456357988421 w_k1_ADM_s113
                                 + 0.0396897314507195 w_k1_ADM_s114
                                 + 0.0495769899407056 w_k1_ADM_s115
                                 + 0.127822596087555 w_k1_ADM_s116
                                 + 0.0387221783133968 w_k1_ADM_s117
                                 + 0.0228341995667191 w_k1_AXP_s000
                                 + 0.0303515189104073 w_k1_AXP_s001
                                 + 0.0148003866243461 w_k1_AXP_s002
                                 + 0.0128274972643615 w_k1_AXP_s003
                                 + 0.0343779889501094 w_k1_AXP_s004
                                 + 0.0141251857411189 w_k1_AXP_s005
                                 + 0.0355385047190937 w_k1_AXP_s006
                                 + 0.0387026774894395 w_k1_AXP_s007
                                 + 0.023603587862186 w_k1_AXP_s010
                                 + 0.0181444769739103 w_k1_AXP_s011
                                 + 0.0240637154778842 w_k1_AXP_s012
                                 + 0.024684632426495 w_k1_AXP_s013
                                 + 0.0112547807692769 w_k1_AXP_s014
                                 + 0.0286889158033467 w_k1_AXP_s015
                                 + 0.0240828290108756 w_k1_AXP_s016
                                 + 0.0269282584711733 w_k1_AXP_s017
                                 + 0.0120136715263665 w_k1_AXP_s100
                                 + 0.0155442172547335 w_k1_AXP_s101
                                 + 0.0165950708445556 w_k1_AXP_s102
                                 + 0.0298142324498173 w_k1_AXP_s103
                                 + 0.0189826310283387 w_k1_AXP_s104
                                 + 0.0117344371373632 w_k1_AXP_s105
                                 + 0.0192079354675736 w_k1_AXP_s106
                                 + 0.0106652736801592 w_k1_AXP_s107
                                 + 0.10789910074888 w_k1_AXP_s110
                                 + 0.0353777209378015 w_k1_AXP_s111
                                 + 0.0622800340275006 w_k1_AXP_s112
                                 + 0.0265799215389565 w_k1_AXP_s113
                                 + 0.0428333164343921 w_k1_AXP_s114
                                 + 0.05712767514851 w_k1_AXP_s115
                                 + 0.14123173622466 w_k1_AXP_s116
                                 + 0.0378336487442035 w_k1_AXP_s117
                                 + 0.0248676363921599 w_k1_DIS_s000
                                 + 0.0295680104430772 w_k1_DIS_s001
                                 + 0.0129903918113575 w_k1_DIS_s002
                                 + 0.0135553428031555 w_k1_DIS_s003
                                 + 0.0350464299867497 w_k1_DIS_s004
                                 + 0.0135400140902748 w_k1_DIS_s005
                                 + 0.0372268590945547 w_k1_DIS_s006
                                 + 0.0403192982948483 w_k1_DIS_s007
                                 + 0.0246454515308074 w_k1_DIS_s010
                                 + 0.0176849625421039 w_k1_DIS_s011
                                 + 0.0251143233397941 w_k1_DIS_s012
                                 + 0.023263136071254 w_k1_DIS_s013
                                 + 0.0115539617073502 w_k1_DIS_s014
                                 + 0.0296557063403137 w_k1_DIS_s015
                                 + 0.0259197829271488 w_k1_DIS_s016
                                 + 0.0268718516626221 w_k1_DIS_s017
                                 + 0.0123968239585464 w_k1_DIS_s100
                                 + 0.0166250853170465 w_k1_DIS_s101
                                 + 0.0159393038135883 w_k1_DIS_s102
                                 + 0.0266648706700552 w_k1_DIS_s103
                                 + 0.0203415292104965 w_k1_DIS_s104
                                 + 0.0114511646990076 w_k1_DIS_s105
                                 + 0.0172415133945766 w_k1_DIS_s106
                                 + 0.0101170102030813 w_k1_DIS_s107
                                 + 0.106147187748224 w_k1_DIS_s110
                                 + 0.0367120820055936 w_k1_DIS_s111
                                 + 0.0684098029592438 w_k1_DIS_s112
                                 + 0.0281891902844711 w_k1_DIS_s113
                                 + 0.0381769259305557 w_k1_DIS_s114
                                 + 0.0599440323198655 w_k1_DIS_s115
                                 + 0.142927313601305 w_k1_DIS_s116
                                 + 0.0417824145552957 w_k1_DIS_s117
                                 + 0.0225886303460345 w_k1_FDX_s000
                                 + 0.030587442393273 w_k1_FDX_s001
                                 + 0.0135458755497045 w_k1_FDX_s002
                                 + 0.0127924660362428 w_k1_FDX_s003
                                 + 0.0334450969920596 w_k1_FDX_s004
                                 + 0.0144452033230542 w_k1_FDX_s005
                                 + 0.0350978824047186 w_k1_FDX_s006
                                 + 0.0380227730647109 w_k1_FDX_s007
                                 + 0.023493247927816 w_k1_FDX_s010
                                 + 0.0181109272747101 w_k1_FDX_s011
                                 + 0.0250210994119535 w_k1_FDX_s012
                                 + 0.0220405962771224 w_k1_FDX_s013
                                 + 0.0107619268695824 w_k1_FDX_s014
                                 + 0.0300178400939163 w_k1_FDX_s015
                                 + 0.0252851569207407 w_k1_FDX_s016
                                 + 0.0251886529883641 w_k1_FDX_s017
                                 + 0.0115716522004484 w_k1_FDX_s100
                                 + 0.0157933669571221 w_k1_FDX_s101
                                 + 0.0164836569753582 w_k1_FDX_s102
                                 + 0.0293329580169094 w_k1_FDX_s103
                                 + 0.0200042420482964 w_k1_FDX_s104
                                 + 0.0109200552603599 w_k1_FDX_s105
                                 + 0.016894611970849 w_k1_FDX_s106
                                 + 0.0101269144835166 w_k1_FDX_s107
                                 + 0.105196646310663 w_k1_FDX_s110
                                 + 0.0342105086645369 w_k1_FDX_s111
                                 + 0.0619284692352737 w_k1_FDX_s112
                                 + 0.0278998737502938 w_k1_FDX_s113
                                 + 0.0402598326369088 w_k1_FDX_s114
                                 + 0.0563231437131434 w_k1_FDX_s115
                                 + 0.132956472758277 w_k1_FDX_s116
                                 + 0.0393614808938643 w_k1_FDX_s117
                                 + 0.0230499557586847 w_k1_IBM_s000
                                 + 0.0329249167673925 w_k1_IBM_s001
                                 + 0.0140697654738063 w_k1_IBM_s002
                                 + 0.0134435948694346 w_k1_IBM_s003
                                 + 0.0323511075571609 w_k1_IBM_s004
                                 + 0.0146530495091354 w_k1_IBM_s005
                                 + 0.0355807298896705 w_k1_IBM_s006
                                 + 0.0381924962531134 w_k1_IBM_s007
                                 + 0.0237309398364107 w_k1_IBM_s010
                                 + 0.0184058440400744 w_k1_IBM_s011
                                 + 0.0246126019142769 w_k1_IBM_s012
                                 + 0.022034618154569 w_k1_IBM_s013
                                 + 0.0112306231025915 w_k1_IBM_s014
                                 + 0.0303058034229025 w_k1_IBM_s015
                                 + 0.0266423008059037 w_k1_IBM_s016
                                 + 0.0262351556156808 w_k1_IBM_s017
                                 + 0.011999139666804 w_k1_IBM_s100
                                 + 0.0165677662229491 w_k1_IBM_s101
                                 + 0.0169705356995655 w_k1_IBM_s102
                                 + 0.030075408538796 w_k1_IBM_s103
                                 + 0.0203010364968359 w_k1_IBM_s104
                                 + 0.0108733732088801 w_k1_IBM_s105
                                 + 0.0164071243253757 w_k1_IBM_s106
                                 + 0.00976233692635887 w_k1_IBM_s107
                                 + 0.105271184262584 w_k1_IBM_s110
                                 + 0.0365534082855208 w_k1_IBM_s111
                                 + 0.0637975292240919 w_k1_IBM_s112
                                 + 0.0314030670499033 w_k1_IBM_s113
                                 + 0.0415924287797242 w_k1_IBM_s114
                                 + 0.0570380419325513 w_k1_IBM_s115
                                 + 0.137871589571872 w_k1_IBM_s116
                                 + 0.0412177670237372 w_k1_IBM_s117
                                 + 0.0219832514904224 w_k1_INTC_s000
                                 + 0.0291471302149925 w_k1_INTC_s001
                                 + 0.0137330921345617 w_k1_INTC_s002
                                 + 0.0125145935956879 w_k1_INTC_s003
                                 + 0.0397500018603117 w_k1_INTC_s004
                                 + 0.0137448001471722 w_k1_INTC_s005
                                 + 0.0368101447030301 w_k1_INTC_s006
                                 + 0.0420928604935658 w_k1_INTC_s007
                                 + 0.0220509288429449 w_k1_INTC_s010
                                 + 0.0197281600872898 w_k1_INTC_s011
                                 + 0.0249912236227838 w_k1_INTC_s012
                                 + 0.0221013465997042 w_k1_INTC_s013
                                 + 0.0112440983885052 w_k1_INTC_s014
                                 + 0.0277586928965029 w_k1_INTC_s015
                                 + 0.0241999035464229 w_k1_INTC_s016
                                 + 0.0279305871924605 w_k1_INTC_s017
                                 + 0.0135924738156868 w_k1_INTC_s100
                                 + 0.016887525659543 w_k1_INTC_s101
                                 + 0.0133493153855834 w_k1_INTC_s102
                                 + 0.0308196751474027 w_k1_INTC_s103
                                 + 0.0214839293100649 w_k1_INTC_s104
                                 + 0.0130518964434377 w_k1_INTC_s105
                                 + 0.0194010174274261 w_k1_INTC_s106
                                 + 0.00951597022952126 w_k1_INTC_s107
                                 + 0.0897999587951949 w_k1_INTC_s110
                                 + 0.0310780212845276 w_k1_INTC_s111
                                 + 0.0648636481419121 w_k1_INTC_s112
                                 + 0.0281809747233173 w_k1_INTC_s113
                                 + 0.0404597077067824 w_k1_INTC_s114
                                 + 0.0506417584137848 w_k1_INTC_s115
                                 + 0.131560623599714 w_k1_INTC_s116
                                 + 0.0429044968463783 w_k1_INTC_s117
                                 + 0.0245225994124831 w_k1_KO_s000
                                 + 0.0315551577602381 w_k1_KO_s001
                                 + 0.0129750348660171 w_k1_KO_s002
                                 + 0.0132348291524038 w_k1_KO_s003
                                 + 0.0348451374472851 w_k1_KO_s004
                                 + 0.0136994599576802 w_k1_KO_s005
                                 + 0.0411671677280437 w_k1_KO_s006
                                 + 0.0419353849254262 w_k1_KO_s007
                                 + 0.0217017806618671 w_k1_KO_s010
                                 + 0.0169917195323186 w_k1_KO_s011
                                 + 0.0245152768514757 w_k1_KO_s012
                                 + 0.0212507085454464 w_k1_KO_s013
                                 + 0.010775570856651 w_k1_KO_s014
                                 + 0.0295556845215427 w_k1_KO_s015
                                 + 0.0246372142555412 w_k1_KO_s016
                                 + 0.0281078488528165 w_k1_KO_s017
                                 + 0.0120528621467635 w_k1_KO_s100
                                 + 0.0163831281481621 w_k1_KO_s101
                                 + 0.0171161864212552 w_k1_KO_s102
                                 + 0.0317024385657022 w_k1_KO_s103
                                 + 0.0205431724585418 w_k1_KO_s104
                                 + 0.0106926625079572 w_k1_KO_s105
                                 + 0.0170424293655979 w_k1_KO_s106
                                 + 0.0102283957765756 w_k1_KO_s107
                                 + 0.0940670765732243 w_k1_KO_s110
                                 + 0.0351506913862469 w_k1_KO_s111
                                 + 0.0556978843772418 w_k1_KO_s112
                                 + 0.026068649248468 w_k1_KO_s113
                                 + 0.0388082977317754 w_k1_KO_s114
                                 + 0.0621565865700256 w_k1_KO_s115
                                 + 0.128568345989953 w_k1_KO_s116
                                 + 0.0438920149571339 w_k1_KO_s117
                                 + 0.0211854786308186 w_k1_MAT_s000
                                 + 0.0394985403976217 w_k1_MAT_s001
                                 + 0.0149116178374709 w_k1_MAT_s002
                                 + 0.0110082648445766 w_k1_MAT_s003
                                 + 0.0332410100130446 w_k1_MAT_s004
                                 + 0.0120741528607444 w_k1_MAT_s005
                                 + 0.039161910059202 w_k1_MAT_s006
                                 + 0.0326369570641354 w_k1_MAT_s007
                                 + 0.0264330598598626 w_k1_MAT_s010
                                 + 0.0209008941889442 w_k1_MAT_s011
                                 + 0.0327581005323726 w_k1_MAT_s012
                                 + 0.0220620747483365 w_k1_MAT_s013
                                 + 0.0115456318212544 w_k1_MAT_s014
                                 + 0.0315071194357393 w_k1_MAT_s015
                                 + 0.022062346273212 w_k1_MAT_s016
                                 + 0.0290070527854229 w_k1_MAT_s017
                                 + 0.0139944218912561 w_k1_MAT_s100
                                 + 0.0145719818067422 w_k1_MAT_s101
                                 + 0.0186874833072657 w_k1_MAT_s102
                                 + 0.0301176507337085 w_k1_MAT_s103
                                 + 0.0244846534687337 w_k1_MAT_s104
                                 + 0.0106043230148882 w_k1_MAT_s105
                                 + 0.0147847841961793 w_k1_MAT_s106
                                 + 0.0106236116088068 w_k1_MAT_s107
                                 + 0.1092307740459 w_k1_MAT_s110
                                 + 0.0361054735110171 w_k1_MAT_s111
                                 + 0.0528506122514081 w_k1_MAT_s112
                                 + 0.0277132378702388 w_k1_MAT_s113
                                 + 0.0442343348461052 w_k1_MAT_s114
                                 + 0.0561060117899907 w_k1_MAT_s115
                                 + 0.146229787019093 w_k1_MAT_s116
                                 + 0.0410453380388123 w_k1_MAT_s117
                                 + 0.0208249860015162 w_k1_MSFT_s000
                                 + 0.0285736534228721 w_k1_MSFT_s001
                                 + 0.0135565002384708 w_k1_MSFT_s002
                                 + 0.0125559528861713 w_k1_MSFT_s003
                                 + 0.0333822696409012 w_k1_MSFT_s004
                                 + 0.0127657102576365 w_k1_MSFT_s005
                                 + 0.034962903223582 w_k1_MSFT_s006
                                 + 0.0402759522508624 w_k1_MSFT_s007
                                 + 0.0235098619314804 w_k1_MSFT_s010
                                 + 0.0188832953553235 w_k1_MSFT_s011
                                 + 0.0256715274906826 w_k1_MSFT_s012
                                 + 0.0191738387288582 w_k1_MSFT_s013
                                 + 0.0102914240672496 w_k1_MSFT_s014
                                 + 0.0301038801534328 w_k1_MSFT_s015
                                 + 0.0241355581573014 w_k1_MSFT_s016
                                 + 0.0280524285539027 w_k1_MSFT_s017
                                 + 0.013875543463056 w_k1_MSFT_s100
                                 + 0.0163011725582035 w_k1_MSFT_s101
                                 + 0.0168461455509343 w_k1_MSFT_s102
                                 + 0.0283820838235747 w_k1_MSFT_s103
                                 + 0.0193318415201539 w_k1_MSFT_s104
                                 + 0.0105959950132246 w_k1_MSFT_s105
                                 + 0.0170011573253556 w_k1_MSFT_s106
                                 + 0.00958186817935531 w_k1_MSFT_s107
                                 + 0.098531379683324 w_k1_MSFT_s110
                                 + 0.0339645501523618 w_k1_MSFT_s111
                                 + 0.0604059612358657 w_k1_MSFT_s112
                                 + 0.0263534981308706 w_k1_MSFT_s113
                                 + 0.0410020571318222 w_k1_MSFT_s114
                                 + 0.0573405317323699 w_k1_MSFT_s115
                                 + 0.138383001126433 w_k1_MSFT_s116
                                 + 0.037766419738512 w_k1_MSFT_s117
                                 + 0.0244403717473408 w_k1_PG_s000
                                 + 0.0306539230027618 w_k1_PG_s001
                                 + 0.0127524982217949 w_k1_PG_s002
                                 + 0.0144741103726477 w_k1_PG_s003
                                 + 0.0335787067778224 w_k1_PG_s004
                                 + 0.0138909522633083 w_k1_PG_s005
                                 + 0.0346301337728244 w_k1_PG_s006
                                 + 0.0386597302728812 w_k1_PG_s007
                                 + 0.0211504860269472 w_k1_PG_s010
                                 + 0.0170182459991317 w_k1_PG_s011
                                 + 0.0247821634647914 w_k1_PG_s012
                                 + 0.0229316480827349 w_k1_PG_s013
                                 + 0.011106421875138 w_k1_PG_s014
                                 + 0.0307978059803062 w_k1_PG_s015
                                 + 0.0239720144292627 w_k1_PG_s016
                                 + 0.0255129360945727 w_k1_PG_s017
                                 + 0.0118747347483123 w_k1_PG_s100
                                 + 0.0164637946345175 w_k1_PG_s101
                                 + 0.0160692349649549 w_k1_PG_s102
                                 + 0.0279768707292833 w_k1_PG_s103
                                 + 0.0194598629411103 w_k1_PG_s104
                                 + 0.0118371039364002 w_k1_PG_s105
                                 + 0.0161737789383594 w_k1_PG_s106
                                 + 0.0107996452363142 w_k1_PG_s107
                                 + 0.0960323055951454 w_k1_PG_s110
                                 + 0.0340171862123046 w_k1_PG_s111
                                 + 0.0578430529528241 w_k1_PG_s112
                                 + 0.0287493450370543 w_k1_PG_s113
                                 + 0.0391468049251831 w_k1_PG_s114
                                 + 0.0552762742849896 w_k1_PG_s115
                                 + 0.131957130235306 w_k1_PG_s116
                                 + 0.0428469869796292 w_k1_PG_s117 >= 1.229
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
