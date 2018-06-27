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
 w_k1_ADM_s0_return_constraint:  1.13602 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.8726 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.08811 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.970657 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.903833 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.13721 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.02576 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.920363 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.9628 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.12833 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.06203 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.845323 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.70579 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.822275 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.44936 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.16216 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.782227 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.11204 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.39829 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.8616 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.923995 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.00112 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.00785 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.07426 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.921223 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.25348 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.796675 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.747555 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.985096 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.13612 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.29783 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.859682 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.04855 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.14451 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.957184 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.25513 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.948369 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.830245 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23449 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.908656 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.16325 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.03181 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.83286 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23141 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.70506 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.880753 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.936517 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.29437 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.05772 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.570075 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.54289 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.681904 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.62843 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.13935 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.678827 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.10871 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.24675 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.872296 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.723497 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.07964 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.820141 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.02368 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.672087 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.19826 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.649438 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.886195 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.679226 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.21689 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.32248 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.862563 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.09873 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.35614 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.727223 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.60144 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.809771 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.642528 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30552 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.865252 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.32949 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.873843 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.77247 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.29934 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.08251 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.868468 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.92447 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.21038 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.22118 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.615479 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.77182 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.654869 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.66018 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.995652 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.665702 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.17644 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.45883 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.939687 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.753449 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.09596 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.851438 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.01087 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.651244 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.75083 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.565913 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.830371 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.08417 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.26124 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.49505 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.800417 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.16399 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.48276 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.813672 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.22338 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.831698 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.569818 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18402 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.899216 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.13596 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.927844 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.871549 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.18372 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.00086 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.903986 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.945774 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.17395 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.965181 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.70337 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.49691 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.865552 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.33283 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.20809 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.729463 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.11367 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.38301 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.85888 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.885716 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.997268 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.05103 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.09178 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.732868 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.45433 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.760053 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.809116 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.982961 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.17519 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.4327 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.830252 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.06479 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.13137 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.85737 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.25517 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.957123 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.712301 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20223 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.910452 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.12321 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.02482 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.858155 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.20875 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.739697 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.13368 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.860882 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.967553 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.6722 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.757124 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.1007 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.477291 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.86392 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.702326 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.892629 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.06026 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.685941 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.74459 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.597887 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.35984 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.531268 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.851468 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.854444 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.920121 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.558111 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.36582 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.844231 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.33929 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.838877 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.27828 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.07109 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.88445 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.897165 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.08925 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.600287 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.683273 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15047 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.941202 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.02359 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.985031 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.870107 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1879 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.19543 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.895434 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.12281 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.18211 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.01868 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.740994 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.42508 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.00866 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.27093 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.0338 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.773184 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.978911 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.02457 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.88008 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.932732 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.99328 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.832164 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.950046 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.800653 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.32604 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.698416 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.92615 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.10408 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.14951 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.12321 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.830964 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.935723 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.13768 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.808889 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.26552 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.98532 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.71849 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19744 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.892369 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.32197 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.959751 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.780053 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.22722 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.758743 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.942106 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.850335 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.996124 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.24094 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.693854 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.12394 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.559067 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.56401 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.87018 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.817669 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.08073 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.1052 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.13289 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.705773 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.13043 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.773835 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.924223 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.879355 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.33847 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.731938 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.12499 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.16891 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.51587 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.07645 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.993622 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.08365 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.59962 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.940191 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.12641 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.760541 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.613617 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.3149 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.930487 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.12396 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.843271 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.663299 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.21024 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.443382 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.07001 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.713745 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.01475 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.56365 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.685687 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.992258 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.392019 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 2.3432 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.879281 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.871861 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.11048 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.25294 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.88425 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.731922 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.88866 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.719158 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.30744 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.08532 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.26288 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.688431 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.59318 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.87856 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.74644 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.910282 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.15924 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.35793 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.64546 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.715993 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.36848 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.491577 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.551655 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.19344 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.904948 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.12554 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.891993 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.870196 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.19306 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.0244 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.892683 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.00251 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.11831 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.03419 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.778638 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.37608 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.852383 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.30794 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.11145 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.762698 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.99328 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.31155 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.842833 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.887029 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.06014 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.978146 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.06053 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.82389 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.46063 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.732765 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.928475 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.04033 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.26995 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.20703 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.830949 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.09289 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.15601 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.867433 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.14498 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.940989 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.716835 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12229 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.933526 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.08988 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.891902 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.892037 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.15766 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.887131 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.00489 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.994663 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.11222 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.07456 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.772688 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.17124 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.748002 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.35917 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.11396 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.899145 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.12328 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.18678 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.05729 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.955431 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.1769 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.953973 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.998057 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.804825 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.1893 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.763453 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.01233 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.979959 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.17549 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.08282 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.946476 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.00292 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.1326 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.926183 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.05319 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.861039 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.761479 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.261000249524569 z_k1_s0000
                                 - 0.201000700599643 z_k1_s0001
                                 - 0.0840000189952319 z_k1_s0002
                                 - 0.0690000016639333 z_k1_s0003
                                 - 0.245000543278011 z_k1_s0004
                                 - 0.164000241528892 z_k1_s0005
                                 - 0.176999840229136 z_k1_s0006
                                 - 0.100000115541991 z_k1_s0007
                                 - 0.166000536125094 z_k1_s0010
                                 - 0.361002024322842 z_k1_s0011
                                 - 0.0980002624158552 z_k1_s0012
                                 - 0.122000370501504 z_k1_s0013
                                 - 0.103000325600489 z_k1_s0014
                                 - 0.377000664908927 z_k1_s0015
                                 - 0.447001549493798 z_k1_s0016
                                 - 0.278000780257331 z_k1_s0017
                                 - 0.320000016682941 z_k1_s0020
                                 - 0.108000187630684 z_k1_s0021
                                 - 0.20699973779149 z_k1_s0022
                                 - 0.178000191280077 z_k1_s0023
                                 - 0.222000193573982 z_k1_s0024
                                 - 0.335000472465433 z_k1_s0025
                                 - 0.277999832493114 z_k1_s0026
                                 - 0.172000190967272 z_k1_s0027
                                 - 0.251998478915024 z_k1_s0030
                                 - 0.390000249269108 z_k1_s0031
                                 - 0.177999769692602 z_k1_s0032
                                 - 0.368000579284149 z_k1_s0033
                                 - 0.188999604685082 z_k1_s0034
                                 - 0.194999627635345 z_k1_s0035
                                 - 0.264999978221621 z_k1_s0036
                                 - 0.647999993632418 z_k1_s0037
                                 - 0.237000434159869 z_k1_s0040
                                 - 0.141000065447035 z_k1_s0041
                                 - 0.180000056536644 z_k1_s0042
                                 - 0.432000389607913 z_k1_s0043
                                 - 0.286999391864626 z_k1_s0044
                                 - 0.141000065447035 z_k1_s0045
                                 - 0.308000689220405 z_k1_s0046
                                 - 0.389999910896088 z_k1_s0047
                                 - 0.165000117447376 z_k1_s0050
                                 - 0.0970002034996334 z_k1_s0051
                                 - 0.233001102397244 z_k1_s0052
                                 - 0.161000689510399 z_k1_s0053
                                 - 0.108000160329357 z_k1_s0054
                                 - 0.412000747627077 z_k1_s0055
                                 - 0.175000217292856 z_k1_s0056
                                 - 0.178999645229833 z_k1_s0057
                                 - 0.12100028644166 z_k1_s0060
                                 - 0.1290003383321 z_k1_s0061
                                 - 0.358000186091863 z_k1_s0062
                                 - 0.146000234998753 z_k1_s0063
                                 - 0.108000273319873 z_k1_s0064
                                 - 0.0900001565663838 z_k1_s0065
                                 - 0.136000312536058 z_k1_s0066
                                 - 0.336000755394925 z_k1_s0067
                                 - 0.165000315178729 z_k1_s0070
                                 - 0.0770001125167121 z_k1_s0071
                                 - 0.122000038210781 z_k1_s0072
                                 - 0.0700000457242377 z_k1_s0073
                                 - 0.0800000522562717 z_k1_s0074
                                 - 0.112999907891849 z_k1_s0075
                                 - 0.116999620144426 z_k1_s0076
                                 - 0.293000440268797 z_k1_s0077
                                 - 0.10699986500794 z_k1_s0100
                                 - 0.320999285423999 z_k1_s0101
                                 - 0.129999880839527 z_k1_s0102
                                 - 0.081999989138741 z_k1_s0103
                                 - 0.171999728027748 z_k1_s0104
                                 - 0.279000521835148 z_k1_s0105
                                 - 0.292999077698698 z_k1_s0106
                                 - 0.163999668677662 z_k1_s0107
                                 - 0.468998635380849 z_k1_s0110
                                 - 0.125999878275358 z_k1_s0111
                                 - 0.400999103568596 z_k1_s0112
                                 - 0.411999649332104 z_k1_s0113
                                 - 0.287001131471105 z_k1_s0114
                                 - 0.288000503086231 z_k1_s0115
                                 - 0.29100110393065 z_k1_s0116
                                 - 0.212000031955272 z_k1_s0117
                                 - 1.2349994030096 z_k1_s0120
                                 - 0.622999444691381 z_k1_s0121
                                 - 0.301000171271291 z_k1_s0122
                                 - 0.330999928398559 z_k1_s0123
                                 - 0.351000069750192 z_k1_s0124
                                 - 0.924002345363757 z_k1_s0125
                                 - 0.385000219067931 z_k1_s0126
                                 - 0.399000227034037 z_k1_s0127
                                 - 0.484000509340134 z_k1_s0130
                                 - 0.658998185985908 z_k1_s0131
                                 - 0.565000900861503 z_k1_s0132
                                 - 1.24500333960276 z_k1_s0133
                                 - 0.366000429983713 z_k1_s0134
                                 - 1.40700050704083 z_k1_s0135
                                 - 0.734003904957063 z_k1_s0136
                                 - 0.566000615553984 z_k1_s0137
                                 - 0.180999884014317 z_k1_s0140
                                 - 0.149999939486423 z_k1_s0141
                                 - 0.192999862541244 z_k1_s0142
                                 - 0.513999080236704 z_k1_s0143
                                 - 0.34499938264915 z_k1_s0144
                                 - 0.279999498961629 z_k1_s0145
                                 - 0.188999869698935 z_k1_s0146
                                 - 0.226999593801035 z_k1_s0147
                                 - 0.109999938671659 z_k1_s0150
                                 - 0.222999881252546 z_k1_s0151
                                 - 0.249999916435588 z_k1_s0152
                                 - 0.181999418493124 z_k1_s0153
                                 - 0.0860000301978404 z_k1_s0154
                                 - 0.0770000184701597 z_k1_s0155
                                 - 0.0659999632029956 z_k1_s0156
                                 - 0.234000304919699 z_k1_s0157
                                 - 0.481998807045869 z_k1_s0160
                                 - 0.950997468590589 z_k1_s0161
                                 - 0.429000519540034 z_k1_s0162
                                 - 0.333999686831528 z_k1_s0163
                                 - 0.263999726824933 z_k1_s0164
                                 - 0.407999889238377 z_k1_s0165
                                 - 0.884999035583606 z_k1_s0166
                                 - 0.528998015550679 z_k1_s0167
                                 - 0.300000372755332 z_k1_s0170
                                 - 0.305000106916071 z_k1_s0171
                                 - 0.475999280958913 z_k1_s0172
                                 - 0.877999301176735 z_k1_s0173
                                 - 0.175999950104281 z_k1_s0174
                                 - 0.833999834601015 z_k1_s0175
                                 - 0.189999969814865 z_k1_s0176
                                 - 0.314000045165682 z_k1_s0177
                                 - 0.12999995596786 z_k1_s1000
                                 - 0.179000010919613 z_k1_s1001
                                 - 0.360999395154448 z_k1_s1002
                                 - 0.445000054786198 z_k1_s1003
                                 - 0.21699939521465 z_k1_s1004
                                 - 0.236999175317333 z_k1_s1005
                                 - 0.274999944912792 z_k1_s1006
                                 - 0.135000098893591 z_k1_s1007
                                 - 1.38699845470538 z_k1_s1010
                                 - 1.08800044726031 z_k1_s1011
                                 - 0.67499654617712 z_k1_s1012
                                 - 0.434999509100471 z_k1_s1013
                                 - 0.445999337156121 z_k1_s1014
                                 - 0.863996273065689 z_k1_s1015
                                 - 0.564999422218914 z_k1_s1016
                                 - 0.322999735199742 z_k1_s1017
                                 - 0.34700005314967 z_k1_s1020
                                 - 0.478001124178221 z_k1_s1021
                                 - 0.275000196375802 z_k1_s1022
                                 - 0.19700010645391 z_k1_s1023
                                 - 0.17000023368877 z_k1_s1024
                                 - 0.818001591555762 z_k1_s1025
                                 - 0.481000946651991 z_k1_s1026
                                 - 0.175000231911962 z_k1_s1027
                                 - 0.293000011103784 z_k1_s1030
                                 - 0.201000288759151 z_k1_s1031
                                 - 0.843000417459341 z_k1_s1032
                                 - 0.284999484247257 z_k1_s1033
                                 - 0.343000548954719 z_k1_s1034
                                 - 0.277999023247796 z_k1_s1035
                                 - 0.188000259118003 z_k1_s1036
                                 - 0.323999297670467 z_k1_s1037
                                 - 0.115999972097091 z_k1_s1040
                                 - 0.400999766758018 z_k1_s1041
                                 - 0.25399935995385 z_k1_s1042
                                 - 0.281000940795758 z_k1_s1043
                                 - 0.353000236373114 z_k1_s1044
                                 - 0.143000076938679 z_k1_s1045
                                 - 0.186000025241902 z_k1_s1046
                                 - 0.11100002101533 z_k1_s1047
                                 - 0.868001139482026 z_k1_s1050
                                 - 0.303000544615444 z_k1_s1051
                                 - 0.774998888642129 z_k1_s1052
                                 - 0.716998569517464 z_k1_s1053
                                 - 0.443000476967084 z_k1_s1054
                                 - 0.195000282400947 z_k1_s1055
                                 - 0.926001458606691 z_k1_s1056
                                 - 0.358000436164205 z_k1_s1057
                                 - 0.346999250439229 z_k1_s1060
                                 - 0.194000181124153 z_k1_s1061
                                 - 0.144000058338484 z_k1_s1062
                                 - 0.225999821254622 z_k1_s1063
                                 - 0.227999700893946 z_k1_s1064
                                 - 0.649001208993666 z_k1_s1065
                                 - 0.227999700893946 z_k1_s1066
                                 - 0.221000122156311 z_k1_s1067
                                 - 0.801003884882739 z_k1_s1070
                                 - 1.74000601824033 z_k1_s1071
                                 - 0.492001068517898 z_k1_s1072
                                 - 0.797999771425947 z_k1_s1073
                                 - 0.944004092438465 z_k1_s1074
                                 - 0.752005251948113 z_k1_s1075
                                 - 0.954999588391566 z_k1_s1076
                                 - 0.863003204415221 z_k1_s1077
                                 - 0.272999727045535 z_k1_s1100
                                 - 0.515999345723648 z_k1_s1101
                                 - 0.245999629514855 z_k1_s1102
                                 - 0.454001360160756 z_k1_s1103
                                 - 0.609000521061322 z_k1_s1104
                                 - 0.688999411043333 z_k1_s1105
                                 - 0.594999351291635 z_k1_s1106
                                 - 0.814997347990502 z_k1_s1107
                                 - 0.62099962422189 z_k1_s1110
                                 - 0.677000235263191 z_k1_s1111
                                 - 0.352000179114719 z_k1_s1112
                                 - 0.344999969234482 z_k1_s1113
                                 - 0.752000576836098 z_k1_s1114
                                 - 0.400000397721379 z_k1_s1115
                                 - 0.506000524477556 z_k1_s1116
                                 - 1.68700305511068 z_k1_s1117
                                 - 0.441999740875465 z_k1_s1120
                                 - 0.608999516019529 z_k1_s1121
                                 - 0.682998936963633 z_k1_s1122
                                 - 0.878995886693243 z_k1_s1123
                                 - 0.582999571515037 z_k1_s1124
                                 - 0.746002355585256 z_k1_s1125
                                 - 0.728997522819454 z_k1_s1126
                                 - 2.17199957506204 z_k1_s1127
                                 - 0.478001089321832 z_k1_s1130
                                 - 0.341000684392807 z_k1_s1131
                                 - 0.396999023109015 z_k1_s1132
                                 - 0.137999928349127 z_k1_s1133
                                 - 0.216000035202926 z_k1_s1134
                                 - 0.470000209362798 z_k1_s1135
                                 - 0.230999925401783 z_k1_s1136
                                 - 0.336000460293287 z_k1_s1137
                                 - 0.714999281290568 z_k1_s1140
                                 - 0.281999899195589 z_k1_s1141
                                 - 0.813999050023203 z_k1_s1142
                                 - 0.203999899745765 z_k1_s1143
                                 - 0.787001059458356 z_k1_s1144
                                 - 0.262999822473251 z_k1_s1145
                                 - 0.899000970832526 z_k1_s1146
                                 - 0.3179999318801 z_k1_s1147
                                 - 0.200000774863673 z_k1_s1150
                                 - 0.156000023333889 z_k1_s1151
                                 - 0.266000620408843 z_k1_s1152
                                 - 0.190999475335295 z_k1_s1153
                                 - 0.149999897548018 z_k1_s1154
                                 - 0.189999710720884 z_k1_s1155
                                 - 0.342999585711871 z_k1_s1156
                                 - 0.212999423850352 z_k1_s1157
                                 - 0.830996408244739 z_k1_s1160
                                 - 0.392998956830196 z_k1_s1161
                                 - 0.227999266301839 z_k1_s1162
                                 - 0.232999232633068 z_k1_s1163
                                 - 0.914995842609381 z_k1_s1164
                                 - 0.793998661388167 z_k1_s1165
                                 - 0.393998950096442 z_k1_s1166
                                 - 0.219999320171873 z_k1_s1167
                                 - 0.309000331057051 z_k1_s1170
                                 - 0.225999520424624 z_k1_s1171
                                 - 0.285999627581273 z_k1_s1172
                                 - 0.382999138418791 z_k1_s1173
                                 - 0.249000223900403 z_k1_s1174
                                 - 0.174999650141161 z_k1_s1175
                                 - 0.221000615493343 z_k1_s1176
                                 - 0.359000420354258 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.851839 w_k1_ADM_s000
                                 + 0.906173 w_k1_AXP_s000
                                 + 0.768038 w_k1_DIS_s000
                                 + 0.822412 w_k1_FDX_s000
                                 + 0.887956 w_k1_IBM_s000
                                 + 0.780865 w_k1_INTC_s000
                                 + 0.915274 w_k1_KO_s000
                                 + 1.27535 w_k1_MAT_s000
                                 + 0.848621 w_k1_MSFT_s000
                                 + 0.924387 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.07044 w_k1_ADM_s000
                                 + 1.07074 w_k1_AXP_s000
                                 + 1.11663 w_k1_DIS_s000
                                 + 1.21503 w_k1_FDX_s000
                                 + 0.803259 w_k1_IBM_s000
                                 + 0.837289 w_k1_INTC_s000
                                 + 1.0743 w_k1_KO_s000 + 1.13221 w_k1_MAT_s000
                                 + 0.954502 w_k1_MSFT_s000
                                 + 1.03147 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.831702 w_k1_ADM_s000
                                 + 0.958252 w_k1_AXP_s000
                                 + 0.620632 w_k1_DIS_s000
                                 + 0.846337 w_k1_FDX_s000
                                 + 0.598189 w_k1_IBM_s000
                                 + 0.774226 w_k1_INTC_s000
                                 + 0.727741 w_k1_KO_s000
                                 + 0.907161 w_k1_MAT_s000
                                 + 0.891001 w_k1_MSFT_s000
                                 + 0.87521 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.877596 w_k1_ADM_s000
                                 + 1.0263 w_k1_AXP_s000 + 1.13684 w_k1_DIS_s000
                                 + 0.926034 w_k1_FDX_s000
                                 + 1.32078 w_k1_IBM_s000
                                 + 0.849481 w_k1_INTC_s000
                                 + 1.18318 w_k1_KO_s000 + 2.11046 w_k1_MAT_s000
                                 + 1.05988 w_k1_MSFT_s000
                                 + 1.30201 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.801249 w_k1_ADM_s000
                                 + 1.13238 w_k1_AXP_s000
                                 + 0.973171 w_k1_DIS_s000
                                 + 0.945704 w_k1_FDX_s000
                                 + 1.1372 w_k1_IBM_s000
                                 + 0.747671 w_k1_INTC_s000
                                 + 1.14076 w_k1_KO_s000 + 1.58397 w_k1_MAT_s000
                                 + 0.950709 w_k1_MSFT_s000
                                 + 0.85889 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.898982 w_k1_ADM_s000
                                 + 1.09129 w_k1_AXP_s000
                                 + 0.88568 w_k1_DIS_s000
                                 + 1.00874 w_k1_FDX_s000
                                 + 0.694418 w_k1_IBM_s000
                                 + 0.900973 w_k1_INTC_s000
                                 + 0.831059 w_k1_KO_s000
                                 + 1.04424 w_k1_MAT_s000
                                 + 0.971831 w_k1_MSFT_s000
                                 + 1.00968 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.08937 w_k1_ADM_s000
                                 + 1.66304 w_k1_AXP_s000
                                 + 1.47551 w_k1_DIS_s000
                                 + 1.31874 w_k1_FDX_s000
                                 + 1.05137 w_k1_IBM_s000
                                 + 1.06397 w_k1_INTC_s000
                                 + 1.34537 w_k1_KO_s000 + 1.95992 w_k1_MAT_s000
                                 + 1.35034 w_k1_MSFT_s000
                                 + 1.31504 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.40191 w_k1_ADM_s000
                                 + 1.72742 w_k1_AXP_s000
                                 + 1.85514 w_k1_DIS_s000
                                 + 1.29383 w_k1_FDX_s000
                                 + 1.56984 w_k1_IBM_s000
                                 + 1.01984 w_k1_INTC_s000
                                 + 1.74957 w_k1_KO_s000 + 2.61086 w_k1_MAT_s000
                                 + 1.24727 w_k1_MSFT_s000
                                 + 1.41266 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.2888 w_k1_ADM_s001
                                 + 1.49549 w_k1_AXP_s001
                                 + 1.45403 w_k1_DIS_s001
                                 + 1.34194 w_k1_FDX_s001
                                 + 0.980402 w_k1_IBM_s001
                                 + 1.43962 w_k1_INTC_s001
                                 + 1.13534 w_k1_KO_s001
                                 + 0.875706 w_k1_MAT_s001
                                 + 1.17032 w_k1_MSFT_s001
                                 + 1.00623 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.920988 w_k1_ADM_s001
                                 + 1.04626 w_k1_AXP_s001
                                 + 0.876398 w_k1_DIS_s001
                                 + 0.908433 w_k1_FDX_s001
                                 + 0.755946 w_k1_IBM_s001
                                 + 1.02759 w_k1_INTC_s001
                                 + 0.754841 w_k1_KO_s001
                                 + 0.689668 w_k1_MAT_s001
                                 + 0.853885 w_k1_MSFT_s001
                                 + 0.809191 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.710978 w_k1_ADM_s001
                                 + 0.707292 w_k1_AXP_s001
                                 + 0.588162 w_k1_DIS_s001
                                 + 0.656842 w_k1_FDX_s001
                                 + 0.883881 w_k1_IBM_s001
                                 + 0.829047 w_k1_INTC_s001
                                 + 0.742844 w_k1_KO_s001
                                 + 0.765484 w_k1_MAT_s001
                                 + 0.693325 w_k1_MSFT_s001
                                 + 0.862547 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.00537 w_k1_ADM_s001
                                 + 1.04744 w_k1_AXP_s001
                                 + 1.24347 w_k1_DIS_s001
                                 + 1.02623 w_k1_FDX_s001 + 1.4524 w_k1_IBM_s001
                                 + 1.15083 w_k1_INTC_s001 + 1.3286 w_k1_KO_s001
                                 + 1.32115 w_k1_MAT_s001
                                 + 1.09371 w_k1_MSFT_s001
                                 + 1.08211 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.742534 w_k1_ADM_s001
                                 + 0.899781 w_k1_AXP_s001
                                 + 0.874778 w_k1_DIS_s001
                                 + 0.748023 w_k1_FDX_s001
                                 + 1.25734 w_k1_IBM_s001
                                 + 1.21661 w_k1_INTC_s001
                                 + 1.03886 w_k1_KO_s001
                                 + 0.836005 w_k1_MAT_s001
                                 + 0.9014 w_k1_MSFT_s001
                                 + 0.953023 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.40894 w_k1_ADM_s001
                                 + 1.39075 w_k1_AXP_s001
                                 + 1.50224 w_k1_DIS_s001
                                 + 1.28707 w_k1_FDX_s001 + 1.4437 w_k1_IBM_s001
                                 + 1.50101 w_k1_INTC_s001
                                 + 1.23056 w_k1_KO_s001 + 1.42674 w_k1_MAT_s001
                                 + 1.35046 w_k1_MSFT_s001
                                 + 1.21348 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.72602 w_k1_ADM_s001
                                 + 0.913454 w_k1_AXP_s001
                                 + 0.957656 w_k1_DIS_s001
                                 + 0.890898 w_k1_FDX_s001
                                 + 0.900737 w_k1_IBM_s001
                                 + 1.11181 w_k1_INTC_s001
                                 + 0.912663 w_k1_KO_s001
                                 + 0.879208 w_k1_MAT_s001
                                 + 0.952485 w_k1_MSFT_s001
                                 + 0.891416 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.0803 w_k1_ADM_s001
                                 + 1.13935 w_k1_AXP_s001
                                 + 1.08436 w_k1_DIS_s001
                                 + 1.04747 w_k1_FDX_s001 + 1.0483 w_k1_IBM_s001
                                 + 1.1921 w_k1_INTC_s001
                                 + 0.937632 w_k1_KO_s001
                                 + 0.830067 w_k1_MAT_s001
                                 + 1.03089 w_k1_MSFT_s001
                                 + 0.922004 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08453 w_k1_ADM_s002
                                 + 0.898703 w_k1_AXP_s002
                                 + 0.873011 w_k1_DIS_s002
                                 + 1.03395 w_k1_FDX_s002
                                 + 0.661139 w_k1_IBM_s002
                                 + 0.874196 w_k1_INTC_s002
                                 + 0.782616 w_k1_KO_s002
                                 + 0.940973 w_k1_MAT_s002
                                 + 0.962131 w_k1_MSFT_s002
                                 + 0.989901 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.985348 w_k1_ADM_s002
                                 + 0.789349 w_k1_AXP_s002
                                 + 0.98193 w_k1_DIS_s002
                                 + 0.859382 w_k1_FDX_s002
                                 + 1.35296 w_k1_IBM_s002
                                 + 0.783466 w_k1_INTC_s002
                                 + 1.09737 w_k1_KO_s002 + 1.62891 w_k1_MAT_s002
                                 + 0.841808 w_k1_MSFT_s002
                                 + 0.959647 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.22838 w_k1_ADM_s002
                                 + 1.13572 w_k1_AXP_s002
                                 + 1.68177 w_k1_DIS_s002
                                 + 1.21567 w_k1_FDX_s002
                                 + 1.53494 w_k1_IBM_s002
                                 + 1.10475 w_k1_INTC_s002
                                 + 1.41512 w_k1_KO_s002 + 1.87976 w_k1_MAT_s002
                                 + 1.28919 w_k1_MSFT_s002
                                 + 1.11688 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.96253 w_k1_ADM_s002
                                 + 0.710526 w_k1_AXP_s002
                                 + 0.828125 w_k1_DIS_s002
                                 + 0.910727 w_k1_FDX_s002
                                 + 0.781445 w_k1_IBM_s002
                                 + 0.84077 w_k1_INTC_s002
                                 + 0.859603 w_k1_KO_s002
                                 + 0.665943 w_k1_MAT_s002
                                 + 0.936464 w_k1_MSFT_s002
                                 + 0.840331 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.900544 w_k1_ADM_s002
                                 + 0.729129 w_k1_AXP_s002
                                 + 0.842127 w_k1_DIS_s002
                                 + 0.83512 w_k1_FDX_s002
                                 + 1.10527 w_k1_IBM_s002
                                 + 1.00263 w_k1_INTC_s002
                                 + 1.02223 w_k1_KO_s002 + 1.53624 w_k1_MAT_s002
                                 + 0.924802 w_k1_MSFT_s002
                                 + 0.976735 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.06801 w_k1_ADM_s002
                                 + 1.01719 w_k1_AXP_s002
                                 + 1.13308 w_k1_DIS_s002
                                 + 1.06812 w_k1_FDX_s002
                                 + 1.10611 w_k1_IBM_s002
                                 + 1.18201 w_k1_INTC_s002
                                 + 1.01974 w_k1_KO_s002 + 1.03309 w_k1_MAT_s002
                                 + 1.08405 w_k1_MSFT_s002
                                 + 1.00667 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.27629 w_k1_ADM_s002
                                 + 1.09492 w_k1_AXP_s002
                                 + 1.28661 w_k1_DIS_s002
                                 + 1.32787 w_k1_FDX_s002
                                 + 0.988106 w_k1_IBM_s002
                                 + 1.18561 w_k1_INTC_s002 + 1.0338 w_k1_KO_s002
                                 + 1.25621 w_k1_MAT_s002
                                 + 1.26457 w_k1_MSFT_s002
                                 + 1.01263 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.603982 w_k1_ADM_s002
                                 + 0.616907 w_k1_AXP_s002
                                 + 0.642991 w_k1_DIS_s002
                                 + 0.736449 w_k1_FDX_s002
                                 + 0.754246 w_k1_IBM_s002
                                 + 0.85896 w_k1_INTC_s002
                                 + 0.74633 w_k1_KO_s002
                                 + 0.959672 w_k1_MAT_s002
                                 + 0.753928 w_k1_MSFT_s002
                                 + 0.791002 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.04665 w_k1_ADM_s003
                                 + 0.810918 w_k1_AXP_s003
                                 + 0.743136 w_k1_DIS_s003
                                 + 0.838852 w_k1_FDX_s003
                                 + 0.808631 w_k1_IBM_s003
                                 + 0.877692 w_k1_INTC_s003
                                 + 0.785266 w_k1_KO_s003
                                 + 0.57434 w_k1_MAT_s003
                                 + 0.729649 w_k1_MSFT_s003
                                 + 0.857092 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.13683 w_k1_ADM_s003
                                 + 1.22044 w_k1_AXP_s003
                                 + 1.49784 w_k1_DIS_s003
                                 + 1.18145 w_k1_FDX_s003
                                 + 1.65695 w_k1_IBM_s003
                                 + 1.05429 w_k1_INTC_s003
                                 + 1.34619 w_k1_KO_s003 + 1.53203 w_k1_MAT_s003
                                 + 1.14694 w_k1_MSFT_s003
                                 + 1.09112 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.855652 w_k1_ADM_s003
                                 + 0.617373 w_k1_AXP_s003
                                 + 0.575278 w_k1_DIS_s003
                                 + 0.717935 w_k1_FDX_s003
                                 + 0.743326 w_k1_IBM_s003
                                 + 0.782782 w_k1_INTC_s003
                                 + 0.681265 w_k1_KO_s003
                                 + 0.694765 w_k1_MAT_s003
                                 + 0.731719 w_k1_MSFT_s003
                                 + 0.75373 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.39512 w_k1_ADM_s003
                                 + 1.00305 w_k1_AXP_s003
                                 + 1.06101 w_k1_DIS_s003
                                 + 1.15695 w_k1_FDX_s003
                                 + 0.873924 w_k1_IBM_s003
                                 + 1.16858 w_k1_INTC_s003
                                 + 0.928757 w_k1_KO_s003
                                 + 0.939922 w_k1_MAT_s003
                                 + 1.05536 w_k1_MSFT_s003
                                 + 1.09602 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.12932 w_k1_ADM_s003
                                 + 0.747016 w_k1_AXP_s003
                                 + 0.719522 w_k1_DIS_s003
                                 + 0.945501 w_k1_FDX_s003
                                 + 0.671507 w_k1_IBM_s003
                                 + 0.947721 w_k1_INTC_s003
                                 + 0.720842 w_k1_KO_s003
                                 + 0.632435 w_k1_MAT_s003
                                 + 0.888439 w_k1_MSFT_s003
                                 + 0.805422 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.964124 w_k1_ADM_s003
                                 + 0.717204 w_k1_AXP_s003
                                 + 0.992636 w_k1_DIS_s003
                                 + 0.963651 w_k1_FDX_s003
                                 + 0.738823 w_k1_IBM_s003
                                 + 1.12663 w_k1_INTC_s003
                                 + 0.865324 w_k1_KO_s003
                                 + 0.613843 w_k1_MAT_s003
                                 + 1.10873 w_k1_MSFT_s003
                                 + 0.909904 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.63094 w_k1_ADM_s003
                                 + 1.28824 w_k1_AXP_s003
                                 + 1.58973 w_k1_DIS_s003
                                 + 1.33523 w_k1_FDX_s003
                                 + 1.14761 w_k1_IBM_s003
                                 + 1.1676 w_k1_INTC_s003 + 1.14974 w_k1_KO_s003
                                 + 1.32514 w_k1_MAT_s003
                                 + 1.21809 w_k1_MSFT_s003 + 1.1374 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.894919 w_k1_ADM_s003
                                 + 0.983405 w_k1_AXP_s003
                                 + 0.950097 w_k1_DIS_s003
                                 + 0.969314 w_k1_FDX_s003
                                 + 1.19501 w_k1_IBM_s003
                                 + 0.960601 w_k1_INTC_s003
                                 + 1.04874 w_k1_KO_s003 + 1.23452 w_k1_MAT_s003
                                 + 1.05646 w_k1_MSFT_s003
                                 + 0.939956 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.665481 w_k1_ADM_s004
                                 + 0.753608 w_k1_AXP_s004
                                 + 0.684381 w_k1_DIS_s004
                                 + 0.76906 w_k1_FDX_s004
                                 + 0.980975 w_k1_IBM_s004
                                 + 0.818701 w_k1_INTC_s004
                                 + 0.771887 w_k1_KO_s004
                                 + 0.885798 w_k1_MAT_s004
                                 + 0.773022 w_k1_MSFT_s004
                                 + 0.802557 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.945865 w_k1_ADM_s004
                                 + 0.792094 w_k1_AXP_s004
                                 + 0.912547 w_k1_DIS_s004
                                 + 0.95009 w_k1_FDX_s004
                                 + 0.670722 w_k1_IBM_s004
                                 + 0.954306 w_k1_INTC_s004
                                 + 0.904138 w_k1_KO_s004
                                 + 0.423186 w_k1_MAT_s004
                                 + 0.866634 w_k1_MSFT_s004
                                 + 0.920797 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.882664 w_k1_ADM_s004
                                 + 0.990321 w_k1_AXP_s004
                                 + 0.796807 w_k1_DIS_s004
                                 + 0.740628 w_k1_FDX_s004
                                 + 1.34647 w_k1_IBM_s004
                                 + 0.896597 w_k1_INTC_s004
                                 + 1.04324 w_k1_KO_s004 + 1.30842 w_k1_MAT_s004
                                 + 0.765391 w_k1_MSFT_s004
                                 + 0.881562 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.961369 w_k1_ADM_s004
                                 + 1.55561 w_k1_AXP_s004
                                 + 1.50248 w_k1_DIS_s004
                                 + 1.41444 w_k1_FDX_s004
                                 + 1.45423 w_k1_IBM_s004
                                 + 1.54025 w_k1_INTC_s004
                                 + 1.32623 w_k1_KO_s004 + 1.35725 w_k1_MAT_s004
                                 + 1.19498 w_k1_MSFT_s004 + 1.3243 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.57383 w_k1_ADM_s004
                                 + 0.716913 w_k1_AXP_s004
                                 + 0.516774 w_k1_DIS_s004
                                 + 0.671138 w_k1_FDX_s004
                                 + 0.675419 w_k1_IBM_s004
                                 + 0.737046 w_k1_INTC_s004
                                 + 0.570124 w_k1_KO_s004
                                 + 0.738626 w_k1_MAT_s004
                                 + 0.670844 w_k1_MSFT_s004
                                 + 0.762119 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.835031 w_k1_ADM_s004
                                 + 0.86654 w_k1_AXP_s004
                                 + 0.711833 w_k1_DIS_s004
                                 + 0.973936 w_k1_FDX_s004
                                 + 0.842679 w_k1_IBM_s004
                                 + 0.952785 w_k1_INTC_s004
                                 + 0.763593 w_k1_KO_s004
                                 + 0.830696 w_k1_MAT_s004
                                 + 0.909849 w_k1_MSFT_s004
                                 + 0.91816 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.889881 w_k1_ADM_s004
                                 + 0.996912 w_k1_AXP_s004
                                 + 1.11988 w_k1_DIS_s004
                                 + 1.05422 w_k1_FDX_s004
                                 + 1.18611 w_k1_IBM_s004
                                 + 1.0861 w_k1_INTC_s004 + 1.1079 w_k1_KO_s004
                                 + 1.27884 w_k1_MAT_s004
                                 + 1.12544 w_k1_MSFT_s004
                                 + 1.20722 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.974004 w_k1_ADM_s004
                                 + 1.34796 w_k1_AXP_s004
                                 + 1.16731 w_k1_DIS_s004 + 1.177 w_k1_FDX_s004
                                 + 0.881222 w_k1_IBM_s004
                                 + 1.34024 w_k1_INTC_s004
                                 + 0.946564 w_k1_KO_s004
                                 + 0.933679 w_k1_MAT_s004
                                 + 1.2162 w_k1_MSFT_s004 + 1.00054 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.875153 w_k1_ADM_s005
                                 + 0.901289 w_k1_AXP_s005
                                 + 0.796656 w_k1_DIS_s005
                                 + 0.871407 w_k1_FDX_s005
                                 + 0.612118 w_k1_IBM_s005
                                 + 0.937228 w_k1_INTC_s005
                                 + 0.737266 w_k1_KO_s005
                                 + 0.567376 w_k1_MAT_s005
                                 + 0.827149 w_k1_MSFT_s005
                                 + 0.847975 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.61414 w_k1_ADM_s005
                                 + 0.67274 w_k1_AXP_s005
                                 + 0.589053 w_k1_DIS_s005
                                 + 0.678594 w_k1_FDX_s005
                                 + 0.715675 w_k1_IBM_s005
                                 + 0.922057 w_k1_INTC_s005
                                 + 0.638991 w_k1_KO_s005
                                 + 0.656886 w_k1_MAT_s005
                                 + 0.733993 w_k1_MSFT_s005
                                 + 0.753234 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.927141 w_k1_ADM_s005
                                 + 0.86333 w_k1_AXP_s005
                                 + 0.909063 w_k1_DIS_s005
                                 + 0.899761 w_k1_FDX_s005
                                 + 0.916219 w_k1_IBM_s005
                                 + 0.914143 w_k1_INTC_s005
                                 + 0.798493 w_k1_KO_s005
                                 + 0.852094 w_k1_MAT_s005
                                 + 0.910149 w_k1_MSFT_s005
                                 + 0.927621 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.724718 w_k1_ADM_s005
                                 + 0.790904 w_k1_AXP_s005
                                 + 0.792473 w_k1_DIS_s005
                                 + 0.763017 w_k1_FDX_s005
                                 + 1.18509 w_k1_IBM_s005
                                 + 1.05928 w_k1_INTC_s005
                                 + 0.878332 w_k1_KO_s005
                                 + 0.866673 w_k1_MAT_s005
                                 + 0.829059 w_k1_MSFT_s005
                                 + 0.889364 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.647779 w_k1_ADM_s005
                                 + 0.905974 w_k1_AXP_s005
                                 + 0.919367 w_k1_DIS_s005
                                 + 0.843955 w_k1_FDX_s005
                                 + 1.62563 w_k1_IBM_s005
                                 + 0.873782 w_k1_INTC_s005
                                 + 1.17714 w_k1_KO_s005 + 1.39432 w_k1_MAT_s005
                                 + 0.812075 w_k1_MSFT_s005
                                 + 1.10245 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.00741 w_k1_ADM_s005
                                 + 1.21136 w_k1_AXP_s005
                                 + 1.16257 w_k1_DIS_s005
                                 + 1.06312 w_k1_FDX_s005
                                 + 1.01484 w_k1_IBM_s005
                                 + 1.41778 w_k1_INTC_s005
                                 + 0.986666 w_k1_KO_s005
                                 + 0.794562 w_k1_MAT_s005
                                 + 1.04762 w_k1_MSFT_s005
                                 + 1.04964 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.984098 w_k1_ADM_s005
                                 + 1.37669 w_k1_AXP_s005
                                 + 1.45573 w_k1_DIS_s005
                                 + 1.24169 w_k1_FDX_s005
                                 + 1.45832 w_k1_IBM_s005
                                 + 1.37907 w_k1_INTC_s005
                                 + 1.22652 w_k1_KO_s005 + 1.22543 w_k1_MAT_s005
                                 + 1.10736 w_k1_MSFT_s005
                                 + 1.23197 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.1398 w_k1_ADM_s005
                                 + 1.01617 w_k1_AXP_s005
                                 + 1.57428 w_k1_DIS_s005
                                 + 1.33245 w_k1_FDX_s005
                                 + 0.909864 w_k1_IBM_s005
                                 + 1.22734 w_k1_INTC_s005
                                 + 1.08622 w_k1_KO_s005
                                 + 0.776474 w_k1_MAT_s005
                                 + 1.22204 w_k1_MSFT_s005
                                 + 1.22123 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.00131 w_k1_ADM_s006
                                 + 1.17272 w_k1_AXP_s006 + 1.2697 w_k1_DIS_s006
                                 + 1.37171 w_k1_FDX_s006
                                 + 0.507182 w_k1_IBM_s006
                                 + 1.16105 w_k1_INTC_s006
                                 + 0.869034 w_k1_KO_s006
                                 + 0.618394 w_k1_MAT_s006
                                 + 1.13653 w_k1_MSFT_s006
                                 + 1.07927 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.979872 w_k1_ADM_s006
                                 + 1.62841 w_k1_AXP_s006
                                 + 1.89189 w_k1_DIS_s006
                                 + 1.30956 w_k1_FDX_s006
                                 + 1.65433 w_k1_IBM_s006
                                 + 1.48552 w_k1_INTC_s006
                                 + 1.46618 w_k1_KO_s006 + 1.63579 w_k1_MAT_s006
                                 + 1.16923 w_k1_MSFT_s006
                                 + 1.25655 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.959336 w_k1_ADM_s006
                                 + 1.20785 w_k1_AXP_s006
                                 + 1.27467 w_k1_DIS_s006
                                 + 1.16028 w_k1_FDX_s006 + 1.294 w_k1_IBM_s006
                                 + 0.955078 w_k1_INTC_s006
                                 + 1.24974 w_k1_KO_s006 + 1.65366 w_k1_MAT_s006
                                 + 1.10216 w_k1_MSFT_s006
                                 + 1.18881 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.954352 w_k1_ADM_s006
                                 + 1.15806 w_k1_AXP_s006
                                 + 0.962258 w_k1_DIS_s006
                                 + 1.16591 w_k1_FDX_s006
                                 + 0.930083 w_k1_IBM_s006
                                 + 1.14166 w_k1_INTC_s006
                                 + 0.911496 w_k1_KO_s006
                                 + 0.886854 w_k1_MAT_s006
                                 + 1.12846 w_k1_MSFT_s006
                                 + 0.949331 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.818177 w_k1_ADM_s006
                                 + 0.948641 w_k1_AXP_s006
                                 + 0.687193 w_k1_DIS_s006
                                 + 0.826481 w_k1_FDX_s006
                                 + 1.10161 w_k1_IBM_s006
                                 + 0.861676 w_k1_INTC_s006
                                 + 0.901932 w_k1_KO_s006
                                 + 1.49859 w_k1_MAT_s006
                                 + 0.679619 w_k1_MSFT_s006
                                 + 0.895968 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.817876 w_k1_ADM_s006
                                 + 0.63898 w_k1_AXP_s006
                                 + 0.888206 w_k1_DIS_s006
                                 + 0.86918 w_k1_FDX_s006
                                 + 1.12219 w_k1_IBM_s006
                                 + 0.884206 w_k1_INTC_s006
                                 + 1.0619 w_k1_KO_s006 + 0.857582 w_k1_MAT_s006
                                 + 0.974641 w_k1_MSFT_s006
                                 + 1.08728 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.601288 w_k1_ADM_s006
                                 + 0.641764 w_k1_AXP_s006
                                 + 0.506041 w_k1_DIS_s006
                                 + 0.795387 w_k1_FDX_s006
                                 + 0.455452 w_k1_IBM_s006
                                 + 0.728745 w_k1_INTC_s006
                                 + 0.581183 w_k1_KO_s006
                                 + 0.490625 w_k1_MAT_s006
                                 + 0.743955 w_k1_MSFT_s006
                                 + 0.85792 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.730391 w_k1_ADM_s006
                                 + 0.822813 w_k1_AXP_s006
                                 + 0.834975 w_k1_DIS_s006
                                 + 0.954476 w_k1_FDX_s006
                                 + 0.718192 w_k1_IBM_s006
                                 + 1.04017 w_k1_INTC_s006
                                 + 0.90522 w_k1_KO_s006
                                 + 0.827362 w_k1_MAT_s006
                                 + 0.983428 w_k1_MSFT_s006
                                 + 0.814583 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.752783 w_k1_ADM_s007
                                 + 0.774213 w_k1_AXP_s007
                                 + 0.754288 w_k1_DIS_s007
                                 + 0.746091 w_k1_FDX_s007
                                 + 1.04863 w_k1_IBM_s007
                                 + 0.753197 w_k1_INTC_s007
                                 + 0.797778 w_k1_KO_s007
                                 + 0.793091 w_k1_MAT_s007
                                 + 0.813291 w_k1_MSFT_s007
                                 + 0.948646 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.26811 w_k1_ADM_s007
                                 + 0.954281 w_k1_AXP_s007
                                 + 0.79617 w_k1_DIS_s007
                                 + 1.01498 w_k1_FDX_s007 + 1.169 w_k1_IBM_s007
                                 + 0.741117 w_k1_INTC_s007
                                 + 0.904176 w_k1_KO_s007
                                 + 1.30223 w_k1_MAT_s007
                                 + 0.846608 w_k1_MSFT_s007
                                 + 1.06324 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.10515 w_k1_ADM_s007
                                 + 1.35107 w_k1_AXP_s007
                                 + 1.46467 w_k1_DIS_s007
                                 + 1.28761 w_k1_FDX_s007
                                 + 1.11308 w_k1_IBM_s007
                                 + 1.02778 w_k1_INTC_s007
                                 + 1.13027 w_k1_KO_s007 + 1.38112 w_k1_MAT_s007
                                 + 1.16593 w_k1_MSFT_s007
                                 + 1.27103 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.863392 w_k1_ADM_s007
                                 + 1.2358 w_k1_AXP_s007 + 1.47572 w_k1_DIS_s007
                                 + 1.18401 w_k1_FDX_s007
                                 + 1.78499 w_k1_IBM_s007
                                 + 1.06911 w_k1_INTC_s007
                                 + 1.29178 w_k1_KO_s007 + 1.73619 w_k1_MAT_s007
                                 + 1.1774 w_k1_MSFT_s007 + 1.11457 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.969112 w_k1_ADM_s007
                                 + 1.05941 w_k1_AXP_s007
                                 + 1.08274 w_k1_DIS_s007
                                 + 0.946093 w_k1_FDX_s007
                                 + 1.54579 w_k1_IBM_s007
                                 + 0.96181 w_k1_INTC_s007
                                 + 1.23912 w_k1_KO_s007 + 1.61775 w_k1_MAT_s007
                                 + 1.00281 w_k1_MSFT_s007
                                 + 1.13485 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.02308 w_k1_ADM_s007
                                 + 1.16366 w_k1_AXP_s007
                                 + 1.27565 w_k1_DIS_s007
                                 + 1.26342 w_k1_FDX_s007
                                 + 0.881904 w_k1_IBM_s007
                                 + 0.886132 w_k1_INTC_s007
                                 + 1.15568 w_k1_KO_s007 + 1.201 w_k1_MAT_s007
                                 + 1.20823 w_k1_MSFT_s007
                                 + 1.19366 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.09557 w_k1_ADM_s007
                                 + 1.02876 w_k1_AXP_s007
                                 + 0.998486 w_k1_DIS_s007
                                 + 1.01015 w_k1_FDX_s007
                                 + 0.849261 w_k1_IBM_s007
                                 + 0.897073 w_k1_INTC_s007
                                 + 0.968308 w_k1_KO_s007
                                 + 1.33827 w_k1_MAT_s007
                                 + 0.963053 w_k1_MSFT_s007
                                 + 1.04693 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.966191 w_k1_ADM_s007
                                 + 0.620473 w_k1_AXP_s007
                                 + 0.89766 w_k1_DIS_s007
                                 + 0.905512 w_k1_FDX_s007
                                 + 0.950275 w_k1_IBM_s007
                                 + 0.823097 w_k1_INTC_s007
                                 + 0.889149 w_k1_KO_s007
                                 + 0.745449 w_k1_MAT_s007
                                 + 0.980311 w_k1_MSFT_s007
                                 + 0.955874 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.626227 w_k1_ADM_s010
                                 + 0.902187 w_k1_AXP_s010
                                 + 1.12731 w_k1_DIS_s010
                                 + 0.795747 w_k1_FDX_s010
                                 + 1.74775 w_k1_IBM_s010
                                 + 0.800843 w_k1_INTC_s010
                                 + 1.18759 w_k1_KO_s010 + 1.88343 w_k1_MAT_s010
                                 + 0.855142 w_k1_MSFT_s010
                                 + 1.10757 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.682113 w_k1_ADM_s010
                                 + 0.770605 w_k1_AXP_s010
                                 + 0.78415 w_k1_DIS_s010 + 0.8307 w_k1_FDX_s010
                                 + 0.814806 w_k1_IBM_s010
                                 + 0.870976 w_k1_INTC_s010
                                 + 0.848264 w_k1_KO_s010 + 1.1194 w_k1_MAT_s010
                                 + 0.81793 w_k1_MSFT_s010
                                 + 0.965414 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.866111 w_k1_ADM_s010
                                 + 1.0221 w_k1_AXP_s010
                                 + 0.873843 w_k1_DIS_s010
                                 + 0.892372 w_k1_FDX_s010
                                 + 1.29032 w_k1_IBM_s010
                                 + 0.985851 w_k1_INTC_s010
                                 + 0.923644 w_k1_KO_s010
                                 + 1.45667 w_k1_MAT_s010
                                 + 0.897508 w_k1_MSFT_s010
                                 + 0.953668 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.3698 w_k1_ADM_s010
                                 + 1.69461 w_k1_AXP_s010
                                 + 1.70244 w_k1_DIS_s010
                                 + 1.35925 w_k1_FDX_s010
                                 + 1.73965 w_k1_IBM_s010
                                 + 1.39749 w_k1_INTC_s010
                                 + 1.54936 w_k1_KO_s010 + 2.36851 w_k1_MAT_s010
                                 + 1.32184 w_k1_MSFT_s010 + 1.3317 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.04641 w_k1_ADM_s010
                                 + 1.46123 w_k1_AXP_s010
                                 + 1.44386 w_k1_DIS_s010
                                 + 1.45513 w_k1_FDX_s010
                                 + 0.804765 w_k1_IBM_s010
                                 + 1.14043 w_k1_INTC_s010
                                 + 0.937735 w_k1_KO_s010 + 1.1423 w_k1_MAT_s010
                                 + 1.14303 w_k1_MSFT_s010
                                 + 1.17063 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.933532 w_k1_ADM_s010
                                 + 0.928719 w_k1_AXP_s010
                                 + 1.0229 w_k1_DIS_s010 + 1.04908 w_k1_FDX_s010
                                 + 0.837288 w_k1_IBM_s010
                                 + 0.871547 w_k1_INTC_s010
                                 + 0.958379 w_k1_KO_s010
                                 + 0.571979 w_k1_MAT_s010
                                 + 0.993887 w_k1_MSFT_s010
                                 + 1.01454 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.826854 w_k1_ADM_s010
                                 + 0.89468 w_k1_AXP_s010
                                 + 1.02446 w_k1_DIS_s010
                                 + 0.916048 w_k1_FDX_s010
                                 + 1.14601 w_k1_IBM_s010
                                 + 0.821473 w_k1_INTC_s010
                                 + 0.931789 w_k1_KO_s010
                                 + 0.87773 w_k1_MAT_s010
                                 + 0.727145 w_k1_MSFT_s010
                                 + 1.00865 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.10691 w_k1_ADM_s010
                                 + 1.65748 w_k1_AXP_s010
                                 + 1.43041 w_k1_DIS_s010
                                 + 1.28074 w_k1_FDX_s010
                                 + 1.25803 w_k1_IBM_s010
                                 + 1.01046 w_k1_INTC_s010
                                 + 1.37848 w_k1_KO_s010 + 1.74573 w_k1_MAT_s010
                                 + 1.06839 w_k1_MSFT_s010 + 1.2665 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.00958 w_k1_ADM_s011
                                 + 0.9779 w_k1_AXP_s011 + 1.0643 w_k1_DIS_s011
                                 + 0.922219 w_k1_FDX_s011
                                 + 1.30417 w_k1_IBM_s011
                                 + 0.908332 w_k1_INTC_s011
                                 + 1.21312 w_k1_KO_s011 + 1.13343 w_k1_MAT_s011
                                 + 0.901945 w_k1_MSFT_s011
                                 + 0.988939 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.745487 w_k1_ADM_s011
                                 + 0.676595 w_k1_AXP_s011
                                 + 0.632092 w_k1_DIS_s011
                                 + 0.621887 w_k1_FDX_s011
                                 + 1.03985 w_k1_IBM_s011
                                 + 0.829454 w_k1_INTC_s011
                                 + 0.848906 w_k1_KO_s011 + 0.9388 w_k1_MAT_s011
                                 + 0.662323 w_k1_MSFT_s011
                                 + 0.776012 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.981186 w_k1_ADM_s011
                                 + 0.969975 w_k1_AXP_s011
                                 + 0.985733 w_k1_DIS_s011
                                 + 0.998618 w_k1_FDX_s011
                                 + 0.794216 w_k1_IBM_s011
                                 + 0.969085 w_k1_INTC_s011
                                 + 0.816051 w_k1_KO_s011
                                 + 0.841105 w_k1_MAT_s011
                                 + 0.866845 w_k1_MSFT_s011
                                 + 0.90106 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.850497 w_k1_ADM_s011
                                 + 0.652001 w_k1_AXP_s011
                                 + 0.689959 w_k1_DIS_s011
                                 + 0.796737 w_k1_FDX_s011
                                 + 0.832322 w_k1_IBM_s011
                                 + 0.768099 w_k1_INTC_s011
                                 + 0.774144 w_k1_KO_s011
                                 + 0.714554 w_k1_MAT_s011
                                 + 0.785273 w_k1_MSFT_s011
                                 + 0.826567 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.851056 w_k1_ADM_s011
                                 + 0.937737 w_k1_AXP_s011
                                 + 0.723367 w_k1_DIS_s011
                                 + 0.853723 w_k1_FDX_s011
                                 + 0.722331 w_k1_IBM_s011
                                 + 0.874639 w_k1_INTC_s011
                                 + 0.762359 w_k1_KO_s011
                                 + 1.01927 w_k1_MAT_s011
                                 + 0.784672 w_k1_MSFT_s011
                                 + 0.918049 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.12799 w_k1_ADM_s011
                                 + 1.15944 w_k1_AXP_s011 + 1.407 w_k1_DIS_s011
                                 + 1.22385 w_k1_FDX_s011
                                 + 1.05005 w_k1_IBM_s011
                                 + 1.12793 w_k1_INTC_s011
                                 + 1.15911 w_k1_KO_s011 + 1.19498 w_k1_MAT_s011
                                 + 1.17187 w_k1_MSFT_s011
                                 + 1.10107 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.12877 w_k1_ADM_s011
                                 + 1.67133 w_k1_AXP_s011
                                 + 1.57948 w_k1_DIS_s011 + 1.2333 w_k1_FDX_s011
                                 + 1.48588 w_k1_IBM_s011
                                 + 1.36341 w_k1_INTC_s011
                                 + 1.37925 w_k1_KO_s011 + 2.54431 w_k1_MAT_s011
                                 + 1.15717 w_k1_MSFT_s011
                                 + 1.05407 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.938448 w_k1_ADM_s011
                                 + 0.870203 w_k1_AXP_s011
                                 + 1.23978 w_k1_DIS_s011
                                 + 1.05446 w_k1_FDX_s011
                                 + 1.02277 w_k1_IBM_s011
                                 + 0.985344 w_k1_INTC_s011
                                 + 1.10391 w_k1_KO_s011
                                 + 0.828886 w_k1_MAT_s011
                                 + 1.09567 w_k1_MSFT_s011
                                 + 0.883073 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.939549 w_k1_ADM_s012
                                 + 0.830843 w_k1_AXP_s012
                                 + 0.992705 w_k1_DIS_s012
                                 + 0.953107 w_k1_FDX_s012
                                 + 0.905293 w_k1_IBM_s012
                                 + 0.888585 w_k1_INTC_s012
                                 + 0.884358 w_k1_KO_s012
                                 + 0.853569 w_k1_MAT_s012
                                 + 0.921872 w_k1_MSFT_s012
                                 + 0.976195 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.18004 w_k1_ADM_s012
                                 + 1.23508 w_k1_AXP_s012
                                 + 1.47871 w_k1_DIS_s012
                                 + 1.28651 w_k1_FDX_s012
                                 + 1.13225 w_k1_IBM_s012
                                 + 1.1058 w_k1_INTC_s012 + 1.18348 w_k1_KO_s012
                                 + 1.20056 w_k1_MAT_s012
                                 + 1.49644 w_k1_MSFT_s012
                                 + 1.19667 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.891562 w_k1_ADM_s012
                                 + 0.825849 w_k1_AXP_s012
                                 + 0.636089 w_k1_DIS_s012
                                 + 0.759874 w_k1_FDX_s012
                                 + 0.838696 w_k1_IBM_s012
                                 + 0.805133 w_k1_INTC_s012
                                 + 0.753229 w_k1_KO_s012
                                 + 0.853243 w_k1_MAT_s012
                                 + 0.781367 w_k1_MSFT_s012
                                 + 0.883322 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.826515 w_k1_ADM_s012
                                 + 0.618004 w_k1_AXP_s012
                                 + 0.541932 w_k1_DIS_s012
                                 + 0.640718 w_k1_FDX_s012
                                 + 1.00078 w_k1_IBM_s012
                                 + 0.671422 w_k1_INTC_s012
                                 + 0.791401 w_k1_KO_s012
                                 + 0.78783 w_k1_MAT_s012
                                 + 0.710766 w_k1_MSFT_s012
                                 + 0.782665 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.07987 w_k1_ADM_s012
                                 + 1.38716 w_k1_AXP_s012 + 1.2928 w_k1_DIS_s012
                                 + 1.27461 w_k1_FDX_s012
                                 + 0.914832 w_k1_IBM_s012
                                 + 1.08001 w_k1_INTC_s012
                                 + 1.09591 w_k1_KO_s012 + 1.16595 w_k1_MAT_s012
                                 + 1.10166 w_k1_MSFT_s012
                                 + 1.22333 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.06162 w_k1_ADM_s012
                                 + 1.17209 w_k1_AXP_s012
                                 + 1.01497 w_k1_DIS_s012 + 1.0056 w_k1_FDX_s012
                                 + 1.45958 w_k1_IBM_s012
                                 + 0.87776 w_k1_INTC_s012
                                 + 0.988033 w_k1_KO_s012
                                 + 1.51021 w_k1_MAT_s012
                                 + 0.937173 w_k1_MSFT_s012
                                 + 0.972685 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.06167 w_k1_ADM_s012
                                 + 1.15949 w_k1_AXP_s012
                                 + 1.13967 w_k1_DIS_s012
                                 + 1.23033 w_k1_FDX_s012
                                 + 0.784297 w_k1_IBM_s012
                                 + 1.16374 w_k1_INTC_s012
                                 + 0.918259 w_k1_KO_s012
                                 + 0.998999 w_k1_MAT_s012
                                 + 1.19596 w_k1_MSFT_s012
                                 + 1.08473 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.927425 w_k1_ADM_s012
                                 + 0.636371 w_k1_AXP_s012
                                 + 0.625591 w_k1_DIS_s012
                                 + 0.774679 w_k1_FDX_s012
                                 + 0.556775 w_k1_IBM_s012
                                 + 0.736983 w_k1_INTC_s012
                                 + 0.618989 w_k1_KO_s012
                                 + 0.610498 w_k1_MAT_s012
                                 + 0.778884 w_k1_MSFT_s012
                                 + 0.846705 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.11929 w_k1_ADM_s013
                                 + 1.08408 w_k1_AXP_s013
                                 + 1.11696 w_k1_DIS_s013
                                 + 1.04375 w_k1_FDX_s013
                                 + 1.60267 w_k1_IBM_s013
                                 + 1.29067 w_k1_INTC_s013
                                 + 1.38631 w_k1_KO_s013 + 1.49941 w_k1_MAT_s013
                                 + 1.21806 w_k1_MSFT_s013
                                 + 1.14469 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.906834 w_k1_ADM_s013
                                 + 0.897622 w_k1_AXP_s013
                                 + 1.16054 w_k1_DIS_s013
                                 + 0.98309 w_k1_FDX_s013
                                 + 1.19636 w_k1_IBM_s013
                                 + 0.962251 w_k1_INTC_s013
                                 + 1.19059 w_k1_KO_s013 + 1.114 w_k1_MAT_s013
                                 + 1.02155 w_k1_MSFT_s013
                                 + 0.962823 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.16187 w_k1_ADM_s013
                                 + 1.24486 w_k1_AXP_s013
                                 + 1.51233 w_k1_DIS_s013
                                 + 1.32154 w_k1_FDX_s013
                                 + 1.15127 w_k1_IBM_s013
                                 + 1.39289 w_k1_INTC_s013
                                 + 1.28726 w_k1_KO_s013 + 1.40748 w_k1_MAT_s013
                                 + 1.37534 w_k1_MSFT_s013 + 1.0925 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.691321 w_k1_ADM_s013
                                 + 0.551997 w_k1_AXP_s013
                                 + 0.662427 w_k1_DIS_s013
                                 + 0.739354 w_k1_FDX_s013
                                 + 0.852496 w_k1_IBM_s013
                                 + 0.838997 w_k1_INTC_s013
                                 + 0.770496 w_k1_KO_s013
                                 + 0.856912 w_k1_MAT_s013
                                 + 0.88518 w_k1_MSFT_s013
                                 + 0.847032 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.61356 w_k1_ADM_s013
                                 + 1.93863 w_k1_AXP_s013
                                 + 1.68927 w_k1_DIS_s013
                                 + 1.46732 w_k1_FDX_s013
                                 + 1.36809 w_k1_IBM_s013
                                 + 1.34767 w_k1_INTC_s013
                                 + 1.52373 w_k1_KO_s013 + 2.1065 w_k1_MAT_s013
                                 + 1.39516 w_k1_MSFT_s013 + 1.3293 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.07992 w_k1_ADM_s013
                                 + 0.818513 w_k1_AXP_s013
                                 + 0.95177 w_k1_DIS_s013
                                 + 0.952548 w_k1_FDX_s013
                                 + 0.856817 w_k1_IBM_s013
                                 + 1.18149 w_k1_INTC_s013
                                 + 0.906154 w_k1_KO_s013
                                 + 0.824033 w_k1_MAT_s013
                                 + 1.00184 w_k1_MSFT_s013
                                 + 0.971286 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.922909 w_k1_ADM_s013
                                 + 1.09914 w_k1_AXP_s013
                                 + 0.923834 w_k1_DIS_s013
                                 + 1.04391 w_k1_FDX_s013
                                 + 1.09709 w_k1_IBM_s013
                                 + 0.953731 w_k1_INTC_s013
                                 + 0.974775 w_k1_KO_s013
                                 + 1.16643 w_k1_MAT_s013 + 1.071 w_k1_MSFT_s013
                                 + 0.999485 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.22188 w_k1_ADM_s013
                                 + 1.14587 w_k1_AXP_s013
                                 + 1.52627 w_k1_DIS_s013
                                 + 1.27767 w_k1_FDX_s013
                                 + 0.900179 w_k1_IBM_s013
                                 + 1.19881 w_k1_INTC_s013 + 1.1971 w_k1_KO_s013
                                 + 1.27793 w_k1_MAT_s013
                                 + 1.42983 w_k1_MSFT_s013
                                 + 1.17569 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.702632 w_k1_ADM_s014
                                 + 0.618977 w_k1_AXP_s014
                                 + 0.544307 w_k1_DIS_s014
                                 + 0.767854 w_k1_FDX_s014
                                 + 0.60464 w_k1_IBM_s014
                                 + 0.805194 w_k1_INTC_s014
                                 + 0.565303 w_k1_KO_s014
                                 + 0.43208 w_k1_MAT_s014
                                 + 0.707907 w_k1_MSFT_s014
                                 + 0.818228 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.918187 w_k1_ADM_s014
                                 + 0.931096 w_k1_AXP_s014
                                 + 0.824562 w_k1_DIS_s014
                                 + 0.981366 w_k1_FDX_s014
                                 + 0.83595 w_k1_IBM_s014
                                 + 0.997576 w_k1_INTC_s014
                                 + 0.750603 w_k1_KO_s014
                                 + 0.697963 w_k1_MAT_s014
                                 + 0.978767 w_k1_MSFT_s014
                                 + 0.96228 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.47942 w_k1_ADM_s014
                                 + 1.85903 w_k1_AXP_s014
                                 + 1.71961 w_k1_DIS_s014
                                 + 1.62665 w_k1_FDX_s014
                                 + 1.42099 w_k1_IBM_s014
                                 + 1.7526 w_k1_INTC_s014 + 1.30566 w_k1_KO_s014
                                 + 1.42125 w_k1_MAT_s014
                                 + 1.43354 w_k1_MSFT_s014
                                 + 1.35622 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.934346 w_k1_ADM_s014
                                 + 0.854801 w_k1_AXP_s014
                                 + 0.959794 w_k1_DIS_s014
                                 + 0.913359 w_k1_FDX_s014
                                 + 1.14248 w_k1_IBM_s014
                                 + 1.05401 w_k1_INTC_s014
                                 + 0.936082 w_k1_KO_s014
                                 + 0.608414 w_k1_MAT_s014
                                 + 0.905149 w_k1_MSFT_s014
                                 + 0.974135 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.633403 w_k1_ADM_s014
                                 + 0.676918 w_k1_AXP_s014
                                 + 0.69474 w_k1_DIS_s014
                                 + 0.752377 w_k1_FDX_s014
                                 + 0.953757 w_k1_IBM_s014
                                 + 0.957321 w_k1_INTC_s014
                                 + 0.77983 w_k1_KO_s014
                                 + 0.625232 w_k1_MAT_s014
                                 + 0.805223 w_k1_MSFT_s014
                                 + 0.895466 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.954274 w_k1_ADM_s014
                                 + 1.12136 w_k1_AXP_s014
                                 + 1.18757 w_k1_DIS_s014
                                 + 1.20054 w_k1_FDX_s014
                                 + 1.11927 w_k1_IBM_s014
                                 + 1.25624 w_k1_INTC_s014
                                 + 0.996587 w_k1_KO_s014
                                 + 1.10508 w_k1_MAT_s014
                                 + 1.18953 w_k1_MSFT_s014
                                 + 1.13756 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14361 w_k1_ADM_s014
                                 + 1.35948 w_k1_AXP_s014
                                 + 1.19989 w_k1_DIS_s014
                                 + 1.20672 w_k1_FDX_s014
                                 + 1.50952 w_k1_IBM_s014
                                 + 1.26383 w_k1_INTC_s014 + 1.1404 w_k1_KO_s014
                                 + 1.31415 w_k1_MAT_s014
                                 + 1.26819 w_k1_MSFT_s014
                                 + 1.18617 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.878957 w_k1_ADM_s014
                                 + 0.996267 w_k1_AXP_s014
                                 + 1.02154 w_k1_DIS_s014
                                 + 1.15653 w_k1_FDX_s014
                                 + 0.873731 w_k1_IBM_s014
                                 + 1.24044 w_k1_INTC_s014
                                 + 0.806993 w_k1_KO_s014
                                 + 0.511862 w_k1_MAT_s014
                                 + 1.10709 w_k1_MSFT_s014
                                 + 1.01439 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.936953 w_k1_ADM_s015
                                 + 1.57307 w_k1_AXP_s015 + 1.5926 w_k1_DIS_s015
                                 + 1.14336 w_k1_FDX_s015
                                 + 1.75859 w_k1_IBM_s015
                                 + 1.12185 w_k1_INTC_s015
                                 + 1.48457 w_k1_KO_s015 + 1.83944 w_k1_MAT_s015
                                 + 1.25222 w_k1_MSFT_s015
                                 + 1.40282 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.11432 w_k1_ADM_s015
                                 + 0.909196 w_k1_AXP_s015
                                 + 1.03353 w_k1_DIS_s015
                                 + 0.865131 w_k1_FDX_s015
                                 + 0.968521 w_k1_IBM_s015
                                 + 0.90061 w_k1_INTC_s015
                                 + 1.00226 w_k1_KO_s015
                                 + 0.812855 w_k1_MAT_s015
                                 + 1.01192 w_k1_MSFT_s015
                                 + 0.945472 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.878418 w_k1_ADM_s015
                                 + 1.26367 w_k1_AXP_s015
                                 + 1.18905 w_k1_DIS_s015
                                 + 0.981343 w_k1_FDX_s015
                                 + 1.29927 w_k1_IBM_s015
                                 + 1.11025 w_k1_INTC_s015
                                 + 1.20825 w_k1_KO_s015 + 1.27056 w_k1_MAT_s015
                                 + 1.06949 w_k1_MSFT_s015
                                 + 1.11123 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.65193 w_k1_ADM_s015
                                 + 1.635 w_k1_AXP_s015 + 1.487 w_k1_DIS_s015
                                 + 1.26577 w_k1_FDX_s015
                                 + 0.989114 w_k1_IBM_s015
                                 + 1.16071 w_k1_INTC_s015
                                 + 1.10498 w_k1_KO_s015
                                 + 0.953693 w_k1_MAT_s015
                                 + 1.41906 w_k1_MSFT_s015
                                 + 1.11494 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.871612 w_k1_ADM_s015
                                 + 0.642834 w_k1_AXP_s015
                                 + 0.546165 w_k1_DIS_s015
                                 + 0.632644 w_k1_FDX_s015
                                 + 0.866213 w_k1_IBM_s015
                                 + 0.849205 w_k1_INTC_s015
                                 + 0.818649 w_k1_KO_s015
                                 + 0.709766 w_k1_MAT_s015
                                 + 0.876075 w_k1_MSFT_s015
                                 + 0.811692 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.877558 w_k1_ADM_s015
                                 + 0.790005 w_k1_AXP_s015
                                 + 0.681045 w_k1_DIS_s015
                                 + 0.773981 w_k1_FDX_s015
                                 + 1.17917 w_k1_IBM_s015
                                 + 0.81579 w_k1_INTC_s015
                                 + 1.04728 w_k1_KO_s015 + 1.16615 w_k1_MAT_s015
                                 + 0.906097 w_k1_MSFT_s015
                                 + 0.975473 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.38584 w_k1_ADM_s015
                                 + 1.36866 w_k1_AXP_s015
                                 + 1.04739 w_k1_DIS_s015
                                 + 1.09598 w_k1_FDX_s015
                                 + 1.01809 w_k1_IBM_s015
                                 + 1.03446 w_k1_INTC_s015
                                 + 1.03686 w_k1_KO_s015 + 1.09054 w_k1_MAT_s015
                                 + 1.15374 w_k1_MSFT_s015
                                 + 1.13805 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.952574 w_k1_ADM_s015
                                 + 0.755827 w_k1_AXP_s015
                                 + 0.772006 w_k1_DIS_s015
                                 + 0.855694 w_k1_FDX_s015
                                 + 0.72941 w_k1_IBM_s015
                                 + 0.860183 w_k1_INTC_s015
                                 + 0.971214 w_k1_KO_s015
                                 + 0.644887 w_k1_MAT_s015
                                 + 0.959291 w_k1_MSFT_s015
                                 + 0.971371 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.02038 w_k1_ADM_s016
                                 + 1.032 w_k1_AXP_s016 + 1.28678 w_k1_DIS_s016
                                 + 1.06416 w_k1_FDX_s016
                                 + 1.55397 w_k1_IBM_s016
                                 + 1.10129 w_k1_INTC_s016
                                 + 1.40936 w_k1_KO_s016 + 1.38132 w_k1_MAT_s016
                                 + 1.17104 w_k1_MSFT_s016
                                 + 1.07296 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.05892 w_k1_ADM_s016
                                 + 1.15989 w_k1_AXP_s016
                                 + 1.14926 w_k1_DIS_s016
                                 + 1.14959 w_k1_FDX_s016
                                 + 0.948493 w_k1_IBM_s016
                                 + 0.892116 w_k1_INTC_s016
                                 + 1.20875 w_k1_KO_s016 + 1.05658 w_k1_MAT_s016
                                 + 1.005 w_k1_MSFT_s016 + 1.07566 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.2167 w_k1_ADM_s016
                                 + 1.44727 w_k1_AXP_s016 + 1.6793 w_k1_DIS_s016
                                 + 1.3623 w_k1_FDX_s016 + 1.32555 w_k1_IBM_s016
                                 + 1.21933 w_k1_INTC_s016
                                 + 1.42932 w_k1_KO_s016 + 1.73378 w_k1_MAT_s016
                                 + 1.29932 w_k1_MSFT_s016
                                 + 1.21004 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.793838 w_k1_ADM_s016
                                 + 0.974445 w_k1_AXP_s016
                                 + 0.845454 w_k1_DIS_s016
                                 + 0.825075 w_k1_FDX_s016
                                 + 1.35354 w_k1_IBM_s016
                                 + 0.967103 w_k1_INTC_s016
                                 + 1.08264 w_k1_KO_s016 + 1.22269 w_k1_MAT_s016
                                 + 0.849254 w_k1_MSFT_s016
                                 + 0.995626 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.00153 w_k1_ADM_s016
                                 + 0.81456 w_k1_AXP_s016
                                 + 1.16884 w_k1_DIS_s016
                                 + 0.940279 w_k1_FDX_s016
                                 + 1.07755 w_k1_IBM_s016
                                 + 0.841973 w_k1_INTC_s016
                                 + 1.25465 w_k1_KO_s016
                                 + 0.705911 w_k1_MAT_s016
                                 + 0.96423 w_k1_MSFT_s016
                                 + 0.906577 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.78846 w_k1_ADM_s016
                                 + 0.78803 w_k1_AXP_s016
                                 + 0.998804 w_k1_DIS_s016
                                 + 0.920547 w_k1_FDX_s016
                                 + 0.935946 w_k1_IBM_s016
                                 + 0.83534 w_k1_INTC_s016
                                 + 1.01287 w_k1_KO_s016
                                 + 0.677169 w_k1_MAT_s016
                                 + 0.926406 w_k1_MSFT_s016
                                 + 0.976383 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.581358 w_k1_ADM_s016
                                 + 0.580914 w_k1_AXP_s016
                                 + 0.615486 w_k1_DIS_s016
                                 + 0.740972 w_k1_FDX_s016
                                 + 0.768322 w_k1_IBM_s016
                                 + 0.762788 w_k1_INTC_s016
                                 + 0.70242 w_k1_KO_s016
                                 + 0.572086 w_k1_MAT_s016
                                 + 0.816587 w_k1_MSFT_s016
                                 + 0.782097 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.914475 w_k1_ADM_s016
                                 + 0.944581 w_k1_AXP_s016
                                 + 0.854631 w_k1_DIS_s016
                                 + 1.00036 w_k1_FDX_s016
                                 + 0.833498 w_k1_IBM_s016
                                 + 0.868883 w_k1_INTC_s016
                                 + 0.814007 w_k1_KO_s016
                                 + 0.808135 w_k1_MAT_s016
                                 + 0.95929 w_k1_MSFT_s016
                                 + 0.869349 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.940288 w_k1_ADM_s017
                                 + 1.43027 w_k1_AXP_s017
                                 + 1.58931 w_k1_DIS_s017
                                 + 1.30673 w_k1_FDX_s017
                                 + 1.19705 w_k1_IBM_s017
                                 + 1.37356 w_k1_INTC_s017
                                 + 1.39245 w_k1_KO_s017 + 1.25586 w_k1_MAT_s017
                                 + 1.30019 w_k1_MSFT_s017
                                 + 1.24595 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.918357 w_k1_ADM_s017
                                 + 1.3086 w_k1_AXP_s017 + 1.23934 w_k1_DIS_s017
                                 + 1.17293 w_k1_FDX_s017
                                 + 1.46823 w_k1_IBM_s017
                                 + 1.07717 w_k1_INTC_s017
                                 + 1.49446 w_k1_KO_s017 + 1.68559 w_k1_MAT_s017
                                 + 1.06887 w_k1_MSFT_s017
                                 + 1.20599 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.03477 w_k1_ADM_s017
                                 + 0.726421 w_k1_AXP_s017
                                 + 0.997682 w_k1_DIS_s017
                                 + 0.944648 w_k1_FDX_s017
                                 + 1.08992 w_k1_IBM_s017
                                 + 1.07331 w_k1_INTC_s017
                                 + 1.17059 w_k1_KO_s017
                                 + 0.688336 w_k1_MAT_s017
                                 + 1.01819 w_k1_MSFT_s017
                                 + 0.974592 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.16558 w_k1_ADM_s017
                                 + 1.15183 w_k1_AXP_s017
                                 + 0.840117 w_k1_DIS_s017
                                 + 1.04321 w_k1_FDX_s017
                                 + 0.903993 w_k1_IBM_s017
                                 + 1.01628 w_k1_INTC_s017
                                 + 1.03861 w_k1_KO_s017
                                 + 0.956569 w_k1_MAT_s017
                                 + 0.939404 w_k1_MSFT_s017
                                 + 0.936435 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.3104 w_k1_ADM_s017
                                 + 2.27564 w_k1_AXP_s017
                                 + 1.79678 w_k1_DIS_s017
                                 + 1.65276 w_k1_FDX_s017
                                 + 0.975742 w_k1_IBM_s017
                                 + 1.39056 w_k1_INTC_s017
                                 + 1.29313 w_k1_KO_s017 + 1.75279 w_k1_MAT_s017
                                 + 1.33383 w_k1_MSFT_s017
                                 + 1.27281 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.842904 w_k1_ADM_s017
                                 + 0.692226 w_k1_AXP_s017
                                 + 0.63721 w_k1_DIS_s017
                                 + 0.786381 w_k1_FDX_s017
                                 + 0.788547 w_k1_IBM_s017
                                 + 0.859787 w_k1_INTC_s017
                                 + 0.746214 w_k1_KO_s017
                                 + 0.615615 w_k1_MAT_s017
                                 + 0.76596 w_k1_MSFT_s017
                                 + 0.840115 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.808349 w_k1_ADM_s017
                                 + 0.864678 w_k1_AXP_s017
                                 + 0.932719 w_k1_DIS_s017
                                 + 0.733657 w_k1_FDX_s017
                                 + 1.42413 w_k1_IBM_s017
                                 + 0.789629 w_k1_INTC_s017
                                 + 1.07412 w_k1_KO_s017 + 1.09432 w_k1_MAT_s017
                                 + 0.795015 w_k1_MSFT_s017
                                 + 0.869145 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.39996 w_k1_ADM_s017
                                 + 1.3721 w_k1_AXP_s017 + 1.42381 w_k1_DIS_s017
                                 + 1.36494 w_k1_FDX_s017
                                 + 1.04746 w_k1_IBM_s017
                                 + 1.00793 w_k1_INTC_s017 + 1.2161 w_k1_KO_s017
                                 + 1.27532 w_k1_MAT_s017
                                 + 1.09787 w_k1_MSFT_s017
                                 + 1.19634 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.40884 w_k1_ADM_s100
                                 + 1.49343 w_k1_AXP_s100
                                 + 0.781307 w_k1_DIS_s100
                                 + 0.963249 w_k1_FDX_s100 + 1.109 w_k1_IBM_s100
                                 + 1.07123 w_k1_INTC_s100
                                 + 0.877029 w_k1_KO_s100
                                 + 1.06252 w_k1_MAT_s100
                                 + 0.858956 w_k1_MSFT_s100
                                 + 0.989148 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.904117 w_k1_ADM_s100
                                 + 0.773081 w_k1_AXP_s100
                                 + 0.901848 w_k1_DIS_s100
                                 + 0.911441 w_k1_FDX_s100
                                 + 0.681097 w_k1_IBM_s100
                                 + 1.11697 w_k1_INTC_s100
                                 + 0.829454 w_k1_KO_s100
                                 + 0.70454 w_k1_MAT_s100
                                 + 0.981097 w_k1_MSFT_s100
                                 + 0.84039 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.15726 w_k1_ADM_s100
                                 + 1.04055 w_k1_AXP_s100
                                 + 1.25826 w_k1_DIS_s100
                                 + 1.06406 w_k1_FDX_s100
                                 + 1.13736 w_k1_IBM_s100
                                 + 1.25717 w_k1_INTC_s100
                                 + 1.30405 w_k1_KO_s100
                                 + 0.989297 w_k1_MAT_s100
                                 + 1.14307 w_k1_MSFT_s100
                                 + 1.05878 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.913091 w_k1_ADM_s100
                                 + 0.812259 w_k1_AXP_s100
                                 + 0.887135 w_k1_DIS_s100
                                 + 0.837306 w_k1_FDX_s100
                                 + 0.924788 w_k1_IBM_s100
                                 + 1.04878 w_k1_INTC_s100
                                 + 1.04734 w_k1_KO_s100
                                 + 0.757637 w_k1_MAT_s100
                                 + 0.917864 w_k1_MSFT_s100
                                 + 0.842993 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.765823 w_k1_ADM_s100
                                 + 0.859989 w_k1_AXP_s100
                                 + 0.747482 w_k1_DIS_s100
                                 + 0.736164 w_k1_FDX_s100
                                 + 1.26907 w_k1_IBM_s100
                                 + 1.07016 w_k1_INTC_s100
                                 + 1.07671 w_k1_KO_s100 + 1.29973 w_k1_MAT_s100
                                 + 0.772059 w_k1_MSFT_s100
                                 + 0.933025 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.36495 w_k1_ADM_s100
                                 + 1.76232 w_k1_AXP_s100
                                 + 1.63539 w_k1_DIS_s100
                                 + 1.34024 w_k1_FDX_s100
                                 + 1.45381 w_k1_IBM_s100
                                 + 1.44152 w_k1_INTC_s100
                                 + 1.43859 w_k1_KO_s100 + 1.78561 w_k1_MAT_s100
                                 + 1.42065 w_k1_MSFT_s100
                                 + 1.10736 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.48224 w_k1_ADM_s100
                                 + 1.3507 w_k1_AXP_s100 + 1.21721 w_k1_DIS_s100
                                 + 1.24654 w_k1_FDX_s100
                                 + 0.933403 w_k1_IBM_s100
                                 + 1.33318 w_k1_INTC_s100
                                 + 1.17187 w_k1_KO_s100 + 0.93375 w_k1_MAT_s100
                                 + 1.2057 w_k1_MSFT_s100 + 1.03816 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.982006 w_k1_ADM_s100
                                 + 0.937296 w_k1_AXP_s100
                                 + 0.944996 w_k1_DIS_s100
                                 + 1.02754 w_k1_FDX_s100
                                 + 0.752813 w_k1_IBM_s100
                                 + 1.25042 w_k1_INTC_s100
                                 + 0.930536 w_k1_KO_s100
                                 + 0.693348 w_k1_MAT_s100
                                 + 1.119 w_k1_MSFT_s100 + 0.884014 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.907311 w_k1_ADM_s101
                                 + 1.02247 w_k1_AXP_s101
                                 + 0.962323 w_k1_DIS_s101
                                 + 0.969266 w_k1_FDX_s101
                                 + 0.936997 w_k1_IBM_s101
                                 + 1.12111 w_k1_INTC_s101
                                 + 1.13171 w_k1_KO_s101
                                 + 0.841305 w_k1_MAT_s101
                                 + 0.953555 w_k1_MSFT_s101
                                 + 0.925697 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.938645 w_k1_ADM_s101
                                 + 0.870347 w_k1_AXP_s101
                                 + 0.80002 w_k1_DIS_s101
                                 + 0.899465 w_k1_FDX_s101
                                 + 0.752895 w_k1_IBM_s101
                                 + 0.899878 w_k1_INTC_s101
                                 + 0.93466 w_k1_KO_s101
                                 + 0.813072 w_k1_MAT_s101
                                 + 0.945714 w_k1_MSFT_s101
                                 + 0.97011 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.34877 w_k1_ADM_s101
                                 + 1.34207 w_k1_AXP_s101
                                 + 1.51314 w_k1_DIS_s101
                                 + 1.33825 w_k1_FDX_s101
                                 + 0.91116 w_k1_IBM_s101
                                 + 1.09321 w_k1_INTC_s101
                                 + 1.26099 w_k1_KO_s101
                                 + 0.941956 w_k1_MAT_s101
                                 + 1.17689 w_k1_MSFT_s101
                                 + 1.17243 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.04693 w_k1_ADM_s101
                                 + 1.11064 w_k1_AXP_s101
                                 + 1.17573 w_k1_DIS_s101
                                 + 1.04328 w_k1_FDX_s101
                                 + 1.29725 w_k1_IBM_s101
                                 + 1.15193 w_k1_INTC_s101
                                 + 1.43091 w_k1_KO_s101
                                 + 0.780373 w_k1_MAT_s101
                                 + 1.04125 w_k1_MSFT_s101
                                 + 1.07933 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.0207 w_k1_ADM_s101
                                 + 1.13058 w_k1_AXP_s101
                                 + 1.20111 w_k1_DIS_s101
                                 + 0.966082 w_k1_FDX_s101
                                 + 1.49406 w_k1_IBM_s101
                                 + 1.00316 w_k1_INTC_s101
                                 + 1.39634 w_k1_KO_s101 + 1.25292 w_k1_MAT_s101
                                 + 1.02864 w_k1_MSFT_s101
                                 + 1.20119 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.783808 w_k1_ADM_s101
                                 + 0.78981 w_k1_AXP_s101
                                 + 0.724369 w_k1_DIS_s101
                                 + 0.769572 w_k1_FDX_s101
                                 + 1.11623 w_k1_IBM_s101
                                 + 0.826499 w_k1_INTC_s101
                                 + 1.04953 w_k1_KO_s101
                                 + 0.819235 w_k1_MAT_s101
                                 + 0.743733 w_k1_MSFT_s101
                                 + 0.831596 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.29392 w_k1_ADM_s101
                                 + 1.10976 w_k1_AXP_s101
                                 + 1.08975 w_k1_DIS_s101
                                 + 1.09847 w_k1_FDX_s101
                                 + 1.06907 w_k1_IBM_s101
                                 + 1.12548 w_k1_INTC_s101
                                 + 1.21164 w_k1_KO_s101
                                 + 0.908517 w_k1_MAT_s101
                                 + 0.995216 w_k1_MSFT_s101
                                 + 1.07344 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.45151 w_k1_ADM_s101
                                 + 1.66355 w_k1_AXP_s101
                                 + 1.74893 w_k1_DIS_s101
                                 + 1.30131 w_k1_FDX_s101
                                 + 1.51879 w_k1_IBM_s101
                                 + 1.3809 w_k1_INTC_s101 + 1.5349 w_k1_KO_s101
                                 + 1.3894 w_k1_MAT_s101
                                 + 1.35871 w_k1_MSFT_s101
                                 + 1.25707 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.993069 w_k1_ADM_s102
                                 + 1.15848 w_k1_AXP_s102
                                 + 1.11901 w_k1_DIS_s102 + 1.1526 w_k1_FDX_s102
                                 + 0.952768 w_k1_IBM_s102
                                 + 1.06884 w_k1_INTC_s102
                                 + 1.05489 w_k1_KO_s102
                                 + 0.848826 w_k1_MAT_s102
                                 + 1.02456 w_k1_MSFT_s102
                                 + 1.10407 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.694015 w_k1_ADM_s102
                                 + 1.03371 w_k1_AXP_s102 + 1.1164 w_k1_DIS_s102
                                 + 0.961479 w_k1_FDX_s102
                                 + 1.37637 w_k1_IBM_s102
                                 + 1.12853 w_k1_INTC_s102
                                 + 1.10715 w_k1_KO_s102 + 1.14835 w_k1_MAT_s102
                                 + 0.88881 w_k1_MSFT_s102
                                 + 1.00154 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.916631 w_k1_ADM_s102
                                 + 1.69298 w_k1_AXP_s102
                                 + 1.35481 w_k1_DIS_s102
                                 + 1.21376 w_k1_FDX_s102
                                 + 1.37629 w_k1_IBM_s102
                                 + 1.25389 w_k1_INTC_s102
                                 + 1.16473 w_k1_KO_s102 + 1.7215 w_k1_MAT_s102
                                 + 1.28024 w_k1_MSFT_s102
                                 + 1.13038 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.944513 w_k1_ADM_s102
                                 + 1.0808 w_k1_AXP_s102 + 1.78182 w_k1_DIS_s102
                                 + 1.28824 w_k1_FDX_s102
                                 + 1.32167 w_k1_IBM_s102
                                 + 1.39712 w_k1_INTC_s102
                                 + 1.41206 w_k1_KO_s102 + 1.08821 w_k1_MAT_s102
                                 + 1.36902 w_k1_MSFT_s102
                                 + 1.28548 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.992561 w_k1_ADM_s102
                                 + 0.973144 w_k1_AXP_s102
                                 + 1.09382 w_k1_DIS_s102
                                 + 0.968269 w_k1_FDX_s102
                                 + 1.01524 w_k1_IBM_s102
                                 + 0.967723 w_k1_INTC_s102
                                 + 1.07305 w_k1_KO_s102
                                 + 0.862782 w_k1_MAT_s102
                                 + 0.970996 w_k1_MSFT_s102
                                 + 1.0748 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.590397 w_k1_ADM_s102
                                 + 0.774772 w_k1_AXP_s102
                                 + 0.616834 w_k1_DIS_s102
                                 + 0.78802 w_k1_FDX_s102
                                 + 0.885518 w_k1_IBM_s102
                                 + 0.830575 w_k1_INTC_s102
                                 + 0.795502 w_k1_KO_s102
                                 + 0.804467 w_k1_MAT_s102
                                 + 0.778964 w_k1_MSFT_s102
                                 + 0.921346 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.956068 w_k1_ADM_s102
                                 + 0.913402 w_k1_AXP_s102
                                 + 1.01699 w_k1_DIS_s102
                                 + 1.10776 w_k1_FDX_s102
                                 + 0.720367 w_k1_IBM_s102
                                 + 1.14119 w_k1_INTC_s102
                                 + 0.885634 w_k1_KO_s102
                                 + 0.514347 w_k1_MAT_s102
                                 + 1.07783 w_k1_MSFT_s102
                                 + 1.03911 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.06195 w_k1_ADM_s102
                                 + 1.20591 w_k1_AXP_s102 + 1.4909 w_k1_DIS_s102
                                 + 1.40899 w_k1_FDX_s102
                                 + 0.856767 w_k1_IBM_s102
                                 + 1.23389 w_k1_INTC_s102
                                 + 1.14422 w_k1_KO_s102 + 0.85322 w_k1_MAT_s102
                                 + 1.16351 w_k1_MSFT_s102
                                 + 1.23824 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.27198 w_k1_ADM_s103
                                 + 1.46793 w_k1_AXP_s103
                                 + 1.00224 w_k1_DIS_s103
                                 + 1.20207 w_k1_FDX_s103
                                 + 0.980577 w_k1_IBM_s103
                                 + 1.19386 w_k1_INTC_s103 + 1.0664 w_k1_KO_s103
                                 + 1.20642 w_k1_MAT_s103
                                 + 1.13783 w_k1_MSFT_s103
                                 + 1.16485 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.13865 w_k1_ADM_s103
                                 + 0.933 w_k1_AXP_s103 + 0.979846 w_k1_DIS_s103
                                 + 1.03396 w_k1_FDX_s103
                                 + 0.757314 w_k1_IBM_s103
                                 + 0.834118 w_k1_INTC_s103
                                 + 1.00697 w_k1_KO_s103
                                 + 0.856866 w_k1_MAT_s103
                                 + 1.05735 w_k1_MSFT_s103
                                 + 1.03099 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.87025 w_k1_ADM_s103
                                 + 0.770655 w_k1_AXP_s103
                                 + 0.673243 w_k1_DIS_s103
                                 + 0.809972 w_k1_FDX_s103
                                 + 0.85142 w_k1_IBM_s103
                                 + 0.840122 w_k1_INTC_s103
                                 + 0.860702 w_k1_KO_s103
                                 + 0.71098 w_k1_MAT_s103
                                 + 0.880554 w_k1_MSFT_s103
                                 + 0.972706 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.93796 w_k1_ADM_s103
                                 + 0.981492 w_k1_AXP_s103
                                 + 1.07092 w_k1_DIS_s103
                                 + 0.862354 w_k1_FDX_s103
                                 + 1.4115 w_k1_IBM_s103
                                 + 0.962671 w_k1_INTC_s103
                                 + 1.22547 w_k1_KO_s103 + 1.18257 w_k1_MAT_s103
                                 + 0.956598 w_k1_MSFT_s103
                                 + 1.02831 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.919451 w_k1_ADM_s103
                                 + 1.26704 w_k1_AXP_s103
                                 + 1.23724 w_k1_DIS_s103
                                 + 1.02395 w_k1_FDX_s103
                                 + 1.38104 w_k1_IBM_s103
                                 + 1.05227 w_k1_INTC_s103
                                 + 1.22207 w_k1_KO_s103 + 1.59539 w_k1_MAT_s103
                                 + 1.12727 w_k1_MSFT_s103
                                 + 1.10334 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.1591 w_k1_ADM_s103
                                 + 1.09447 w_k1_AXP_s103
                                 + 1.22308 w_k1_DIS_s103
                                 + 1.17381 w_k1_FDX_s103
                                 + 0.895271 w_k1_IBM_s103
                                 + 1.17323 w_k1_INTC_s103
                                 + 1.05584 w_k1_KO_s103 + 0.87524 w_k1_MAT_s103
                                 + 1.13723 w_k1_MSFT_s103 + 1.1063 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.989611 w_k1_ADM_s103
                                 + 0.835333 w_k1_AXP_s103
                                 + 0.726206 w_k1_DIS_s103
                                 + 0.973144 w_k1_FDX_s103
                                 + 0.547429 w_k1_IBM_s103
                                 + 0.923958 w_k1_INTC_s103
                                 + 0.748788 w_k1_KO_s103
                                 + 0.502673 w_k1_MAT_s103
                                 + 0.909209 w_k1_MSFT_s103
                                 + 1.02144 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.51153 w_k1_ADM_s103
                                 + 1.79394 w_k1_AXP_s103
                                 + 1.58723 w_k1_DIS_s103
                                 + 1.43541 w_k1_FDX_s103
                                 + 1.51328 w_k1_IBM_s103
                                 + 1.45215 w_k1_INTC_s103 + 1.4495 w_k1_KO_s103
                                 + 1.95643 w_k1_MAT_s103
                                 + 1.31138 w_k1_MSFT_s103
                                 + 1.31768 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.803543 w_k1_ADM_s104
                                 + 0.462965 w_k1_AXP_s104
                                 + 0.644512 w_k1_DIS_s104
                                 + 0.723371 w_k1_FDX_s104
                                 + 0.566401 w_k1_IBM_s104
                                 + 0.874631 w_k1_INTC_s104
                                 + 0.626036 w_k1_KO_s104
                                 + 0.429412 w_k1_MAT_s104
                                 + 0.881784 w_k1_MSFT_s104
                                 + 0.746537 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.70478 w_k1_ADM_s104
                                 + 0.878527 w_k1_AXP_s104
                                 + 0.615392 w_k1_DIS_s104
                                 + 0.799287 w_k1_FDX_s104
                                 + 0.927684 w_k1_IBM_s104
                                 + 0.952211 w_k1_INTC_s104
                                 + 0.817474 w_k1_KO_s104
                                 + 1.03513 w_k1_MAT_s104
                                 + 0.808403 w_k1_MSFT_s104
                                 + 0.869597 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.345 w_k1_ADM_s104
                                 + 1.40293 w_k1_AXP_s104
                                 + 1.78863 w_k1_DIS_s104
                                 + 1.30233 w_k1_FDX_s104
                                 + 1.31976 w_k1_IBM_s104
                                 + 1.24257 w_k1_INTC_s104
                                 + 1.21429 w_k1_KO_s104 + 1.31096 w_k1_MAT_s104
                                 + 1.29872 w_k1_MSFT_s104
                                 + 1.23263 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.856184 w_k1_ADM_s104
                                 + 1.24935 w_k1_AXP_s104
                                 + 1.09889 w_k1_DIS_s104
                                 + 1.06878 w_k1_FDX_s104
                                 + 1.37114 w_k1_IBM_s104
                                 + 1.17361 w_k1_INTC_s104
                                 + 0.939612 w_k1_KO_s104
                                 + 1.48569 w_k1_MAT_s104
                                 + 1.05402 w_k1_MSFT_s104
                                 + 1.04596 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.980207 w_k1_ADM_s104
                                 + 0.878049 w_k1_AXP_s104
                                 + 1.22583 w_k1_DIS_s104 + 1.0507 w_k1_FDX_s104
                                 + 0.97212 w_k1_IBM_s104
                                 + 1.18906 w_k1_INTC_s104
                                 + 0.956159 w_k1_KO_s104
                                 + 1.06935 w_k1_MAT_s104
                                 + 1.15804 w_k1_MSFT_s104
                                 + 1.01408 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.859779 w_k1_ADM_s104
                                 + 0.696918 w_k1_AXP_s104
                                 + 0.900147 w_k1_DIS_s104
                                 + 0.949851 w_k1_FDX_s104
                                 + 0.695629 w_k1_IBM_s104
                                 + 1.01456 w_k1_INTC_s104
                                 + 0.865892 w_k1_KO_s104
                                 + 0.62146 w_k1_MAT_s104
                                 + 1.11735 w_k1_MSFT_s104
                                 + 0.983232 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.874846 w_k1_ADM_s104
                                 + 0.92153 w_k1_AXP_s104
                                 + 1.05419 w_k1_DIS_s104
                                 + 0.822482 w_k1_FDX_s104
                                 + 1.34166 w_k1_IBM_s104
                                 + 0.956655 w_k1_INTC_s104
                                 + 1.01811 w_k1_KO_s104 + 1.03069 w_k1_MAT_s104
                                 + 0.983609 w_k1_MSFT_s104
                                 + 1.04912 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.23234 w_k1_ADM_s104
                                 + 1.08497 w_k1_AXP_s104
                                 + 1.23075 w_k1_DIS_s104
                                 + 1.21902 w_k1_FDX_s104
                                 + 0.83871 w_k1_IBM_s104
                                 + 1.33882 w_k1_INTC_s104
                                 + 0.881628 w_k1_KO_s104
                                 + 0.554244 w_k1_MAT_s104
                                 + 1.20516 w_k1_MSFT_s104
                                 + 1.06607 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.51939 w_k1_ADM_s105
                                 + 1.29829 w_k1_AXP_s105
                                 + 1.13252 w_k1_DIS_s105
                                 + 1.16391 w_k1_FDX_s105
                                 + 0.979846 w_k1_IBM_s105
                                 + 1.01973 w_k1_INTC_s105
                                 + 0.977138 w_k1_KO_s105
                                 + 0.999658 w_k1_MAT_s105
                                 + 0.999681 w_k1_MSFT_s105
                                 + 0.946471 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.825741 w_k1_ADM_s105
                                 + 0.519773 w_k1_AXP_s105
                                 + 0.662987 w_k1_DIS_s105
                                 + 0.716877 w_k1_FDX_s105
                                 + 0.873807 w_k1_IBM_s105
                                 + 0.6714 w_k1_INTC_s105
                                 + 0.828923 w_k1_KO_s105
                                 + 0.812178 w_k1_MAT_s105
                                 + 0.788316 w_k1_MSFT_s105
                                 + 0.807392 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.13878 w_k1_ADM_s105
                                 + 0.879363 w_k1_AXP_s105
                                 + 0.848426 w_k1_DIS_s105
                                 + 0.952722 w_k1_FDX_s105
                                 + 0.70736 w_k1_IBM_s105
                                 + 1.12452 w_k1_INTC_s105
                                 + 0.830117 w_k1_KO_s105
                                 + 0.711952 w_k1_MAT_s105
                                 + 0.931932 w_k1_MSFT_s105
                                 + 0.886865 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.17966 w_k1_ADM_s105
                                 + 1.18003 w_k1_AXP_s105
                                 + 1.42691 w_k1_DIS_s105
                                 + 1.08488 w_k1_FDX_s105 + 1.6398 w_k1_IBM_s105
                                 + 1.20405 w_k1_INTC_s105
                                 + 1.41821 w_k1_KO_s105 + 1.34163 w_k1_MAT_s105
                                 + 0.972964 w_k1_MSFT_s105
                                 + 1.14955 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.788228 w_k1_ADM_s105
                                 + 0.74395 w_k1_AXP_s105
                                 + 0.753916 w_k1_DIS_s105
                                 + 0.791293 w_k1_FDX_s105
                                 + 1.02517 w_k1_IBM_s105
                                 + 0.961543 w_k1_INTC_s105
                                 + 0.870566 w_k1_KO_s105
                                 + 0.947029 w_k1_MAT_s105
                                 + 0.788834 w_k1_MSFT_s105
                                 + 0.834192 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.49384 w_k1_ADM_s105
                                 + 1.20975 w_k1_AXP_s105
                                 + 1.31402 w_k1_DIS_s105
                                 + 1.39815 w_k1_FDX_s105
                                 + 0.808446 w_k1_IBM_s105
                                 + 1.25496 w_k1_INTC_s105
                                 + 0.968625 w_k1_KO_s105
                                 + 0.667526 w_k1_MAT_s105
                                 + 1.09461 w_k1_MSFT_s105
                                 + 1.05867 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.11694 w_k1_ADM_s105
                                 + 0.867462 w_k1_AXP_s105
                                 + 1.1111 w_k1_DIS_s105 + 1.03119 w_k1_FDX_s105
                                 + 1.14381 w_k1_IBM_s105
                                 + 0.967221 w_k1_INTC_s105
                                 + 1.22539 w_k1_KO_s105 + 1.05174 w_k1_MAT_s105
                                 + 1.01758 w_k1_MSFT_s105
                                 + 1.08688 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.02962 w_k1_ADM_s105
                                 + 0.7489 w_k1_AXP_s105
                                 + 0.570489 w_k1_DIS_s105
                                 + 0.870132 w_k1_FDX_s105
                                 + 0.630597 w_k1_IBM_s105
                                 + 0.801455 w_k1_INTC_s105
                                 + 0.594464 w_k1_KO_s105
                                 + 0.578623 w_k1_MAT_s105
                                 + 0.788801 w_k1_MSFT_s105
                                 + 0.749222 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.787075 w_k1_ADM_s106
                                 + 0.938837 w_k1_AXP_s106
                                 + 0.854062 w_k1_DIS_s106
                                 + 0.884757 w_k1_FDX_s106
                                 + 1.06586 w_k1_IBM_s106
                                 + 0.852987 w_k1_INTC_s106
                                 + 0.875809 w_k1_KO_s106
                                 + 0.951195 w_k1_MAT_s106
                                 + 0.915911 w_k1_MSFT_s106
                                 + 0.944167 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.11879 w_k1_ADM_s106
                                 + 1.29562 w_k1_AXP_s106
                                 + 1.28099 w_k1_DIS_s106
                                 + 1.11743 w_k1_FDX_s106 + 1.6756 w_k1_IBM_s106
                                 + 1.02213 w_k1_INTC_s106
                                 + 1.20968 w_k1_KO_s106 + 1.67017 w_k1_MAT_s106
                                 + 1.13067 w_k1_MSFT_s106
                                 + 1.21262 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.26993 w_k1_ADM_s106
                                 + 1.30661 w_k1_AXP_s106
                                 + 1.35799 w_k1_DIS_s106 + 1.3848 w_k1_FDX_s106
                                 + 0.921489 w_k1_IBM_s106
                                 + 1.21116 w_k1_INTC_s106
                                 + 1.11464 w_k1_KO_s106 + 1.03817 w_k1_MAT_s106
                                 + 1.23718 w_k1_MSFT_s106
                                 + 1.26837 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.899042 w_k1_ADM_s106
                                 + 1.49091 w_k1_AXP_s106
                                 + 1.17449 w_k1_DIS_s106 + 1.073 w_k1_FDX_s106
                                 + 1.18866 w_k1_IBM_s106
                                 + 1.07365 w_k1_INTC_s106
                                 + 1.03764 w_k1_KO_s106 + 1.15101 w_k1_MAT_s106
                                 + 1.18454 w_k1_MSFT_s106
                                 + 1.09833 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.914859 w_k1_ADM_s106
                                 + 1.1231 w_k1_AXP_s106 + 1.39603 w_k1_DIS_s106
                                 + 1.15602 w_k1_FDX_s106
                                 + 1.02561 w_k1_IBM_s106
                                 + 1.08616 w_k1_INTC_s106
                                 + 1.09278 w_k1_KO_s106 + 1.04034 w_k1_MAT_s106
                                 + 1.27081 w_k1_MSFT_s106
                                 + 1.02148 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.869442 w_k1_ADM_s106
                                 + 0.885602 w_k1_AXP_s106
                                 + 0.867492 w_k1_DIS_s106
                                 + 0.898441 w_k1_FDX_s106
                                 + 0.826787 w_k1_IBM_s106
                                 + 0.863328 w_k1_INTC_s106
                                 + 0.838666 w_k1_KO_s106
                                 + 0.975588 w_k1_MAT_s106
                                 + 1.0253 w_k1_MSFT_s106
                                 + 0.986585 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.861921 w_k1_ADM_s106
                                 + 0.854665 w_k1_AXP_s106
                                 + 0.775283 w_k1_DIS_s106
                                 + 0.916271 w_k1_FDX_s106
                                 + 0.686376 w_k1_IBM_s106
                                 + 0.834848 w_k1_INTC_s106
                                 + 0.72055 w_k1_KO_s106
                                 + 0.610494 w_k1_MAT_s106
                                 + 0.858944 w_k1_MSFT_s106
                                 + 0.83706 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.771625 w_k1_ADM_s106
                                 + 0.972669 w_k1_AXP_s106
                                 + 0.871777 w_k1_DIS_s106
                                 + 0.787723 w_k1_FDX_s106
                                 + 1.35963 w_k1_IBM_s106
                                 + 0.843598 w_k1_INTC_s106
                                 + 0.946312 w_k1_KO_s106
                                 + 1.41716 w_k1_MAT_s106
                                 + 0.874875 w_k1_MSFT_s106
                                 + 0.891422 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.921574 w_k1_ADM_s107
                                 + 0.918411 w_k1_AXP_s107
                                 + 0.946475 w_k1_DIS_s107
                                 + 0.830768 w_k1_FDX_s107
                                 + 1.24886 w_k1_IBM_s107
                                 + 1.01194 w_k1_INTC_s107
                                 + 0.999115 w_k1_KO_s107
                                 + 1.04607 w_k1_MAT_s107
                                 + 0.913585 w_k1_MSFT_s107
                                 + 0.976711 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.0598 w_k1_ADM_s107
                                 + 1.17884 w_k1_AXP_s107
                                 + 1.17094 w_k1_DIS_s107
                                 + 1.14683 w_k1_FDX_s107
                                 + 0.860556 w_k1_IBM_s107
                                 + 1.06321 w_k1_INTC_s107
                                 + 0.939802 w_k1_KO_s107
                                 + 0.825321 w_k1_MAT_s107
                                 + 1.01353 w_k1_MSFT_s107
                                 + 0.947491 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.0925 w_k1_ADM_s107
                                 + 0.997116 w_k1_AXP_s107
                                 + 1.0876 w_k1_DIS_s107
                                 + 0.958256 w_k1_FDX_s107
                                 + 1.11377 w_k1_IBM_s107
                                 + 0.975084 w_k1_INTC_s107
                                 + 1.06422 w_k1_KO_s107 + 1.11418 w_k1_MAT_s107
                                 + 0.891424 w_k1_MSFT_s107
                                 + 0.957953 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.972696 w_k1_ADM_s107
                                 + 0.911449 w_k1_AXP_s107
                                 + 0.911805 w_k1_DIS_s107
                                 + 0.985398 w_k1_FDX_s107
                                 + 0.637195 w_k1_IBM_s107
                                 + 0.965719 w_k1_INTC_s107
                                 + 0.738254 w_k1_KO_s107
                                 + 0.542627 w_k1_MAT_s107
                                 + 0.910221 w_k1_MSFT_s107
                                 + 0.899898 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.18072 w_k1_ADM_s107
                                 + 1.28728 w_k1_AXP_s107
                                 + 1.39971 w_k1_DIS_s107
                                 + 1.05524 w_k1_FDX_s107 + 1.731 w_k1_IBM_s107
                                 + 1.09317 w_k1_INTC_s107
                                 + 1.21882 w_k1_KO_s107 + 1.70111 w_k1_MAT_s107
                                 + 0.911227 w_k1_MSFT_s107
                                 + 1.17702 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.01377 w_k1_ADM_s107
                                 + 0.905298 w_k1_AXP_s107
                                 + 0.820048 w_k1_DIS_s107
                                 + 0.876829 w_k1_FDX_s107
                                 + 0.986762 w_k1_IBM_s107
                                 + 0.795445 w_k1_INTC_s107
                                 + 1.00035 w_k1_KO_s107 + 1.14619 w_k1_MAT_s107
                                 + 0.80284 w_k1_MSFT_s107
                                 + 0.928129 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.937736 w_k1_ADM_s107
                                 + 1.43359 w_k1_AXP_s107
                                 + 1.32822 w_k1_DIS_s107
                                 + 1.19726 w_k1_FDX_s107
                                 + 1.29257 w_k1_IBM_s107
                                 + 1.12408 w_k1_INTC_s107 + 1.0783 w_k1_KO_s107
                                 + 1.68134 w_k1_MAT_s107
                                 + 1.06673 w_k1_MSFT_s107
                                 + 1.05313 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.83521 w_k1_ADM_s107
                                 + 0.820787 w_k1_AXP_s107
                                 + 0.683187 w_k1_DIS_s107
                                 + 0.769529 w_k1_FDX_s107
                                 + 0.803717 w_k1_IBM_s107
                                 + 0.845493 w_k1_INTC_s107
                                 + 0.640414 w_k1_KO_s107
                                 + 0.598242 w_k1_MAT_s107
                                 + 0.663324 w_k1_MSFT_s107
                                 + 0.792941 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.919349 w_k1_ADM_s110
                                 + 0.967432 w_k1_AXP_s110
                                 + 0.631815 w_k1_DIS_s110
                                 + 0.909469 w_k1_FDX_s110
                                 + 0.595433 w_k1_IBM_s110
                                 + 1.04631 w_k1_INTC_s110
                                 + 0.657819 w_k1_KO_s110
                                 + 0.55674 w_k1_MAT_s110
                                 + 0.826621 w_k1_MSFT_s110
                                 + 0.886944 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.37473 w_k1_ADM_s110
                                 + 1.48891 w_k1_AXP_s110
                                 + 1.76005 w_k1_DIS_s110
                                 + 1.41515 w_k1_FDX_s110 + 1.3898 w_k1_IBM_s110
                                 + 1.47087 w_k1_INTC_s110
                                 + 1.64657 w_k1_KO_s110 + 1.95431 w_k1_MAT_s110
                                 + 1.63082 w_k1_MSFT_s110
                                 + 1.53075 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.07126 w_k1_ADM_s110
                                 + 0.901165 w_k1_AXP_s110
                                 + 0.901359 w_k1_DIS_s110
                                 + 1.04915 w_k1_FDX_s110
                                 + 0.583763 w_k1_IBM_s110
                                 + 0.98364 w_k1_INTC_s110
                                 + 0.787973 w_k1_KO_s110
                                 + 0.880009 w_k1_MAT_s110
                                 + 1.06478 w_k1_MSFT_s110
                                 + 0.97363 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.970401 w_k1_ADM_s110
                                 + 1.06269 w_k1_AXP_s110
                                 + 1.27495 w_k1_DIS_s110
                                 + 1.04146 w_k1_FDX_s110 + 1.5204 w_k1_IBM_s110
                                 + 0.970979 w_k1_INTC_s110
                                 + 1.22673 w_k1_KO_s110 + 1.50122 w_k1_MAT_s110
                                 + 1.07662 w_k1_MSFT_s110
                                 + 1.13818 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.06414 w_k1_ADM_s110
                                 + 0.926243 w_k1_AXP_s110
                                 + 0.914515 w_k1_DIS_s110
                                 + 0.901617 w_k1_FDX_s110
                                 + 1.27015 w_k1_IBM_s110
                                 + 0.947895 w_k1_INTC_s110
                                 + 1.05181 w_k1_KO_s110 + 1.28005 w_k1_MAT_s110
                                 + 0.942386 w_k1_MSFT_s110
                                 + 1.03992 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.07134 w_k1_ADM_s110
                                 + 0.803295 w_k1_AXP_s110
                                 + 0.831453 w_k1_DIS_s110
                                 + 0.906488 w_k1_FDX_s110
                                 + 0.722655 w_k1_IBM_s110
                                 + 0.998411 w_k1_INTC_s110
                                 + 0.820607 w_k1_KO_s110
                                 + 0.576496 w_k1_MAT_s110
                                 + 0.936953 w_k1_MSFT_s110
                                 + 0.912052 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.859533 w_k1_ADM_s110
                                 + 0.562548 w_k1_AXP_s110
                                 + 0.647647 w_k1_DIS_s110
                                 + 0.707008 w_k1_FDX_s110
                                 + 0.838177 w_k1_IBM_s110
                                 + 0.862193 w_k1_INTC_s110
                                 + 0.91959 w_k1_KO_s110
                                 + 0.699916 w_k1_MAT_s110
                                 + 0.887658 w_k1_MSFT_s110
                                 + 0.909575 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.16257 w_k1_ADM_s110
                                 + 1.15088 w_k1_AXP_s110 + 1.2177 w_k1_DIS_s110
                                 + 1.20672 w_k1_FDX_s110
                                 + 0.980187 w_k1_IBM_s110
                                 + 1.09319 w_k1_INTC_s110
                                 + 1.11241 w_k1_KO_s110
                                 + 0.981584 w_k1_MAT_s110
                                 + 1.11747 w_k1_MSFT_s110
                                 + 1.06166 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.12938 w_k1_ADM_s111
                                 + 1.01964 w_k1_AXP_s111
                                 + 1.06499 w_k1_DIS_s111
                                 + 1.05077 w_k1_FDX_s111
                                 + 1.02852 w_k1_IBM_s111
                                 + 0.882717 w_k1_INTC_s111
                                 + 1.06571 w_k1_KO_s111 + 1.18107 w_k1_MAT_s111
                                 + 1.08331 w_k1_MSFT_s111
                                 + 0.97071 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.689306 w_k1_ADM_s111
                                 + 0.778499 w_k1_AXP_s111
                                 + 0.602109 w_k1_DIS_s111
                                 + 0.697401 w_k1_FDX_s111
                                 + 1.11091 w_k1_IBM_s111
                                 + 0.794837 w_k1_INTC_s111
                                 + 0.856435 w_k1_KO_s111
                                 + 1.09659 w_k1_MAT_s111
                                 + 0.698926 w_k1_MSFT_s111
                                 + 0.837964 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.13033 w_k1_ADM_s111
                                 + 1.24859 w_k1_AXP_s111
                                 + 1.41935 w_k1_DIS_s111
                                 + 1.03105 w_k1_FDX_s111 + 1.7999 w_k1_IBM_s111
                                 + 1.21329 w_k1_INTC_s111
                                 + 1.57167 w_k1_KO_s111 + 1.65868 w_k1_MAT_s111
                                 + 1.14988 w_k1_MSFT_s111
                                 + 1.16887 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.952588 w_k1_ADM_s111
                                 + 1.08168 w_k1_AXP_s111
                                 + 1.09413 w_k1_DIS_s111
                                 + 0.903139 w_k1_FDX_s111
                                 + 1.31886 w_k1_IBM_s111
                                 + 1.01565 w_k1_INTC_s111
                                 + 1.16347 w_k1_KO_s111 + 1.52324 w_k1_MAT_s111
                                 + 0.968952 w_k1_MSFT_s111
                                 + 1.06047 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.01847 w_k1_ADM_s111
                                 + 0.893212 w_k1_AXP_s111
                                 + 0.975604 w_k1_DIS_s111
                                 + 0.909325 w_k1_FDX_s111
                                 + 1.03398 w_k1_IBM_s111
                                 + 0.926735 w_k1_INTC_s111
                                 + 1.01441 w_k1_KO_s111 + 1.11298 w_k1_MAT_s111
                                 + 1.06271 w_k1_MSFT_s111
                                 + 1.00396 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.98226 w_k1_ADM_s111
                                 + 1.60708 w_k1_AXP_s111
                                 + 1.56038 w_k1_DIS_s111 + 1.3379 w_k1_FDX_s111
                                 + 1.26074 w_k1_IBM_s111
                                 + 1.24537 w_k1_INTC_s111
                                 + 1.18077 w_k1_KO_s111 + 1.61489 w_k1_MAT_s111
                                 + 1.27431 w_k1_MSFT_s111
                                 + 1.18577 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.23936 w_k1_ADM_s111
                                 + 1.15169 w_k1_AXP_s111
                                 + 1.32139 w_k1_DIS_s111
                                 + 1.19404 w_k1_FDX_s111
                                 + 0.86968 w_k1_IBM_s111
                                 + 1.17467 w_k1_INTC_s111
                                 + 1.10232 w_k1_KO_s111
                                 + 0.976321 w_k1_MAT_s111
                                 + 1.19335 w_k1_MSFT_s111
                                 + 1.09502 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.846837 w_k1_ADM_s111
                                 + 0.870686 w_k1_AXP_s111
                                 + 0.860703 w_k1_DIS_s111
                                 + 0.90291 w_k1_FDX_s111
                                 + 0.716642 w_k1_IBM_s111
                                 + 0.866319 w_k1_INTC_s111
                                 + 0.931671 w_k1_KO_s111
                                 + 0.792571 w_k1_MAT_s111
                                 + 0.9569 w_k1_MSFT_s111
                                 + 0.878087 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.30309 w_k1_ADM_s112
                                 + 1.40347 w_k1_AXP_s112
                                 + 1.65266 w_k1_DIS_s112
                                 + 1.37793 w_k1_FDX_s112
                                 + 1.28701 w_k1_IBM_s112
                                 + 1.32443 w_k1_INTC_s112
                                 + 1.39033 w_k1_KO_s112 + 1.359 w_k1_MAT_s112
                                 + 1.30961 w_k1_MSFT_s112
                                 + 1.12822 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.46561 w_k1_ADM_s112
                                 + 1.43119 w_k1_AXP_s112
                                 + 1.27843 w_k1_DIS_s112
                                 + 1.21962 w_k1_FDX_s112 + 1.1407 w_k1_IBM_s112
                                 + 1.17403 w_k1_INTC_s112
                                 + 1.12796 w_k1_KO_s112 + 1.0776 w_k1_MAT_s112
                                 + 1.02726 w_k1_MSFT_s112
                                 + 1.14161 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.964166 w_k1_ADM_s112
                                 + 0.936016 w_k1_AXP_s112
                                 + 1.10595 w_k1_DIS_s112
                                 + 0.990811 w_k1_FDX_s112
                                 + 1.21109 w_k1_IBM_s112
                                 + 0.994898 w_k1_INTC_s112
                                 + 1.1626 w_k1_KO_s112 + 1.11133 w_k1_MAT_s112
                                 + 1.03754 w_k1_MSFT_s112
                                 + 1.09801 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.42306 w_k1_ADM_s112
                                 + 1.06648 w_k1_AXP_s112 + 1.1678 w_k1_DIS_s112
                                 + 1.15574 w_k1_FDX_s112
                                 + 0.79434 w_k1_IBM_s112
                                 + 0.994685 w_k1_INTC_s112
                                 + 0.953364 w_k1_KO_s112
                                 + 0.592079 w_k1_MAT_s112
                                 + 1.0734 w_k1_MSFT_s112 + 1.08 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.19504 w_k1_ADM_s112
                                 + 1.22408 w_k1_AXP_s112
                                 + 1.29407 w_k1_DIS_s112
                                 + 1.12129 w_k1_FDX_s112
                                 + 1.55627 w_k1_IBM_s112
                                 + 1.19464 w_k1_INTC_s112
                                 + 1.22099 w_k1_KO_s112 + 1.26156 w_k1_MAT_s112
                                 + 1.09927 w_k1_MSFT_s112 + 1.0899 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.9131 w_k1_ADM_s112
                                 + 1.05289 w_k1_AXP_s112
                                 + 0.853851 w_k1_DIS_s112
                                 + 0.902973 w_k1_FDX_s112
                                 + 1.47267 w_k1_IBM_s112
                                 + 0.841596 w_k1_INTC_s112
                                 + 0.933906 w_k1_KO_s112
                                 + 1.49754 w_k1_MAT_s112
                                 + 0.799741 w_k1_MSFT_s112
                                 + 0.998395 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.930755 w_k1_ADM_s112
                                 + 0.701385 w_k1_AXP_s112
                                 + 0.642472 w_k1_DIS_s112
                                 + 0.877136 w_k1_FDX_s112
                                 + 0.597632 w_k1_IBM_s112
                                 + 0.950981 w_k1_INTC_s112
                                 + 0.676283 w_k1_KO_s112
                                 + 0.442106 w_k1_MAT_s112
                                 + 0.787691 w_k1_MSFT_s112
                                 + 0.876621 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.07312 w_k1_ADM_s112
                                 + 0.963581 w_k1_AXP_s112
                                 + 0.762821 w_k1_DIS_s112
                                 + 0.954024 w_k1_FDX_s112
                                 + 0.904799 w_k1_IBM_s112
                                 + 0.95907 w_k1_INTC_s112
                                 + 0.872814 w_k1_KO_s112
                                 + 0.68765 w_k1_MAT_s112
                                 + 0.789228 w_k1_MSFT_s112
                                 + 0.896454 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.02722 w_k1_ADM_s113
                                 + 0.848565 w_k1_AXP_s113
                                 + 0.717011 w_k1_DIS_s113
                                 + 0.929428 w_k1_FDX_s113
                                 + 0.796455 w_k1_IBM_s113
                                 + 0.849846 w_k1_INTC_s113
                                 + 0.740925 w_k1_KO_s113
                                 + 0.981695 w_k1_MAT_s113
                                 + 0.891876 w_k1_MSFT_s113
                                 + 0.993663 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.794645 w_k1_ADM_s113
                                 + 0.640814 w_k1_AXP_s113
                                 + 0.533271 w_k1_DIS_s113
                                 + 0.740204 w_k1_FDX_s113
                                 + 0.774678 w_k1_IBM_s113
                                 + 0.692636 w_k1_INTC_s113
                                 + 0.648703 w_k1_KO_s113
                                 + 0.786562 w_k1_MAT_s113
                                 + 0.709869 w_k1_MSFT_s113
                                 + 0.868818 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.05493 w_k1_ADM_s113
                                 + 1.16236 w_k1_AXP_s113
                                 + 1.05554 w_k1_DIS_s113
                                 + 1.10178 w_k1_FDX_s113
                                 + 1.23345 w_k1_IBM_s113
                                 + 1.02916 w_k1_INTC_s113
                                 + 0.949114 w_k1_KO_s113
                                 + 1.59658 w_k1_MAT_s113
                                 + 1.0487 w_k1_MSFT_s113
                                 + 0.964341 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.52247 w_k1_ADM_s113
                                 + 1.43713 w_k1_AXP_s113
                                 + 2.08273 w_k1_DIS_s113
                                 + 1.78663 w_k1_FDX_s113
                                 + 1.07238 w_k1_IBM_s113
                                 + 1.59045 w_k1_INTC_s113
                                 + 1.31828 w_k1_KO_s113 + 1.18037 w_k1_MAT_s113
                                 + 1.57228 w_k1_MSFT_s113
                                 + 1.35433 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.949227 w_k1_ADM_s113
                                 + 0.814898 w_k1_AXP_s113
                                 + 1.05712 w_k1_DIS_s113
                                 + 0.903882 w_k1_FDX_s113
                                 + 1.35077 w_k1_IBM_s113
                                 + 0.941299 w_k1_INTC_s113
                                 + 1.13173 w_k1_KO_s113 + 1.30429 w_k1_MAT_s113
                                 + 0.992629 w_k1_MSFT_s113
                                 + 1.09199 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.803223 w_k1_ADM_s113
                                 + 0.982302 w_k1_AXP_s113
                                 + 1.12376 w_k1_DIS_s113
                                 + 1.16478 w_k1_FDX_s113
                                 + 0.82801 w_k1_IBM_s113
                                 + 1.22412 w_k1_INTC_s113
                                 + 0.93357 w_k1_KO_s113
                                 + 0.950351 w_k1_MAT_s113
                                 + 1.16436 w_k1_MSFT_s113
                                 + 1.05728 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.52903 w_k1_ADM_s113
                                 + 1.68099 w_k1_AXP_s113
                                 + 1.49897 w_k1_DIS_s113
                                 + 1.35422 w_k1_FDX_s113
                                 + 1.41549 w_k1_IBM_s113
                                 + 1.05167 w_k1_INTC_s113
                                 + 1.35556 w_k1_KO_s113 + 2.1542 w_k1_MAT_s113
                                 + 1.28897 w_k1_MSFT_s113
                                 + 1.28605 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.698372 w_k1_ADM_s113
                                 + 0.768337 w_k1_AXP_s113
                                 + 0.743369 w_k1_DIS_s113
                                 + 0.785232 w_k1_FDX_s113
                                 + 1.1652 w_k1_IBM_s113
                                 + 0.863355 w_k1_INTC_s113
                                 + 0.863776 w_k1_KO_s113
                                 + 1.28224 w_k1_MAT_s113
                                 + 0.816844 w_k1_MSFT_s113
                                 + 0.967866 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.904864 w_k1_ADM_s114
                                 + 1.0065 w_k1_AXP_s114
                                 + 0.989856 w_k1_DIS_s114
                                 + 0.999648 w_k1_FDX_s114
                                 + 0.881304 w_k1_IBM_s114
                                 + 0.967966 w_k1_INTC_s114
                                 + 0.816224 w_k1_KO_s114
                                 + 1.25472 w_k1_MAT_s114
                                 + 0.985433 w_k1_MSFT_s114
                                 + 0.866475 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.13901 w_k1_ADM_s114
                                 + 1.19819 w_k1_AXP_s114
                                 + 1.02148 w_k1_DIS_s114
                                 + 1.23439 w_k1_FDX_s114
                                 + 0.710574 w_k1_IBM_s114
                                 + 1.06529 w_k1_INTC_s114
                                 + 0.715783 w_k1_KO_s114
                                 + 0.876215 w_k1_MAT_s114
                                 + 1.07381 w_k1_MSFT_s114 + 0.9705 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.966057 w_k1_ADM_s114
                                 + 1.26359 w_k1_AXP_s114
                                 + 1.32737 w_k1_DIS_s114
                                 + 1.12546 w_k1_FDX_s114
                                 + 1.12343 w_k1_IBM_s114
                                 + 1.18976 w_k1_INTC_s114
                                 + 1.01473 w_k1_KO_s114 + 1.27725 w_k1_MAT_s114
                                 + 1.25535 w_k1_MSFT_s114
                                 + 1.09945 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.50049 w_k1_ADM_s114
                                 + 1.40381 w_k1_AXP_s114
                                 + 1.72565 w_k1_DIS_s114
                                 + 1.44389 w_k1_FDX_s114
                                 + 1.00989 w_k1_IBM_s114
                                 + 1.32561 w_k1_INTC_s114
                                 + 1.27524 w_k1_KO_s114 + 1.3653 w_k1_MAT_s114
                                 + 1.39125 w_k1_MSFT_s114
                                 + 1.20087 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.917915 w_k1_ADM_s114
                                 + 0.968597 w_k1_AXP_s114
                                 + 0.840669 w_k1_DIS_s114
                                 + 0.829002 w_k1_FDX_s114
                                 + 1.16262 w_k1_IBM_s114
                                 + 0.934131 w_k1_INTC_s114
                                 + 0.885969 w_k1_KO_s114
                                 + 1.31836 w_k1_MAT_s114
                                 + 0.878396 w_k1_MSFT_s114
                                 + 0.86258 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.11806 w_k1_ADM_s114
                                 + 1.25034 w_k1_AXP_s114
                                 + 1.65894 w_k1_DIS_s114
                                 + 1.32381 w_k1_FDX_s114
                                 + 1.60286 w_k1_IBM_s114
                                 + 1.32481 w_k1_INTC_s114
                                 + 1.33005 w_k1_KO_s114 + 1.21026 w_k1_MAT_s114
                                 + 1.26845 w_k1_MSFT_s114
                                 + 1.04704 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.988753 w_k1_ADM_s114
                                 + 0.897521 w_k1_AXP_s114
                                 + 0.688755 w_k1_DIS_s114
                                 + 0.823316 w_k1_FDX_s114
                                 + 0.784501 w_k1_IBM_s114
                                 + 0.789288 w_k1_INTC_s114
                                 + 0.644867 w_k1_KO_s114
                                 + 1.01897 w_k1_MAT_s114
                                 + 0.814749 w_k1_MSFT_s114
                                 + 0.856036 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.05034 w_k1_ADM_s114
                                 + 0.957223 w_k1_AXP_s114
                                 + 1.1259 w_k1_DIS_s114 + 1.0685 w_k1_FDX_s114
                                 + 1.00062 w_k1_IBM_s114
                                 + 1.02639 w_k1_INTC_s114
                                 + 1.09433 w_k1_KO_s114 + 1.36889 w_k1_MAT_s114
                                 + 0.985013 w_k1_MSFT_s114
                                 + 0.975396 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.27173 w_k1_ADM_s115
                                 + 1.4903 w_k1_AXP_s115 + 1.62956 w_k1_DIS_s115
                                 + 1.35652 w_k1_FDX_s115
                                 + 1.30278 w_k1_IBM_s115
                                 + 0.953148 w_k1_INTC_s115
                                 + 1.3612 w_k1_KO_s115 + 1.42127 w_k1_MAT_s115
                                 + 1.09096 w_k1_MSFT_s115
                                 + 1.21481 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.890231 w_k1_ADM_s115
                                 + 1.07579 w_k1_AXP_s115
                                 + 1.22382 w_k1_DIS_s115
                                 + 1.05898 w_k1_FDX_s115
                                 + 1.55369 w_k1_IBM_s115
                                 + 0.808997 w_k1_INTC_s115
                                 + 1.26434 w_k1_KO_s115 + 1.50251 w_k1_MAT_s115
                                 + 1.02727 w_k1_MSFT_s115
                                 + 1.18835 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.38026 w_k1_ADM_s115
                                 + 1.04693 w_k1_AXP_s115
                                 + 1.12995 w_k1_DIS_s115
                                 + 1.07882 w_k1_FDX_s115
                                 + 1.09437 w_k1_IBM_s115
                                 + 0.89515 w_k1_INTC_s115
                                 + 1.22094 w_k1_KO_s115 + 1.20246 w_k1_MAT_s115
                                 + 1.04101 w_k1_MSFT_s115
                                 + 1.10094 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.842467 w_k1_ADM_s115
                                 + 0.808134 w_k1_AXP_s115
                                 + 0.846768 w_k1_DIS_s115
                                 + 1.00762 w_k1_FDX_s115
                                 + 0.646757 w_k1_IBM_s115
                                 + 0.847125 w_k1_INTC_s115
                                 + 0.808046 w_k1_KO_s115
                                 + 0.66098 w_k1_MAT_s115
                                 + 0.946669 w_k1_MSFT_s115
                                 + 0.904621 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.833439 w_k1_ADM_s115
                                 + 1.04685 w_k1_AXP_s115
                                 + 0.804979 w_k1_DIS_s115
                                 + 0.939832 w_k1_FDX_s115
                                 + 0.819276 w_k1_IBM_s115
                                 + 0.808292 w_k1_INTC_s115
                                 + 0.794843 w_k1_KO_s115
                                 + 0.921494 w_k1_MAT_s115
                                 + 0.813594 w_k1_MSFT_s115
                                 + 0.910192 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.38905 w_k1_ADM_s115
                                 + 1.17123 w_k1_AXP_s115
                                 + 1.03819 w_k1_DIS_s115
                                 + 1.21519 w_k1_FDX_s115
                                 + 0.788583 w_k1_IBM_s115
                                 + 0.921156 w_k1_INTC_s115
                                 + 0.993596 w_k1_KO_s115
                                 + 0.775327 w_k1_MAT_s115
                                 + 0.991461 w_k1_MSFT_s115
                                 + 1.07128 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.01245 w_k1_ADM_s115
                                 + 0.896421 w_k1_AXP_s115
                                 + 0.928236 w_k1_DIS_s115
                                 + 0.941598 w_k1_FDX_s115
                                 + 1.03628 w_k1_IBM_s115
                                 + 0.859868 w_k1_INTC_s115
                                 + 0.843382 w_k1_KO_s115
                                 + 0.86956 w_k1_MAT_s115
                                 + 0.917571 w_k1_MSFT_s115
                                 + 0.913849 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.871731 w_k1_ADM_s115
                                 + 0.594487 w_k1_AXP_s115
                                 + 0.610706 w_k1_DIS_s115
                                 + 0.718359 w_k1_FDX_s115
                                 + 0.762349 w_k1_IBM_s115
                                 + 0.710036 w_k1_INTC_s115
                                 + 0.767244 w_k1_KO_s115
                                 + 0.699007 w_k1_MAT_s115
                                 + 0.7082 w_k1_MSFT_s115
                                 + 0.840255 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.04219 w_k1_ADM_s116
                                 + 0.998207 w_k1_AXP_s116
                                 + 1.08095 w_k1_DIS_s116
                                 + 1.01242 w_k1_FDX_s116
                                 + 1.00163 w_k1_IBM_s116
                                 + 0.914713 w_k1_INTC_s116
                                 + 1.05169 w_k1_KO_s116 + 1.52174 w_k1_MAT_s116
                                 + 1.07988 w_k1_MSFT_s116
                                 + 1.04032 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.44932 w_k1_ADM_s116
                                 + 1.42876 w_k1_AXP_s116
                                 + 1.37616 w_k1_DIS_s116
                                 + 1.14985 w_k1_FDX_s116 + 1.4211 w_k1_IBM_s116
                                 + 0.861471 w_k1_INTC_s116
                                 + 1.2541 w_k1_KO_s116 + 1.85577 w_k1_MAT_s116
                                 + 1.13545 w_k1_MSFT_s116
                                 + 1.30547 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.00029 w_k1_ADM_s116
                                 + 1.13111 w_k1_AXP_s116
                                 + 1.19975 w_k1_DIS_s116
                                 + 1.00408 w_k1_FDX_s116 + 1.4565 w_k1_IBM_s116
                                 + 1.1631 w_k1_INTC_s116 + 1.30531 w_k1_KO_s116
                                 + 1.46308 w_k1_MAT_s116
                                 + 1.11533 w_k1_MSFT_s116
                                 + 1.13287 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.34838 w_k1_ADM_s116
                                 + 1.43857 w_k1_AXP_s116
                                 + 1.73846 w_k1_DIS_s116
                                 + 1.47676 w_k1_FDX_s116
                                 + 1.32746 w_k1_IBM_s116
                                 + 1.22267 w_k1_INTC_s116
                                 + 1.35701 w_k1_KO_s116 + 1.66998 w_k1_MAT_s116
                                 + 1.22645 w_k1_MSFT_s116 + 1.1844 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.974063 w_k1_ADM_s116
                                 + 0.749035 w_k1_AXP_s116
                                 + 0.804629 w_k1_DIS_s116
                                 + 0.920608 w_k1_FDX_s116
                                 + 0.710278 w_k1_IBM_s116
                                 + 0.718636 w_k1_INTC_s116
                                 + 0.790257 w_k1_KO_s116
                                 + 0.991245 w_k1_MAT_s116
                                 + 0.888101 w_k1_MSFT_s116
                                 + 0.825841 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.867654 w_k1_ADM_s116
                                 + 0.727853 w_k1_AXP_s116
                                 + 0.77109 w_k1_DIS_s116
                                 + 0.808693 w_k1_FDX_s116
                                 + 1.03816 w_k1_IBM_s116
                                 + 0.806491 w_k1_INTC_s116
                                 + 0.89641 w_k1_KO_s116 + 0.93085 w_k1_MAT_s116
                                 + 0.892693 w_k1_MSFT_s116
                                 + 0.893189 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.11412 w_k1_ADM_s116
                                 + 1.29428 w_k1_AXP_s116
                                 + 0.975487 w_k1_DIS_s116
                                 + 1.26578 w_k1_FDX_s116
                                 + 0.878688 w_k1_IBM_s116
                                 + 0.944183 w_k1_INTC_s116
                                 + 0.941776 w_k1_KO_s116
                                 + 1.08773 w_k1_MAT_s116
                                 + 0.944966 w_k1_MSFT_s116
                                 + 1.09259 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.16214 w_k1_ADM_s116
                                 + 1.10268 w_k1_AXP_s116
                                 + 1.33288 w_k1_DIS_s116
                                 + 1.18618 w_k1_FDX_s116
                                 + 0.902646 w_k1_IBM_s116
                                 + 0.982861 w_k1_INTC_s116
                                 + 1.18837 w_k1_KO_s116 + 1.23234 w_k1_MAT_s116
                                 + 1.22124 w_k1_MSFT_s116
                                 + 1.09811 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.929858 w_k1_ADM_s117
                                 + 1.36971 w_k1_AXP_s117
                                 + 1.15758 w_k1_DIS_s117
                                 + 1.18344 w_k1_FDX_s117
                                 + 0.803183 w_k1_IBM_s117
                                 + 1.05094 w_k1_INTC_s117
                                 + 0.995028 w_k1_KO_s117
                                 + 1.25462 w_k1_MAT_s117
                                 + 1.03297 w_k1_MSFT_s117
                                 + 1.03272 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.965177 w_k1_ADM_s117
                                 + 1.17488 w_k1_AXP_s117
                                 + 0.917251 w_k1_DIS_s117
                                 + 0.959346 w_k1_FDX_s117
                                 + 1.14181 w_k1_IBM_s117
                                 + 1.00975 w_k1_INTC_s117
                                 + 1.11034 w_k1_KO_s117 + 1.46695 w_k1_MAT_s117
                                 + 0.993684 w_k1_MSFT_s117
                                 + 1.01615 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.20612 w_k1_ADM_s117
                                 + 1.69347 w_k1_AXP_s117
                                 + 1.44379 w_k1_DIS_s117
                                 + 1.35948 w_k1_FDX_s117
                                 + 1.19701 w_k1_IBM_s117
                                 + 0.998827 w_k1_INTC_s117
                                 + 1.4632 w_k1_KO_s117 + 1.98974 w_k1_MAT_s117
                                 + 1.12927 w_k1_MSFT_s117
                                 + 1.14903 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.888176 w_k1_ADM_s117
                                 + 0.762957 w_k1_AXP_s117
                                 + 0.68291 w_k1_DIS_s117
                                 + 0.867822 w_k1_FDX_s117
                                 + 0.691093 w_k1_IBM_s117
                                 + 0.872249 w_k1_INTC_s117
                                 + 0.688865 w_k1_KO_s117
                                 + 0.723636 w_k1_MAT_s117
                                 + 0.814385 w_k1_MSFT_s117
                                 + 0.813996 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.20184 w_k1_ADM_s117
                                 + 0.897891 w_k1_AXP_s117
                                 + 1.20585 w_k1_DIS_s117
                                 + 1.00055 w_k1_FDX_s117
                                 + 0.864031 w_k1_IBM_s117
                                 + 1.10099 w_k1_INTC_s117 + 1.1646 w_k1_KO_s117
                                 + 0.865358 w_k1_MAT_s117
                                 + 1.04502 w_k1_MSFT_s117
                                 + 0.995411 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.93434 w_k1_ADM_s117
                                 + 0.658487 w_k1_AXP_s117
                                 + 0.758589 w_k1_DIS_s117
                                 + 0.82573 w_k1_FDX_s117
                                 + 0.941202 w_k1_IBM_s117
                                 + 0.890774 w_k1_INTC_s117
                                 + 1.01574 w_k1_KO_s117 + 1.06213 w_k1_MAT_s117
                                 + 0.994992 w_k1_MSFT_s117
                                 + 0.921737 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.901201 w_k1_ADM_s117
                                 + 0.795664 w_k1_AXP_s117
                                 + 0.817627 w_k1_DIS_s117
                                 + 0.817537 w_k1_FDX_s117
                                 + 1.1958 w_k1_IBM_s117
                                 + 0.936226 w_k1_INTC_s117
                                 + 1.15468 w_k1_KO_s117 + 1.01115 w_k1_MAT_s117
                                 + 0.801106 w_k1_MSFT_s117
                                 + 0.866329 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.997071 w_k1_ADM_s117
                                 + 1.19863 w_k1_AXP_s117
                                 + 1.23666 w_k1_DIS_s117
                                 + 1.09812 w_k1_FDX_s117
                                 + 1.54745 w_k1_IBM_s117
                                 + 1.27088 w_k1_INTC_s117
                                 + 1.27593 w_k1_KO_s117 + 1.86261 w_k1_MAT_s117
                                 + 1.18765 w_k1_MSFT_s117
                                 + 1.14964 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0124465567412425 w_k1_ADM_s000
                                 + 0.0197126043485976 w_k1_ADM_s001
                                 + 0.0189547047146892 w_k1_ADM_s002
                                 + 0.0278638044378049 w_k1_ADM_s003
                                 + 0.0180165138650959 w_k1_ADM_s004
                                 + 0.0139793209403538 w_k1_ADM_s005
                                 + 0.0121950029046896 w_k1_ADM_s006
                                 + 0.0102153248528489 w_k1_ADM_s007
                                 + 0.0137512038516159 w_k1_ADM_s010
                                 + 0.024078221302494 w_k1_ADM_s011
                                 + 0.045762032226723 w_k1_ADM_s012
                                 + 0.0613550872493231 w_k1_ADM_s013
                                 + 0.0193207005500078 w_k1_ADM_s014
                                 + 0.0132871136219568 w_k1_ADM_s015
                                 + 0.0387031072636813 w_k1_ADM_s016
                                 + 0.0360489985513913 w_k1_ADM_s017
                                 + 0.0219894377355209 w_k1_ADM_s100
                                 + 0.0597785437467732 w_k1_ADM_s101
                                 + 0.0241187020947178 w_k1_ADM_s102
                                 + 0.029158810108753 w_k1_ADM_s103
                                 + 0.0172651813965238 w_k1_ADM_s104
                                 + 0.0534077707103395 w_k1_ADM_s105
                                 + 0.0201611666002862 w_k1_ADM_s106
                                 + 0.0738925636203175 w_k1_ADM_s107
                                 + 0.0450956223139813 w_k1_ADM_s110
                                 + 0.0510905190727726 w_k1_ADM_s111
                                 + 0.0776512629475558 w_k1_ADM_s112
                                 + 0.025613008711582 w_k1_ADM_s113
                                 + 0.0429999471905795 w_k1_ADM_s114
                                 + 0.0184316680759858 w_k1_ADM_s115
                                 + 0.0425268611642227 w_k1_ADM_s116
                                 + 0.0221045896855511 w_k1_ADM_s117
                                 + 0.0152654430984183 w_k1_AXP_s000
                                 + 0.0216510394248919 w_k1_AXP_s001
                                 + 0.0164752392934305 w_k1_AXP_s002
                                 + 0.0241900748692252 w_k1_AXP_s003
                                 + 0.0230126234580985 w_k1_AXP_s004
                                 + 0.0156220332036338 w_k1_AXP_s005
                                 + 0.0147716013522133 w_k1_AXP_s006
                                 + 0.0097097150109241 w_k1_AXP_s007
                                 + 0.0166013925525609 w_k1_AXP_s010
                                 + 0.0247536076418939 w_k1_AXP_s011
                                 + 0.0451890797655542 w_k1_AXP_s012
                                 + 0.0582333546880184 w_k1_AXP_s013
                                 + 0.0208046884854444 w_k1_AXP_s014
                                 + 0.013725840483076 w_k1_AXP_s015
                                 + 0.0409716870394706 w_k1_AXP_s016
                                 + 0.0375824363363223 w_k1_AXP_s017
                                 + 0.0217188351963414 w_k1_AXP_s100
                                 + 0.0610509350009473 w_k1_AXP_s101
                                 + 0.0302417590400723 w_k1_AXP_s102
                                 + 0.0302415910325618 w_k1_AXP_s103
                                 + 0.0181485202484509 w_k1_AXP_s104
                                 + 0.0444884327155869 w_k1_AXP_s105
                                 + 0.0234287012462845 w_k1_AXP_s106
                                 + 0.0797816381432112 w_k1_AXP_s107
                                 + 0.0412676631378667 w_k1_AXP_s110
                                 + 0.0533905779784083 w_k1_AXP_s111
                                 + 0.0717195884563291 w_k1_AXP_s112
                                 + 0.0256808292106405 w_k1_AXP_s113
                                 + 0.0457486955707308 w_k1_AXP_s114
                                 + 0.0171237979345265 w_k1_AXP_s115
                                 + 0.0399989760759532 w_k1_AXP_s116
                                 + 0.0241026778521506 w_k1_AXP_s117
                                 + 0.0138583498985838 w_k1_DIS_s000
                                 + 0.0215306958234781 w_k1_DIS_s001
                                 + 0.0191575114786321 w_k1_DIS_s002
                                 + 0.0263077305039583 w_k1_DIS_s003
                                 + 0.0213222196243222 w_k1_DIS_s004
                                 + 0.0164280809538663 w_k1_DIS_s005
                                 + 0.0149804018089588 w_k1_DIS_s006
                                 + 0.0107835804777373 w_k1_DIS_s007
                                 + 0.0169401891717688 w_k1_DIS_s010
                                 + 0.0259362735667126 w_k1_DIS_s011
                                 + 0.0459805964370741 w_k1_DIS_s012
                                 + 0.0648397654529808 w_k1_DIS_s013
                                 + 0.0207829357560034 w_k1_DIS_s014
                                 + 0.0132274653046748 w_k1_DIS_s015
                                 + 0.0442885970089315 w_k1_DIS_s016
                                 + 0.0353926968956907 w_k1_DIS_s017
                                 + 0.021241033596524 w_k1_DIS_s100
                                 + 0.0608012730588364 w_k1_DIS_s101
                                 + 0.030861305780735 w_k1_DIS_s102
                                 + 0.02778538803747 w_k1_DIS_s103
                                 + 0.0197876743399005 w_k1_DIS_s104
                                 + 0.0468787065864278 w_k1_DIS_s105
                                 + 0.0225658122996781 w_k1_DIS_s106
                                 + 0.0785434783723269 w_k1_DIS_s107
                                 + 0.0438881553952363 w_k1_DIS_s110
                                 + 0.0542451260996158 w_k1_DIS_s111
                                 + 0.0680751711099395 w_k1_DIS_s112
                                 + 0.0258358243148017 w_k1_DIS_s113
                                 + 0.0450344978775418 w_k1_DIS_s114
                                 + 0.0174559601279143 w_k1_DIS_s115
                                 + 0.0414374714298867 w_k1_DIS_s116
                                 + 0.0229713549292768 w_k1_DIS_s117
                                 + 0.0135379180629432 w_k1_FDX_s000
                                 + 0.0199198316207296 w_k1_FDX_s001
                                 + 0.0187646489845824 w_k1_FDX_s002
                                 + 0.0257426804553932 w_k1_FDX_s003
                                 + 0.021742536383383 w_k1_FDX_s004
                                 + 0.0152705518866522 w_k1_FDX_s005
                                 + 0.0151687862345863 w_k1_FDX_s006
                                 + 0.0104318446171835 w_k1_FDX_s007
                                 + 0.0160068365336535 w_k1_FDX_s010
                                 + 0.0241950352407891 w_k1_FDX_s011
                                 + 0.0457872078166889 w_k1_FDX_s012
                                 + 0.0618687084076084 w_k1_FDX_s013
                                 + 0.0215591719986722 w_k1_FDX_s014
                                 + 0.0118096996307743 w_k1_FDX_s015
                                 + 0.0427494766321358 w_k1_FDX_s016
                                 + 0.0363006601320615 w_k1_FDX_s017
                                 + 0.0200399585927434 w_k1_FDX_s100
                                 + 0.0581686615450109 w_k1_FDX_s101
                                 + 0.0303572271550155 w_k1_FDX_s102
                                 + 0.0281416656289085 w_k1_FDX_s103
                                 + 0.0183056312115758 w_k1_FDX_s104
                                 + 0.0463327912784049 w_k1_FDX_s105
                                 + 0.021953586298078 w_k1_FDX_s106
                                 + 0.0738176954366541 w_k1_FDX_s107
                                 + 0.0428721385275642 w_k1_FDX_s110
                                 + 0.051455513794872 w_k1_FDX_s111
                                 + 0.0708330752683973 w_k1_FDX_s112
                                 + 0.0269998648279947 w_k1_FDX_s113
                                 + 0.0435405430099188 w_k1_FDX_s114
                                 + 0.0176376586568311 w_k1_FDX_s115
                                 + 0.0411034912103937 w_k1_FDX_s116
                                 + 0.0227222229663511 w_k1_FDX_s117
                                 + 0.0127016990983215 w_k1_IBM_s000
                                 + 0.0206730054390734 w_k1_IBM_s001
                                 + 0.0183485524695908 w_k1_IBM_s002
                                 + 0.0265336895491611 w_k1_IBM_s003
                                 + 0.022193177848767 w_k1_IBM_s004
                                 + 0.0158645529614842 w_k1_IBM_s005
                                 + 0.0139704900001512 w_k1_IBM_s006
                                 + 0.0112489515551345 w_k1_IBM_s007
                                 + 0.0167307341144948 w_k1_IBM_s010
                                 + 0.0256301593545423 w_k1_IBM_s011
                                 + 0.0460100249106478 w_k1_IBM_s012
                                 + 0.0629695481411881 w_k1_IBM_s013
                                 + 0.022223930439448 w_k1_IBM_s014
                                 + 0.013174269452759 w_k1_IBM_s015
                                 + 0.0445899148990764 w_k1_IBM_s016
                                 + 0.0354829786696702 w_k1_IBM_s017
                                 + 0.0206645810092712 w_k1_IBM_s100
                                 + 0.0602345748369014 w_k1_IBM_s101
                                 + 0.0302074390812431 w_k1_IBM_s102
                                 + 0.0287535398471568 w_k1_IBM_s103
                                 + 0.0194349260487193 w_k1_IBM_s104
                                 + 0.0473593063626135 w_k1_IBM_s105
                                 + 0.0232364719645783 w_k1_IBM_s106
                                 + 0.0785830727998851 w_k1_IBM_s107
                                 + 0.0428255354488942 w_k1_IBM_s110
                                 + 0.0541025448408262 w_k1_IBM_s111
                                 + 0.0719575393286645 w_k1_IBM_s112
                                 + 0.0268195595564298 w_k1_IBM_s113
                                 + 0.0431100125414523 w_k1_IBM_s114
                                 + 0.0170811139875413 w_k1_IBM_s115
                                 + 0.0405120256588314 w_k1_IBM_s116
                                 + 0.0231292643274694 w_k1_IBM_s117
                                 + 0.0111699705321662 w_k1_INTC_s000
                                 + 0.0235116678898087 w_k1_INTC_s001
                                 + 0.0183859601871148 w_k1_INTC_s002
                                 + 0.0253241828570662 w_k1_INTC_s003
                                 + 0.0235845334196266 w_k1_INTC_s004
                                 + 0.017671495741486 w_k1_INTC_s005
                                 + 0.0146196807064145 w_k1_INTC_s006
                                 + 0.0090477345120261 w_k1_INTC_s007
                                 + 0.0145374909806599 w_k1_INTC_s010
                                 + 0.0241708987111371 w_k1_INTC_s011
                                 + 0.0418313295854014 w_k1_INTC_s012
                                 + 0.0662451907732153 w_k1_INTC_s013
                                 + 0.0237785289592099 w_k1_INTC_s014
                                 + 0.0121845521344906 w_k1_INTC_s015
                                 + 0.0392312912309611 w_k1_INTC_s016
                                 + 0.035721136533124 w_k1_INTC_s017
                                 + 0.0236903663629703 w_k1_INTC_s100
                                 + 0.0601647409760918 w_k1_INTC_s101
                                 + 0.0313914824495595 w_k1_INTC_s102
                                 + 0.0283132879255205 w_k1_INTC_s103
                                 + 0.0202005858072909 w_k1_INTC_s104
                                 + 0.0467661610388943 w_k1_INTC_s105
                                 + 0.0209605649973788 w_k1_INTC_s106
                                 + 0.0734424045807461 w_k1_INTC_s107
                                 + 0.0439653553074161 w_k1_INTC_s110
                                 + 0.0511466946303422 w_k1_INTC_s111
                                 + 0.0695488935081179 w_k1_INTC_s112
                                 + 0.0258215480682211 w_k1_INTC_s113
                                 + 0.0435094399361793 w_k1_INTC_s114
                                 + 0.0145917949273679 w_k1_INTC_s115
                                 + 0.0353488512960678 w_k1_INTC_s116
                                 + 0.0226586346050227 w_k1_INTC_s117
                                 + 0.0142645964850235 w_k1_KO_s000
                                 + 0.0193540383613204 w_k1_KO_s001
                                 + 0.0179920603535006 w_k1_KO_s002
                                 + 0.0247518969932803 w_k1_KO_s003
                                 + 0.0205282195049356 w_k1_KO_s004
                                 + 0.0145380574004382 w_k1_KO_s005
                                 + 0.0145095260396319 w_k1_KO_s006
                                 + 0.0103310703238334 w_k1_KO_s007
                                 + 0.0157424720761235 w_k1_KO_s010
                                 + 0.025101108398108 w_k1_KO_s011
                                 + 0.0421628068062057 w_k1_KO_s012
                                 + 0.0636783751512019 w_k1_KO_s013
                                 + 0.0189485493341091 w_k1_KO_s014
                                 + 0.0133671609847159 w_k1_KO_s015
                                 + 0.0460033568769007 w_k1_KO_s016
                                 + 0.0377851622187666 w_k1_KO_s017
                                 + 0.0222179032498112 w_k1_KO_s100
                                 + 0.0677010447795241 w_k1_KO_s101
                                 + 0.0295311299033609 w_k1_KO_s102
                                 + 0.0291278942544629 w_k1_KO_s103
                                 + 0.0172146396850147 w_k1_KO_s104
                                 + 0.0468158991201168 w_k1_KO_s105
                                 + 0.0210046689834513 w_k1_KO_s106
                                 + 0.0703357494483847 w_k1_KO_s107
                                 + 0.044397088358392 w_k1_KO_s110
                                 + 0.0556088913799752 w_k1_KO_s111
                                 + 0.0673080318400857 w_k1_KO_s112
                                 + 0.0242068620819959 w_k1_KO_s113
                                 + 0.0384638277543988 w_k1_KO_s114
                                 + 0.0170929681146483 w_k1_KO_s115
                                 + 0.0404793217230612 w_k1_KO_s116
                                 + 0.0242169384791235 w_k1_KO_s117
                                 + 0.0194958590747653 w_k1_MAT_s000
                                 + 0.0187829495981085 w_k1_MAT_s001
                                 + 0.0218610292214178 w_k1_MAT_s002
                                 + 0.0260214488343547 w_k1_MAT_s003
                                 + 0.0217858369843339 w_k1_MAT_s004
                                 + 0.0132679498263287 w_k1_MAT_s005
                                 + 0.0159108484890233 w_k1_MAT_s006
                                 + 0.01161289134412 w_k1_MAT_s007
                                 + 0.0184396902742702 w_k1_MAT_s010
                                 + 0.0283435151914215 w_k1_MAT_s011
                                 + 0.0475259316924196 w_k1_MAT_s012
                                 + 0.0683179304866126 w_k1_MAT_s013
                                 + 0.016596192738606 w_k1_MAT_s014
                                 + 0.0124852914439562 w_k1_MAT_s015
                                 + 0.0421921265309852 w_k1_MAT_s016
                                 + 0.0348866301822032 w_k1_MAT_s017
                                 + 0.0201413631455515 w_k1_MAT_s100
                                 + 0.0525549524159403 w_k1_MAT_s101
                                 + 0.0273268910172613 w_k1_MAT_s102
                                 + 0.0298101977481257 w_k1_MAT_s103
                                 + 0.0193494104948243 w_k1_MAT_s104
                                 + 0.0435826557666654 w_k1_MAT_s105
                                 + 0.0238644181217752 w_k1_MAT_s106
                                 + 0.0784492308571069 w_k1_MAT_s107
                                 + 0.0445164480348871 w_k1_MAT_s110
                                 + 0.0589921600513165 w_k1_MAT_s111
                                 + 0.0620493473416433 w_k1_MAT_s112
                                 + 0.0319104625982998 w_k1_MAT_s113
                                 + 0.0516962861755416 w_k1_MAT_s114
                                 + 0.0169743135489802 w_k1_MAT_s115
                                 + 0.0506231852162211 w_k1_MAT_s116
                                 + 0.0285891411482066 w_k1_MAT_s117
                                 + 0.0131736353475552 w_k1_MSFT_s000
                                 + 0.0201822632479578 w_k1_MSFT_s001
                                 + 0.0188203961443565 w_k1_MSFT_s002
                                 + 0.0254129268255143 w_k1_MSFT_s003
                                 + 0.0210117902507105 w_k1_MSFT_s004
                                 + 0.0148507936401096 w_k1_MSFT_s005
                                 + 0.0144040797516596 w_k1_MSFT_s006
                                 + 0.0104070686187096 w_k1_MSFT_s007
                                 + 0.0144128608392962 w_k1_MSFT_s010
                                 + 0.0230932058396366 w_k1_MSFT_s011
                                 + 0.0456510029628135 w_k1_MSFT_s012
                                 + 0.0665748304265761 w_k1_MSFT_s013
                                 + 0.0211873081903818 w_k1_MSFT_s014
                                 + 0.0133477581471993 w_k1_MSFT_s015
                                 + 0.0422392307981961 w_k1_MSFT_s016
                                 + 0.0339486529468132 w_k1_MSFT_s017
                                 + 0.0209523948352488 w_k1_MSFT_s100
                                 + 0.0570137122406268 w_k1_MSFT_s101
                                 + 0.0292645299961887 w_k1_MSFT_s102
                                 + 0.0285946996878095 w_k1_MSFT_s103
                                 + 0.0193780460807091 w_k1_MSFT_s104
                                 + 0.0431401924859214 w_k1_MSFT_s105
                                 + 0.0232738213213524 w_k1_MSFT_s106
                                 + 0.0671539050639223 w_k1_MSFT_s107
                                 + 0.0447625644109821 w_k1_MSFT_s110
                                 + 0.0541196653464803 w_k1_MSFT_s111
                                 + 0.063825113025196 w_k1_MSFT_s112
                                 + 0.0263556019379849 w_k1_MSFT_s113
                                 + 0.0438366168280917 w_k1_MSFT_s114
                                 + 0.0161215856215224 w_k1_MSFT_s115
                                 + 0.0404605674890679 w_k1_MSFT_s116
                                 + 0.0221638496638832 w_k1_MSFT_s117
                                 + 0.0136199209269547 w_k1_PG_s000
                                 + 0.0188612857260075 w_k1_PG_s001
                                 + 0.0177281708857708 w_k1_PG_s002
                                 + 0.0241918055162277 w_k1_PG_s003
                                 + 0.0216104153300775 w_k1_PG_s004
                                 + 0.0155801675464454 w_k1_PG_s005
                                 + 0.0144188215517872 w_k1_PG_s006
                                 + 0.0109971559267174 w_k1_PG_s007
                                 + 0.016492274876665 w_k1_PG_s010
                                 + 0.0233799487894422 w_k1_PG_s011
                                 + 0.0455967603143561 w_k1_PG_s012
                                 + 0.0611254172910935 w_k1_PG_s013
                                 + 0.0213679942308704 w_k1_PG_s014
                                 + 0.0129320544033224 w_k1_PG_s015
                                 + 0.0418150116069144 w_k1_PG_s016
                                 + 0.0349316551553628 w_k1_PG_s017
                                 + 0.0190611507414197 w_k1_PG_s100
                                 + 0.0586706668861807 w_k1_PG_s101
                                 + 0.0307882793979451 w_k1_PG_s102
                                 + 0.0296654465868419 w_k1_PG_s103
                                 + 0.0185435231067298 w_k1_PG_s104
                                 + 0.0442838588813505 w_k1_PG_s105
                                 + 0.0225478727136487 w_k1_PG_s106
                                 + 0.0711954056533515 w_k1_PG_s107
                                 + 0.0445641108788514 w_k1_PG_s110
                                 + 0.0521212005379927 w_k1_PG_s111
                                 + 0.0685953660485067 w_k1_PG_s112
                                 + 0.0269605198229816 w_k1_PG_s113
                                 + 0.0406711428419497 w_k1_PG_s114
                                 + 0.0172647368459185 w_k1_PG_s115
                                 + 0.0404870306992707 w_k1_PG_s116
                                 + 0.0220248385319566 w_k1_PG_s117 >= 1.19
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
