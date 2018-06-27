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
 w_k1_ADM_s0_return_constraint:  0.969782 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.10408 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.09911 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.811025 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.23233 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.04005 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.02751 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.44237 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.909001 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.09879 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.993195 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.764957 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.21925 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.29597 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.985392 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.04433 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.22145 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.54614 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.750692 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.33015 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.868137 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.721083 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.29165 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.15336 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.845027 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.2758 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.805127 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.19702 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.978409 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.822079 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.16372 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.927377 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.15913 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.912543 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.67295 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.0163 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.32541 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.997374 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.938049 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.33924 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.13347 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.863802 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.16266 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.85525 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.915883 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.5593 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.26216 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.05789 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.90728 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.683488 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.40958 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.54463 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.845023 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.829627 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.35497 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.43459 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.576616 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.24894 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.813325 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.871387 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.58175 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.1916 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.764547 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.82718 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.749932 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.34739 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.903322 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.816165 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.10364 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.11781 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.18172 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.804665 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.835701 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.47435 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.87814 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.977078 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.944725 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.37648 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.17377 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.795168 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.41025 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.917738 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.929282 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.73141 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.32103 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.27654 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.933024 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.597716 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.09558 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.85533 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.788484 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.963498 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.28513 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.45771 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.43131 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.18095 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.675162 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.631279 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.75623 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.97933 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.950938 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.50998 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.603067 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.31462 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.868793 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.02935 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.940388 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.06115 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.43223 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.856744 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.603202 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.19768 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.70005 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.873063 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.985043 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.17405 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.11178 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.76966 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.17377 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.940055 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.01172 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.3657 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.993099 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.10116 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.913047 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.734033 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.21243 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.52489 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.896922 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.05163 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.19532 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.21983 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.671708 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.27944 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.859102 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.754397 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.42148 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.949863 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.920133 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.40552 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.71591 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.30767 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.02335 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.930409 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.11105 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.0434 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.31045 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.852275 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.708538 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.04681 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.53811 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.05817 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.932513 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.41208 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.00634 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.14057 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.36655 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.897452 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.737263 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.58653 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.88057 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.972688 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.22649 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.750026 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.995364 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.0909 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.94311 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.720345 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.26102 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.84415 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.51441 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.875931 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.592018 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.01642 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.25882 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.3273 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.607902 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.0737 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.800001 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.738474 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.724324 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.995425 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.804349 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.08638 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.817901 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.828994 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.02397 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.546 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.1258 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.64825 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.946138 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.16918 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.03586 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.874056 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13308 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.897084 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.978652 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.44143 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.21488 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.18947 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.04652 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.804425 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.22331 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.41262 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.921793 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.0057 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.15799 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.19065 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.746348 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.05346 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.853499 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.819154 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.33153 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.01408 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.813985 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.1792 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.828901 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.21705 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.850475 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.944257 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.16984 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.10072 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.2102 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.940133 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.917661 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.0139 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.37218 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.06734 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.95985 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.28248 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.07717 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.906748 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.31524 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.952422 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.914398 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.58815 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.46586 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.19732 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.971208 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.711681 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.989696 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.1245 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.0391 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.886705 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.31939 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.48722 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.543004 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.969444 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.725846 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.821056 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.38751 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.15071 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.846611 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.28179 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.744675 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.00064 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.830045 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.976781 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.95549 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.07779 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.17647 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.779969 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.80258 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.14818 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.3505 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.761826 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.931885 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.61809 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.08642 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.988314 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.21857 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.801153 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.846119 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.90669 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.78657 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.02992 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.22258 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.559925 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.20666 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.995965 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.907752 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.716788 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.53193 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.89343 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.421494 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.981677 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.716154 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.12469 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.55534 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.17467 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.764747 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.17623 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.730008 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.900554 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.668489 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.995153 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.787176 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.10364 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.718681 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.850212 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04922 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.86161 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.20661 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.666407 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.979063 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.14924 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.990599 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.817831 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.1209 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.950499 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.05816 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.52748 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.2308 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.22322 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.03001 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.808476 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.22538 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.54315 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.940208 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.06653 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.12615 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.1644 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.673338 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.10718 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.817004 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.78042 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.42499 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.972563 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.01547 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.29739 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.877449 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.37267 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.01522 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.00262 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.07883 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.01103 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.37656 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.861722 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.784011 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.07647 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.22954 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.08958 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.987659 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.12965 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.00067 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.868177 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.21208 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.01187 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.0053 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.43065 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.19393 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.06455 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.10746 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.854325 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.13932 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.31978 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.945643 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.93296 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.14792 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.28907 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.729631 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.09562 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.88328 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.965936 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.18754 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.992259 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.884905 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.27397 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.805312 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.979101 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.91522 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.874667 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.06471 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.947793 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.983587 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.868453 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.757962 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.12352 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.32192 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.854442 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 3.0300079473317 z_k1_s0000
                                 - 0.936002302440726 z_k1_s0001
                                 - 2.51400213399361 z_k1_s0002
                                 - 1.3070013546297 z_k1_s0003
                                 - 0.681002020915687 z_k1_s0004
                                 - 1.48200236134485 z_k1_s0005
                                 - 1.00000219644974 z_k1_s0006
                                 - 0.904002355436219 z_k1_s0007
                                 - 0.593001483478314 z_k1_s0010
                                 - 0.392000293917805 z_k1_s0011
                                 - 1.42199927262716 z_k1_s0012
                                 - 0.538999969061938 z_k1_s0013
                                 - 0.491000169847537 z_k1_s0014
                                 - 0.921999430476955 z_k1_s0015
                                 - 0.854997873756669 z_k1_s0016
                                 - 0.586997447876107 z_k1_s0017
                                 - 0.220000073039503 z_k1_s0020
                                 - 0.42700133482219 z_k1_s0021
                                 - 0.644000996255975 z_k1_s0022
                                 - 0.592999039060973 z_k1_s0023
                                 - 0.80700177819056 z_k1_s0024
                                 - 0.287999968731657 z_k1_s0025
                                 - 0.21000012258773 z_k1_s0026
                                 - 0.688001088403909 z_k1_s0027
                                 - 0.200000125922227 z_k1_s0030
                                 - 0.886001649726279 z_k1_s0031
                                 - 0.370000352506209 z_k1_s0032
                                 - 0.318000048786228 z_k1_s0033
                                 - 0.974002470252629 z_k1_s0034
                                 - 0.326000340743331 z_k1_s0035
                                 - 0.358000313070852 z_k1_s0036
                                 - 0.552998861768851 z_k1_s0037
                                 - 1.14699745550935 z_k1_s0040
                                 - 0.972999301770013 z_k1_s0041
                                 - 1.38000085764577 z_k1_s0042
                                 - 0.75099916380091 z_k1_s0043
                                 - 2.1579980838281 z_k1_s0044
                                 - 1.66800103663272 z_k1_s0045
                                 - 0.601999722666229 z_k1_s0046
                                 - 1.0929974219493 z_k1_s0047
                                 - 1.64799793576582 z_k1_s0050
                                 - 1.69400425649967 z_k1_s0051
                                 - 1.21100119349155 z_k1_s0052
                                 - 1.04300230439642 z_k1_s0053
                                 - 0.835000735397919 z_k1_s0054
                                 - 2.3049976398905 z_k1_s0055
                                 - 3.82500231086908 z_k1_s0056
                                 - 2.1610015364992 z_k1_s0057
                                 - 0.451998863501343 z_k1_s0060
                                 - 0.617998066358212 z_k1_s0061
                                 - 0.454999304070828 z_k1_s0062
                                 - 0.219999923811046 z_k1_s0063
                                 - 0.519000692557005 z_k1_s0064
                                 - 0.527998581267099 z_k1_s0065
                                 - 0.292999887606989 z_k1_s0066
                                 - 0.347999611084873 z_k1_s0067
                                 - 0.607999665083896 z_k1_s0070
                                 - 0.629000289421109 z_k1_s0071
                                 - 0.752000578393917 z_k1_s0072
                                 - 0.516000522275699 z_k1_s0073
                                 - 0.99500241429228 z_k1_s0074
                                 - 0.221000216377765 z_k1_s0075
                                 - 0.475998434966108 z_k1_s0076
                                 - 0.517999683640535 z_k1_s0077
                                 - 0.231000484867387 z_k1_s0100
                                 - 0.497002188952733 z_k1_s0101
                                 - 0.880002643781695 z_k1_s0102
                                 - 0.275000449756225 z_k1_s0103
                                 - 0.441000104363799 z_k1_s0104
                                 - 0.219000372770251 z_k1_s0105
                                 - 0.453002224063895 z_k1_s0106
                                 - 0.3500021541648 z_k1_s0107
                                 - 0.686997824970246 z_k1_s0110
                                 - 0.202999731711247 z_k1_s0111
                                 - 0.53299941770474 z_k1_s0112
                                 - 0.206999605823447 z_k1_s0113
                                 - 0.145999898639497 z_k1_s0114
                                 - 0.659999371986987 z_k1_s0115
                                 - 0.509000792830732 z_k1_s0116
                                 - 0.153999646863898 z_k1_s0117
                                 - 1.98099672191533 z_k1_s0120
                                 - 0.50099958436986 z_k1_s0121
                                 - 0.470999711850671 z_k1_s0122
                                 - 0.499999588619221 z_k1_s0123
                                 - 0.508999550374978 z_k1_s0124
                                 - 0.562999892009319 z_k1_s0125
                                 - 0.698997600796908 z_k1_s0126
                                 - 0.486999643860905 z_k1_s0127
                                 - 0.505000020842537 z_k1_s0130
                                 - 0.221000037368714 z_k1_s0131
                                 - 0.210000137350362 z_k1_s0132
                                 - 0.351999970587273 z_k1_s0133
                                 - 0.631001181052756 z_k1_s0134
                                 - 0.395999570660682 z_k1_s0135
                                 - 0.224999943375388 z_k1_s0136
                                 - 0.629999461051085 z_k1_s0137
                                 - 0.146000005827861 z_k1_s0140
                                 - 0.122000043708203 z_k1_s0141
                                 - 0.137999959387989 z_k1_s0142
                                 - 0.260000180296149 z_k1_s0143
                                 - 0.314000095065379 z_k1_s0144
                                 - 0.119999987798246 z_k1_s0145
                                 - 0.296000123475635 z_k1_s0146
                                 - 0.37600058787435 z_k1_s0147
                                 - 0.156000015679896 z_k1_s0150
                                 - 0.153999945212202 z_k1_s0151
                                 - 0.237000340021391 z_k1_s0152
                                 - 0.542999738378088 z_k1_s0153
                                 - 0.261999112867486 z_k1_s0154
                                 - 0.454999167399666 z_k1_s0155
                                 - 0.120000012061458 z_k1_s0156
                                 - 0.181000053326034 z_k1_s0157
                                 - 0.153999996766571 z_k1_s0160
                                 - 0.262000644834267 z_k1_s0161
                                 - 0.388000822479708 z_k1_s0162
                                 - 0.404999331502762 z_k1_s0163
                                 - 0.304998915069776 z_k1_s0164
                                 - 0.142000145114683 z_k1_s0165
                                 - 0.229999966195518 z_k1_s0166
                                 - 0.592999321116497 z_k1_s0167
                                 - 0.348999996886324 z_k1_s0170
                                 - 0.323000098252892 z_k1_s0171
                                 - 0.268000142989974 z_k1_s0172
                                 - 0.130999913498935 z_k1_s0173
                                 - 0.252000035677206 z_k1_s0174
                                 - 0.149999958207903 z_k1_s0175
                                 - 0.22799877170879 z_k1_s0176
                                 - 0.504999388686921 z_k1_s0177
                                 - 0.0660000464519813 z_k1_s1000
                                 - 0.094999963317107 z_k1_s1001
                                 - 0.0250000517045841 z_k1_s1002
                                 - 0.0220000422840297 z_k1_s1003
                                 - 0.026000028044733 z_k1_s1004
                                 - 0.0420002122878693 z_k1_s1005
                                 - 0.0390000420670994 z_k1_s1006
                                 - 0.0870001525959159 z_k1_s1007
                                 - 0.237000826976278 z_k1_s1010
                                 - 0.207000151345786 z_k1_s1011
                                 - 0.0750000885745947 z_k1_s1012
                                 - 0.0650000088645794 z_k1_s1013
                                 - 0.125000196124374 z_k1_s1014
                                 - 0.174000717653583 z_k1_s1015
                                 - 0.312000333550278 z_k1_s1016
                                 - 0.260000035458318 z_k1_s1017
                                 - 0.027000039647754 z_k1_s1020
                                 - 0.0790001568286853 z_k1_s1021
                                 - 0.0440002353225658 z_k1_s1022
                                 - 0.0650001882262375 z_k1_s1023
                                 - 0.0160000309172006 z_k1_s1024
                                 - 0.0210000531038478 z_k1_s1025
                                 - 0.0300000663197658 z_k1_s1026
                                 - 0.0520002173811074 z_k1_s1027
                                 - 0.0580000629973668 z_k1_s1030
                                 - 0.0280000490608283 z_k1_s1031
                                 - 0.042999910428859 z_k1_s1032
                                 - 0.0380000814397199 z_k1_s1033
                                 - 0.0370000574017966 z_k1_s1034
                                 - 0.055000282084074 z_k1_s1035
                                 - 0.0290000314986834 z_k1_s1036
                                 - 0.128000081649267 z_k1_s1037
                                 - 0.131999810706393 z_k1_s1040
                                 - 0.0600000045028582 z_k1_s1041
                                 - 0.0439999369021308 z_k1_s1042
                                 - 0.187999881309025 z_k1_s1043
                                 - 0.123000125430798 z_k1_s1044
                                 - 0.107999543305389 z_k1_s1045
                                 - 0.0890000232792309 z_k1_s1046
                                 - 0.0859995748543233 z_k1_s1047
                                 - 0.0400002236820315 z_k1_s1050
                                 - 0.0560002779946549 z_k1_s1051
                                 - 0.18800094582498 z_k1_s1052
                                 - 0.0660003632153205 z_k1_s1053
                                 - 0.0660003632153205 z_k1_s1054
                                 - 0.0690001133600378 z_k1_s1055
                                 - 0.0460002513743047 z_k1_s1056
                                 - 0.0550003222128722 z_k1_s1057
                                 - 0.0420000188018119 z_k1_s1060
                                 - 0.0280000125345413 z_k1_s1061
                                 - 0.0880000768514477 z_k1_s1062
                                 - 0.064000016164655 z_k1_s1063
                                 - 0.0320000080823275 z_k1_s1064
                                 - 0.0190000334770317 z_k1_s1065
                                 - 0.0730000389222023 z_k1_s1066
                                 - 0.0910000844372968 z_k1_s1067
                                 - 0.0479999581305535 z_k1_s1070
                                 - 0.0569999548175285 z_k1_s1071
                                 - 0.0999998825220118 z_k1_s1072
                                 - 0.0709999093306344 z_k1_s1073
                                 - 0.102000010852343 z_k1_s1074
                                 - 0.05399998012185 z_k1_s1075
                                 - 0.126000098817529 z_k1_s1076
                                 - 0.167999889756907 z_k1_s1077
                                 - 0.0569999344849601 z_k1_s1100
                                 - 0.0559999546097678 z_k1_s1101
                                 - 0.0859996004653056 z_k1_s1102
                                 - 0.140999741599727 z_k1_s1103
                                 - 0.0809999506891133 z_k1_s1104
                                 - 0.179999705531534 z_k1_s1105
                                 - 0.161000171102803 z_k1_s1106
                                 - 0.069999922462229 z_k1_s1107
                                 - 0.254000097583603 z_k1_s1110
                                 - 0.255001151481598 z_k1_s1111
                                 - 0.0660000799360002 z_k1_s1112
                                 - 0.242999788727042 z_k1_s1113
                                 - 0.147000331912292 z_k1_s1114
                                 - 0.251000159895728 z_k1_s1115
                                 - 0.306000092175473 z_k1_s1116
                                 - 0.0900002262402226 z_k1_s1117
                                 - 0.0810001420234211 z_k1_s1120
                                 - 0.0459999820133995 z_k1_s1121
                                 - 0.10699957603155 z_k1_s1122
                                 - 0.0909996170268549 z_k1_s1123
                                 - 0.0569999638166176 z_k1_s1124
                                 - 0.0529999341154837 z_k1_s1125
                                 - 0.215999846062989 z_k1_s1126
                                 - 0.148000220042834 z_k1_s1127
                                 - 0.254999995255229 z_k1_s1130
                                 - 0.130000030502691 z_k1_s1131
                                 - 0.293000830756636 z_k1_s1132
                                 - 0.227999976004661 z_k1_s1133
                                 - 0.413000729759022 z_k1_s1134
                                 - 0.134000083102812 z_k1_s1135
                                 - 0.0990001265521264 z_k1_s1136
                                 - 0.127000158952725 z_k1_s1137
                                 - 0.726997976385308 z_k1_s1140
                                 - 0.493998299808356 z_k1_s1141
                                 - 0.378000350814599 z_k1_s1142
                                 - 0.177999733465682 z_k1_s1143
                                 - 0.607998405609584 z_k1_s1144
                                 - 0.172999620892096 z_k1_s1145
                                 - 0.230999566787609 z_k1_s1146
                                 - 0.448999877042438 z_k1_s1147
                                 - 0.232000143261185 z_k1_s1150
                                 - 0.378999845118496 z_k1_s1151
                                 - 0.143000122918785 z_k1_s1152
                                 - 0.238000102169615 z_k1_s1153
                                 - 0.190000194701916 z_k1_s1154
                                 - 0.367999999186459 z_k1_s1155
                                 - 0.346000307322386 z_k1_s1156
                                 - 0.465999485490992 z_k1_s1157
                                 - 0.129999750833377 z_k1_s1160
                                 - 0.0999997881026336 z_k1_s1161
                                 - 0.155999490600423 z_k1_s1162
                                 - 0.174999892179146 z_k1_s1163
                                 - 0.117999739441126 z_k1_s1164
                                 - 0.249999470256584 z_k1_s1165
                                 - 0.186999246072554 z_k1_s1166
                                 - 0.198999914963649 z_k1_s1167
                                 - 0.0719999260654838 z_k1_s1170
                                 - 0.153000271626229 z_k1_s1171
                                 - 0.165000274753794 z_k1_s1172
                                 - 0.0449999653784159 z_k1_s1173
                                 - 0.116000030233132 z_k1_s1174
                                 - 0.204000053168611 z_k1_s1175
                                 - 0.0739999265867447 z_k1_s1176
                                 - 0.0979995620422422 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.716456 w_k1_ADM_s000
                                 + 0.727589 w_k1_AXP_s000
                                 + 0.70132 w_k1_DIS_s000
                                 + 0.839161 w_k1_FDX_s000
                                 + 0.733525 w_k1_IBM_s000
                                 + 0.908618 w_k1_INTC_s000
                                 + 0.730749 w_k1_KO_s000
                                 + 0.679898 w_k1_MAT_s000
                                 + 0.832684 w_k1_MSFT_s000
                                 + 0.809302 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.22594 w_k1_ADM_s000
                                 + 1.3278 w_k1_AXP_s000
                                 + 0.918089 w_k1_DIS_s000
                                 + 0.966195 w_k1_FDX_s000
                                 + 1.43556 w_k1_IBM_s000
                                 + 1.04625 w_k1_INTC_s000
                                 + 0.901434 w_k1_KO_s000 + 1.1009 w_k1_MAT_s000
                                 + 0.960778 w_k1_MSFT_s000
                                 + 0.944841 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.947893 w_k1_ADM_s000
                                 + 1.01751 w_k1_AXP_s000
                                 + 0.937155 w_k1_DIS_s000
                                 + 0.970832 w_k1_FDX_s000
                                 + 0.938917 w_k1_IBM_s000
                                 + 0.978779 w_k1_INTC_s000
                                 + 0.883034 w_k1_KO_s000
                                 + 0.840802 w_k1_MAT_s000
                                 + 0.984091 w_k1_MSFT_s000
                                 + 0.975947 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.920714 w_k1_ADM_s000
                                 + 0.764656 w_k1_AXP_s000
                                 + 1.02171 w_k1_DIS_s000
                                 + 0.866084 w_k1_FDX_s000
                                 + 1.29851 w_k1_IBM_s000
                                 + 1.01504 w_k1_INTC_s000
                                 + 1.00877 w_k1_KO_s000
                                 + 0.934025 w_k1_MAT_s000
                                 + 1.03702 w_k1_MSFT_s000
                                 + 0.99318 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.31333 w_k1_ADM_s000
                                 + 1.18064 w_k1_AXP_s000 + 0.8602 w_k1_DIS_s000
                                 + 1.11935 w_k1_FDX_s000
                                 + 0.799749 w_k1_IBM_s000
                                 + 0.97616 w_k1_INTC_s000
                                 + 0.792445 w_k1_KO_s000
                                 + 0.736458 w_k1_MAT_s000
                                 + 0.923404 w_k1_MSFT_s000
                                 + 0.971542 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.08329 w_k1_ADM_s000
                                 + 1.0515 w_k1_AXP_s000 + 1.25946 w_k1_DIS_s000
                                 + 1.25412 w_k1_FDX_s000
                                 + 0.806075 w_k1_IBM_s000
                                 + 1.19815 w_k1_INTC_s000
                                 + 0.865116 w_k1_KO_s000
                                 + 0.788919 w_k1_MAT_s000
                                 + 1.20203 w_k1_MSFT_s000 + 1.1203 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.21743 w_k1_ADM_s000
                                 + 1.25359 w_k1_AXP_s000 + 1.391 w_k1_DIS_s000
                                 + 1.2001 w_k1_FDX_s000 + 1.43091 w_k1_IBM_s000
                                 + 1.17551 w_k1_INTC_s000
                                 + 1.24721 w_k1_KO_s000 + 1.33853 w_k1_MAT_s000
                                 + 1.09758 w_k1_MSFT_s000
                                 + 1.19748 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.2623 w_k1_ADM_s000
                                 + 1.48339 w_k1_AXP_s000
                                 + 1.74944 w_k1_DIS_s000
                                 + 1.51575 w_k1_FDX_s000 + 1.1308 w_k1_IBM_s000
                                 + 1.65 w_k1_INTC_s000 + 1.21573 w_k1_KO_s000
                                 + 1.19335 w_k1_MAT_s000
                                 + 1.36642 w_k1_MSFT_s000
                                 + 1.20701 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.656347 w_k1_ADM_s001
                                 + 0.728345 w_k1_AXP_s001
                                 + 0.819584 w_k1_DIS_s001
                                 + 0.815422 w_k1_FDX_s001
                                 + 0.963922 w_k1_IBM_s001
                                 + 0.779766 w_k1_INTC_s001
                                 + 0.88795 w_k1_KO_s001
                                 + 0.746175 w_k1_MAT_s001
                                 + 0.842235 w_k1_MSFT_s001
                                 + 0.752509 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.824405 w_k1_ADM_s001
                                 + 0.853172 w_k1_AXP_s001
                                 + 1.21789 w_k1_DIS_s001
                                 + 0.988168 w_k1_FDX_s001
                                 + 1.01266 w_k1_IBM_s001
                                 + 0.809271 w_k1_INTC_s001
                                 + 0.981405 w_k1_KO_s001
                                 + 0.865214 w_k1_MAT_s001
                                 + 1.01272 w_k1_MSFT_s001
                                 + 0.964364 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.22152 w_k1_ADM_s001
                                 + 1.45749 w_k1_AXP_s001
                                 + 1.65118 w_k1_DIS_s001
                                 + 1.32059 w_k1_FDX_s001
                                 + 1.33129 w_k1_IBM_s001
                                 + 1.3619 w_k1_INTC_s001 + 1.32143 w_k1_KO_s001
                                 + 1.30451 w_k1_MAT_s001
                                 + 1.24871 w_k1_MSFT_s001
                                 + 1.16761 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.07165 w_k1_ADM_s001
                                 + 1.44935 w_k1_AXP_s001
                                 + 1.09627 w_k1_DIS_s001
                                 + 1.25871 w_k1_FDX_s001
                                 + 0.918251 w_k1_IBM_s001
                                 + 1.03139 w_k1_INTC_s001
                                 + 1.01462 w_k1_KO_s001 + 1.17439 w_k1_MAT_s001
                                 + 1.22769 w_k1_MSFT_s001 + 1.0019 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.662359 w_k1_ADM_s001
                                 + 0.725353 w_k1_AXP_s001
                                 + 0.835659 w_k1_DIS_s001
                                 + 0.873539 w_k1_FDX_s001
                                 + 0.714823 w_k1_IBM_s001
                                 + 0.858003 w_k1_INTC_s001
                                 + 0.677576 w_k1_KO_s001
                                 + 0.577711 w_k1_MAT_s001
                                 + 0.824568 w_k1_MSFT_s001
                                 + 0.786233 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.929445 w_k1_ADM_s001
                                 + 1.17769 w_k1_AXP_s001
                                 + 0.892841 w_k1_DIS_s001
                                 + 0.862383 w_k1_FDX_s001
                                 + 1.25526 w_k1_IBM_s001
                                 + 0.799042 w_k1_INTC_s001
                                 + 0.932301 w_k1_KO_s001
                                 + 1.24631 w_k1_MAT_s001
                                 + 0.766167 w_k1_MSFT_s001
                                 + 0.987949 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.02306 w_k1_ADM_s001
                                 + 1.02309 w_k1_AXP_s001
                                 + 0.917182 w_k1_DIS_s001
                                 + 1.03943 w_k1_FDX_s001
                                 + 0.878292 w_k1_IBM_s001
                                 + 1.10175 w_k1_INTC_s001
                                 + 0.850533 w_k1_KO_s001
                                 + 0.762313 w_k1_MAT_s001
                                 + 1.01791 w_k1_MSFT_s001
                                 + 1.03956 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.778932 w_k1_ADM_s001
                                 + 0.811397 w_k1_AXP_s001
                                 + 0.523612 w_k1_DIS_s001
                                 + 0.736477 w_k1_FDX_s001
                                 + 0.712638 w_k1_IBM_s001
                                 + 0.637213 w_k1_INTC_s001
                                 + 0.614237 w_k1_KO_s001
                                 + 0.608838 w_k1_MAT_s001
                                 + 0.647619 w_k1_MSFT_s001
                                 + 0.746409 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.39869 w_k1_ADM_s002
                                 + 0.917647 w_k1_AXP_s002
                                 + 1.24642 w_k1_DIS_s002
                                 + 1.21066 w_k1_FDX_s002
                                 + 0.793473 w_k1_IBM_s002
                                 + 1.11931 w_k1_INTC_s002
                                 + 1.02721 w_k1_KO_s002
                                 + 0.559878 w_k1_MAT_s002
                                 + 1.23671 w_k1_MSFT_s002
                                 + 1.08277 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.883416 w_k1_ADM_s002
                                 + 0.699606 w_k1_AXP_s002
                                 + 0.614254 w_k1_DIS_s002
                                 + 0.836034 w_k1_FDX_s002
                                 + 0.670487 w_k1_IBM_s002
                                 + 0.80639 w_k1_INTC_s002
                                 + 0.764457 w_k1_KO_s002
                                 + 0.688755 w_k1_MAT_s002
                                 + 0.809879 w_k1_MSFT_s002
                                 + 0.880041 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.58788 w_k1_ADM_s002
                                 + 1.31661 w_k1_AXP_s002 + 1.3595 w_k1_DIS_s002
                                 + 1.26368 w_k1_FDX_s002
                                 + 1.07742 w_k1_IBM_s002
                                 + 1.06801 w_k1_INTC_s002
                                 + 1.08594 w_k1_KO_s002
                                 + 0.995697 w_k1_MAT_s002
                                 + 1.14089 w_k1_MSFT_s002
                                 + 0.998744 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.36309 w_k1_ADM_s002
                                 + 1.39682 w_k1_AXP_s002
                                 + 1.30025 w_k1_DIS_s002
                                 + 1.21901 w_k1_FDX_s002
                                 + 1.58109 w_k1_IBM_s002
                                 + 1.0413 w_k1_INTC_s002 + 1.16729 w_k1_KO_s002
                                 + 1.30549 w_k1_MAT_s002
                                 + 0.933516 w_k1_MSFT_s002
                                 + 1.1755 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.971753 w_k1_ADM_s002
                                 + 0.957813 w_k1_AXP_s002
                                 + 0.908809 w_k1_DIS_s002
                                 + 1.05502 w_k1_FDX_s002
                                 + 0.830407 w_k1_IBM_s002
                                 + 0.985849 w_k1_INTC_s002
                                 + 0.871278 w_k1_KO_s002
                                 + 0.87704 w_k1_MAT_s002
                                 + 0.920448 w_k1_MSFT_s002 + 1.011 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.988265 w_k1_ADM_s002
                                 + 0.942133 w_k1_AXP_s002
                                 + 0.506748 w_k1_DIS_s002
                                 + 0.774816 w_k1_FDX_s002
                                 + 0.874078 w_k1_IBM_s002
                                 + 0.75461 w_k1_INTC_s002
                                 + 0.571013 w_k1_KO_s002
                                 + 0.807043 w_k1_MAT_s002
                                 + 0.607642 w_k1_MSFT_s002
                                 + 0.842838 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.13035 w_k1_ADM_s002
                                 + 1.1069 w_k1_AXP_s002
                                 + 0.905583 w_k1_DIS_s002
                                 + 1.11379 w_k1_FDX_s002
                                 + 0.863901 w_k1_IBM_s002
                                 + 0.910184 w_k1_INTC_s002
                                 + 0.957247 w_k1_KO_s002
                                 + 1.30486 w_k1_MAT_s002
                                 + 0.961201 w_k1_MSFT_s002
                                 + 1.07641 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.07473 w_k1_ADM_s002
                                 + 0.910567 w_k1_AXP_s002
                                 + 0.865077 w_k1_DIS_s002
                                 + 0.939122 w_k1_FDX_s002
                                 + 1.18946 w_k1_IBM_s002
                                 + 0.90678 w_k1_INTC_s002
                                 + 1.09729 w_k1_KO_s002 + 1.22892 w_k1_MAT_s002
                                 + 0.965051 w_k1_MSFT_s002
                                 + 1.01855 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.19729 w_k1_ADM_s003
                                 + 1.45866 w_k1_AXP_s003
                                 + 1.80436 w_k1_DIS_s003
                                 + 1.32014 w_k1_FDX_s003 + 1.3834 w_k1_IBM_s003
                                 + 1.28907 w_k1_INTC_s003
                                 + 1.27106 w_k1_KO_s003 + 1.61355 w_k1_MAT_s003
                                 + 1.38671 w_k1_MSFT_s003
                                 + 1.45015 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.22081 w_k1_ADM_s003
                                 + 1.13634 w_k1_AXP_s003
                                 + 1.23684 w_k1_DIS_s003
                                 + 1.10935 w_k1_FDX_s003
                                 + 1.26881 w_k1_IBM_s003
                                 + 1.08675 w_k1_INTC_s003
                                 + 1.13954 w_k1_KO_s003 + 1.04213 w_k1_MAT_s003
                                 + 1.163 w_k1_MSFT_s003 + 1.18613 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.797315 w_k1_ADM_s003
                                 + 1.003 w_k1_AXP_s003 + 0.961742 w_k1_DIS_s003
                                 + 0.907577 w_k1_FDX_s003
                                 + 1.40273 w_k1_IBM_s003
                                 + 1.00062 w_k1_INTC_s003
                                 + 0.97411 w_k1_KO_s003 + 1.31349 w_k1_MAT_s003
                                 + 0.948501 w_k1_MSFT_s003
                                 + 1.11372 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.895469 w_k1_ADM_s003
                                 + 0.727545 w_k1_AXP_s003
                                 + 0.671674 w_k1_DIS_s003
                                 + 0.892287 w_k1_FDX_s003
                                 + 0.664564 w_k1_IBM_s003
                                 + 1.03633 w_k1_INTC_s003
                                 + 0.675632 w_k1_KO_s003
                                 + 0.557528 w_k1_MAT_s003
                                 + 0.948105 w_k1_MSFT_s003
                                 + 0.903243 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.11886 w_k1_ADM_s003
                                 + 1.11592 w_k1_AXP_s003
                                 + 1.12221 w_k1_DIS_s003
                                 + 1.19984 w_k1_FDX_s003
                                 + 0.891214 w_k1_IBM_s003
                                 + 1.00561 w_k1_INTC_s003
                                 + 0.960194 w_k1_KO_s003
                                 + 1.06106 w_k1_MAT_s003
                                 + 1.23425 w_k1_MSFT_s003 + 1.061 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.994915 w_k1_ADM_s003
                                 + 1.13406 w_k1_AXP_s003
                                 + 0.897435 w_k1_DIS_s003
                                 + 0.95348 w_k1_FDX_s003
                                 + 0.817332 w_k1_IBM_s003
                                 + 0.984286 w_k1_INTC_s003
                                 + 0.721541 w_k1_KO_s003
                                 + 0.868878 w_k1_MAT_s003
                                 + 0.832496 w_k1_MSFT_s003
                                 + 1.04483 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.923979 w_k1_ADM_s003
                                 + 0.85814 w_k1_AXP_s003
                                 + 0.94458 w_k1_DIS_s003
                                 + 0.849913 w_k1_FDX_s003
                                 + 0.945752 w_k1_IBM_s003
                                 + 0.826784 w_k1_INTC_s003
                                 + 0.994311 w_k1_KO_s003
                                 + 0.954914 w_k1_MAT_s003
                                 + 1.02003 w_k1_MSFT_s003
                                 + 1.02577 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.841037 w_k1_ADM_s003
                                 + 0.817831 w_k1_AXP_s003
                                 + 0.704868 w_k1_DIS_s003
                                 + 0.784907 w_k1_FDX_s003
                                 + 0.90322 w_k1_IBM_s003
                                 + 0.763947 w_k1_INTC_s003
                                 + 0.756031 w_k1_KO_s003
                                 + 0.848373 w_k1_MAT_s003
                                 + 0.792583 w_k1_MSFT_s003
                                 + 0.884651 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.13269 w_k1_ADM_s004
                                 + 1.08379 w_k1_AXP_s004
                                 + 1.19548 w_k1_DIS_s004
                                 + 1.14346 w_k1_FDX_s004
                                 + 1.06277 w_k1_IBM_s004
                                 + 1.02941 w_k1_INTC_s004
                                 + 1.23005 w_k1_KO_s004
                                 + 0.920279 w_k1_MAT_s004
                                 + 1.13764 w_k1_MSFT_s004
                                 + 0.994467 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.993503 w_k1_ADM_s004
                                 + 1.32571 w_k1_AXP_s004
                                 + 1.46211 w_k1_DIS_s004
                                 + 1.18543 w_k1_FDX_s004
                                 + 1.02087 w_k1_IBM_s004
                                 + 1.21878 w_k1_INTC_s004
                                 + 1.28773 w_k1_KO_s004
                                 + 0.775497 w_k1_MAT_s004
                                 + 1.23933 w_k1_MSFT_s004
                                 + 1.01832 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.788004 w_k1_ADM_s004
                                 + 0.909843 w_k1_AXP_s004
                                 + 1.05408 w_k1_DIS_s004
                                 + 0.90267 w_k1_FDX_s004
                                 + 1.18757 w_k1_IBM_s004
                                 + 0.885329 w_k1_INTC_s004
                                 + 1.1179 w_k1_KO_s004 + 1.08005 w_k1_MAT_s004
                                 + 0.878984 w_k1_MSFT_s004
                                 + 0.928475 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.886073 w_k1_ADM_s004
                                 + 1.39821 w_k1_AXP_s004
                                 + 0.951899 w_k1_DIS_s004
                                 + 0.98418 w_k1_FDX_s004
                                 + 1.49758 w_k1_IBM_s004
                                 + 0.879561 w_k1_INTC_s004
                                 + 1.21261 w_k1_KO_s004 + 1.40458 w_k1_MAT_s004
                                 + 0.916746 w_k1_MSFT_s004
                                 + 0.987381 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.06789 w_k1_ADM_s004
                                 + 0.99828 w_k1_AXP_s004
                                 + 0.951481 w_k1_DIS_s004
                                 + 0.967529 w_k1_FDX_s004
                                 + 0.824539 w_k1_IBM_s004
                                 + 0.91054 w_k1_INTC_s004
                                 + 0.968751 w_k1_KO_s004
                                 + 0.642963 w_k1_MAT_s004
                                 + 0.958222 w_k1_MSFT_s004
                                 + 0.863555 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.733089 w_k1_ADM_s004
                                 + 0.726447 w_k1_AXP_s004
                                 + 0.755919 w_k1_DIS_s004
                                 + 0.825559 w_k1_FDX_s004
                                 + 0.784924 w_k1_IBM_s004
                                 + 0.821683 w_k1_INTC_s004
                                 + 0.83706 w_k1_KO_s004
                                 + 0.521599 w_k1_MAT_s004
                                 + 0.813289 w_k1_MSFT_s004
                                 + 0.750385 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.694809 w_k1_ADM_s004
                                 + 0.627879 w_k1_AXP_s004
                                 + 0.584946 w_k1_DIS_s004
                                 + 0.689397 w_k1_FDX_s004
                                 + 1.05511 w_k1_IBM_s004
                                 + 0.7626 w_k1_INTC_s004
                                 + 0.888636 w_k1_KO_s004
                                 + 0.753064 w_k1_MAT_s004
                                 + 0.741409 w_k1_MSFT_s004
                                 + 0.74807 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.24155 w_k1_ADM_s004
                                 + 1.60789 w_k1_AXP_s004
                                 + 1.66977 w_k1_DIS_s004
                                 + 1.24295 w_k1_FDX_s004
                                 + 1.47127 w_k1_IBM_s004
                                 + 1.29534 w_k1_INTC_s004
                                 + 1.45024 w_k1_KO_s004 + 1.26365 w_k1_MAT_s004
                                 + 1.34297 w_k1_MSFT_s004
                                 + 1.11815 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.853092 w_k1_ADM_s005
                                 + 0.75079 w_k1_AXP_s005
                                 + 0.949903 w_k1_DIS_s005
                                 + 0.851131 w_k1_FDX_s005
                                 + 0.992857 w_k1_IBM_s005
                                 + 0.990805 w_k1_INTC_s005
                                 + 1.00509 w_k1_KO_s005
                                 + 0.768565 w_k1_MAT_s005
                                 + 0.945921 w_k1_MSFT_s005
                                 + 0.947133 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.22723 w_k1_ADM_s005
                                 + 1.27753 w_k1_AXP_s005
                                 + 1.12774 w_k1_DIS_s005 + 1.1745 w_k1_FDX_s005
                                 + 0.811902 w_k1_IBM_s005
                                 + 1.19154 w_k1_INTC_s005
                                 + 0.943953 w_k1_KO_s005
                                 + 0.868412 w_k1_MAT_s005
                                 + 1.11708 w_k1_MSFT_s005
                                 + 1.01204 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.35707 w_k1_ADM_s005
                                 + 1.07905 w_k1_AXP_s005
                                 + 1.31264 w_k1_DIS_s005
                                 + 1.32329 w_k1_FDX_s005 + 0.8416 w_k1_IBM_s005
                                 + 1.26506 w_k1_INTC_s005
                                 + 0.994928 w_k1_KO_s005
                                 + 1.14477 w_k1_MAT_s005
                                 + 1.36606 w_k1_MSFT_s005 + 1.0929 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.30851 w_k1_ADM_s005
                                 + 1.51307 w_k1_AXP_s005
                                 + 1.28124 w_k1_DIS_s005
                                 + 1.17265 w_k1_FDX_s005
                                 + 1.23769 w_k1_IBM_s005
                                 + 1.14426 w_k1_INTC_s005
                                 + 1.20296 w_k1_KO_s005 + 2.24071 w_k1_MAT_s005
                                 + 1.34737 w_k1_MSFT_s005
                                 + 1.24263 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.954329 w_k1_ADM_s005
                                 + 0.910254 w_k1_AXP_s005
                                 + 0.817347 w_k1_DIS_s005
                                 + 0.843468 w_k1_FDX_s005
                                 + 1.50134 w_k1_IBM_s005
                                 + 0.918551 w_k1_INTC_s005
                                 + 1.16833 w_k1_KO_s005 + 1.8851 w_k1_MAT_s005
                                 + 0.905246 w_k1_MSFT_s005
                                 + 1.0372 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.161 w_k1_ADM_s005
                                 + 1.42568 w_k1_AXP_s005
                                 + 1.60982 w_k1_DIS_s005
                                 + 1.23378 w_k1_FDX_s005
                                 + 1.52362 w_k1_IBM_s005
                                 + 1.35975 w_k1_INTC_s005
                                 + 1.38835 w_k1_KO_s005 + 1.97579 w_k1_MAT_s005
                                 + 1.28789 w_k1_MSFT_s005
                                 + 1.19792 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.853101 w_k1_ADM_s005
                                 + 0.693428 w_k1_AXP_s005
                                 + 0.727895 w_k1_DIS_s005
                                 + 0.807938 w_k1_FDX_s005
                                 + 0.657126 w_k1_IBM_s005
                                 + 0.947625 w_k1_INTC_s005
                                 + 0.748423 w_k1_KO_s005
                                 + 0.522026 w_k1_MAT_s005
                                 + 0.860187 w_k1_MSFT_s005
                                 + 0.786868 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.03692 w_k1_ADM_s005
                                 + 0.945677 w_k1_AXP_s005
                                 + 1.09175 w_k1_DIS_s005
                                 + 0.986834 w_k1_FDX_s005
                                 + 1.17131 w_k1_IBM_s005
                                 + 0.96744 w_k1_INTC_s005
                                 + 1.02323 w_k1_KO_s005 + 1.34565 w_k1_MAT_s005
                                 + 1.03685 w_k1_MSFT_s005
                                 + 0.99685 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.01043 w_k1_ADM_s006
                                 + 1.04721 w_k1_AXP_s006
                                 + 1.04477 w_k1_DIS_s006
                                 + 1.11028 w_k1_FDX_s006
                                 + 0.827455 w_k1_IBM_s006
                                 + 1.18561 w_k1_INTC_s006
                                 + 0.984815 w_k1_KO_s006
                                 + 0.970116 w_k1_MAT_s006
                                 + 1.05293 w_k1_MSFT_s006 + 1.0534 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.3643 w_k1_ADM_s006
                                 + 1.30217 w_k1_AXP_s006
                                 + 1.28375 w_k1_DIS_s006
                                 + 1.27022 w_k1_FDX_s006
                                 + 1.04689 w_k1_IBM_s006
                                 + 1.07297 w_k1_INTC_s006
                                 + 1.05269 w_k1_KO_s006
                                 + 0.954681 w_k1_MAT_s006
                                 + 1.13772 w_k1_MSFT_s006
                                 + 1.13451 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.966258 w_k1_ADM_s006
                                 + 0.782827 w_k1_AXP_s006
                                 + 0.881216 w_k1_DIS_s006
                                 + 0.918159 w_k1_FDX_s006
                                 + 0.849334 w_k1_IBM_s006
                                 + 0.880764 w_k1_INTC_s006
                                 + 0.958453 w_k1_KO_s006
                                 + 0.563914 w_k1_MAT_s006
                                 + 0.954011 w_k1_MSFT_s006
                                 + 0.957763 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.36966 w_k1_ADM_s006
                                 + 1.60601 w_k1_AXP_s006
                                 + 1.67925 w_k1_DIS_s006
                                 + 1.38905 w_k1_FDX_s006
                                 + 1.44807 w_k1_IBM_s006
                                 + 1.10534 w_k1_INTC_s006
                                 + 1.32875 w_k1_KO_s006 + 1.39893 w_k1_MAT_s006
                                 + 1.07742 w_k1_MSFT_s006
                                 + 1.27017 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.860387 w_k1_ADM_s006
                                 + 0.726434 w_k1_AXP_s006
                                 + 0.670958 w_k1_DIS_s006
                                 + 0.769482 w_k1_FDX_s006
                                 + 0.87436 w_k1_IBM_s006
                                 + 0.912776 w_k1_INTC_s006
                                 + 0.816734 w_k1_KO_s006
                                 + 0.814352 w_k1_MAT_s006
                                 + 0.834293 w_k1_MSFT_s006
                                 + 0.821436 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.778478 w_k1_ADM_s006
                                 + 1.07852 w_k1_AXP_s006
                                 + 0.896063 w_k1_DIS_s006
                                 + 0.90408 w_k1_FDX_s006
                                 + 1.18179 w_k1_IBM_s006
                                 + 0.939583 w_k1_INTC_s006
                                 + 1.05109 w_k1_KO_s006 + 1.35222 w_k1_MAT_s006
                                 + 0.91708 w_k1_MSFT_s006
                                 + 0.941407 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.983424 w_k1_ADM_s006
                                 + 1.124 w_k1_AXP_s006 + 1.4055 w_k1_DIS_s006
                                 + 1.08844 w_k1_FDX_s006
                                 + 1.41988 w_k1_IBM_s006
                                 + 1.0953 w_k1_INTC_s006 + 1.38806 w_k1_KO_s006
                                 + 1.62092 w_k1_MAT_s006
                                 + 1.26362 w_k1_MSFT_s006
                                 + 1.11367 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.0787 w_k1_ADM_s006
                                 + 1.09305 w_k1_AXP_s006
                                 + 0.93545 w_k1_DIS_s006
                                 + 0.998314 w_k1_FDX_s006
                                 + 0.947312 w_k1_IBM_s006
                                 + 0.855834 w_k1_INTC_s006
                                 + 1.03317 w_k1_KO_s006 + 1.0415 w_k1_MAT_s006
                                 + 0.919052 w_k1_MSFT_s006
                                 + 1.07159 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.17147 w_k1_ADM_s007
                                 + 0.777523 w_k1_AXP_s007
                                 + 1.03262 w_k1_DIS_s007
                                 + 1.07931 w_k1_FDX_s007
                                 + 0.922864 w_k1_IBM_s007
                                 + 1.02939 w_k1_INTC_s007
                                 + 1.31084 w_k1_KO_s007
                                 + 0.701633 w_k1_MAT_s007
                                 + 1.11981 w_k1_MSFT_s007
                                 + 1.07234 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.4765 w_k1_ADM_s007
                                 + 1.32626 w_k1_AXP_s007
                                 + 1.08445 w_k1_DIS_s007
                                 + 1.17084 w_k1_FDX_s007
                                 + 0.992224 w_k1_IBM_s007
                                 + 1.02942 w_k1_INTC_s007
                                 + 1.22498 w_k1_KO_s007 + 1.16675 w_k1_MAT_s007
                                 + 1.087 w_k1_MSFT_s007 + 1.03751 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.754498 w_k1_ADM_s007
                                 + 0.748716 w_k1_AXP_s007
                                 + 0.931915 w_k1_DIS_s007
                                 + 0.884596 w_k1_FDX_s007
                                 + 0.895934 w_k1_IBM_s007
                                 + 1.0229 w_k1_INTC_s007 + 1.02999 w_k1_KO_s007
                                 + 0.886851 w_k1_MAT_s007
                                 + 0.98119 w_k1_MSFT_s007
                                 + 0.878049 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.03622 w_k1_ADM_s007
                                 + 1.0653 w_k1_AXP_s007
                                 + 0.900846 w_k1_DIS_s007
                                 + 1.04145 w_k1_FDX_s007
                                 + 1.19461 w_k1_IBM_s007
                                 + 1.04919 w_k1_INTC_s007
                                 + 1.21443 w_k1_KO_s007 + 1.6261 w_k1_MAT_s007
                                 + 0.920396 w_k1_MSFT_s007
                                 + 1.10655 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.00605 w_k1_ADM_s007
                                 + 0.801651 w_k1_AXP_s007
                                 + 0.686429 w_k1_DIS_s007
                                 + 0.7844 w_k1_FDX_s007
                                 + 0.915506 w_k1_IBM_s007
                                 + 0.842969 w_k1_INTC_s007
                                 + 0.928639 w_k1_KO_s007
                                 + 1.06607 w_k1_MAT_s007
                                 + 0.758863 w_k1_MSFT_s007
                                 + 0.888203 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.2104 w_k1_ADM_s007
                                 + 1.48431 w_k1_AXP_s007
                                 + 2.01849 w_k1_DIS_s007
                                 + 1.41904 w_k1_FDX_s007
                                 + 1.48405 w_k1_IBM_s007
                                 + 1.50935 w_k1_INTC_s007
                                 + 1.95337 w_k1_KO_s007 + 1.93555 w_k1_MAT_s007
                                 + 1.4423 w_k1_MSFT_s007 + 1.32687 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.737264 w_k1_ADM_s007
                                 + 0.551574 w_k1_AXP_s007
                                 + 0.520331 w_k1_DIS_s007
                                 + 0.758727 w_k1_FDX_s007
                                 + 0.620302 w_k1_IBM_s007
                                 + 0.769087 w_k1_INTC_s007
                                 + 0.775327 w_k1_KO_s007
                                 + 0.558638 w_k1_MAT_s007
                                 + 0.770183 w_k1_MSFT_s007
                                 + 0.748495 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.0839 w_k1_ADM_s007
                                 + 1.19952 w_k1_AXP_s007
                                 + 1.43692 w_k1_DIS_s007
                                 + 1.32105 w_k1_FDX_s007
                                 + 1.13774 w_k1_IBM_s007
                                 + 1.30383 w_k1_INTC_s007
                                 + 1.42495 w_k1_KO_s007
                                 + 0.982511 w_k1_MAT_s007
                                 + 1.35102 w_k1_MSFT_s007
                                 + 1.07727 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.968394 w_k1_ADM_s010
                                 + 0.877033 w_k1_AXP_s010
                                 + 1.19116 w_k1_DIS_s010
                                 + 1.03959 w_k1_FDX_s010
                                 + 1.29534 w_k1_IBM_s010
                                 + 1.16647 w_k1_INTC_s010
                                 + 1.07365 w_k1_KO_s010 + 1.44801 w_k1_MAT_s010
                                 + 1.13664 w_k1_MSFT_s010 + 1.1015 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.21899 w_k1_ADM_s010
                                 + 1.31094 w_k1_AXP_s010
                                 + 1.72459 w_k1_DIS_s010
                                 + 1.24085 w_k1_FDX_s010
                                 + 1.28978 w_k1_IBM_s010
                                 + 1.16399 w_k1_INTC_s010 + 1.2744 w_k1_KO_s010
                                 + 1.62106 w_k1_MAT_s010
                                 + 1.20426 w_k1_MSFT_s010
                                 + 1.24581 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.880992 w_k1_ADM_s010
                                 + 0.845516 w_k1_AXP_s010
                                 + 0.83754 w_k1_DIS_s010
                                 + 0.860817 w_k1_FDX_s010
                                 + 0.952635 w_k1_IBM_s010
                                 + 0.927947 w_k1_INTC_s010
                                 + 0.868153 w_k1_KO_s010
                                 + 0.900197 w_k1_MAT_s010
                                 + 0.839162 w_k1_MSFT_s010
                                 + 0.960382 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.0884 w_k1_ADM_s010
                                 + 0.886213 w_k1_AXP_s010
                                 + 1.02464 w_k1_DIS_s010
                                 + 0.880695 w_k1_FDX_s010
                                 + 1.58081 w_k1_IBM_s010
                                 + 1.08727 w_k1_INTC_s010
                                 + 1.07546 w_k1_KO_s010 + 1.16455 w_k1_MAT_s010
                                 + 0.968984 w_k1_MSFT_s010
                                 + 1.04773 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.10513 w_k1_ADM_s010
                                 + 1.14725 w_k1_AXP_s010 + 1.1914 w_k1_DIS_s010
                                 + 1.13925 w_k1_FDX_s010
                                 + 0.83206 w_k1_IBM_s010
                                 + 1.17237 w_k1_INTC_s010
                                 + 0.950018 w_k1_KO_s010
                                 + 0.861368 w_k1_MAT_s010
                                 + 1.09127 w_k1_MSFT_s010
                                 + 1.04846 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.92026 w_k1_ADM_s010
                                 + 0.620016 w_k1_AXP_s010
                                 + 0.529166 w_k1_DIS_s010
                                 + 0.712944 w_k1_FDX_s010
                                 + 0.80827 w_k1_IBM_s010
                                 + 0.827289 w_k1_INTC_s010
                                 + 0.635305 w_k1_KO_s010
                                 + 0.635748 w_k1_MAT_s010
                                 + 0.705629 w_k1_MSFT_s010
                                 + 0.832709 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.925189 w_k1_ADM_s010
                                 + 0.826775 w_k1_AXP_s010
                                 + 0.882098 w_k1_DIS_s010
                                 + 0.911606 w_k1_FDX_s010
                                 + 0.692103 w_k1_IBM_s010
                                 + 0.854908 w_k1_INTC_s010
                                 + 0.668718 w_k1_KO_s010
                                 + 0.717541 w_k1_MAT_s010
                                 + 0.832664 w_k1_MSFT_s010
                                 + 0.853846 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.941007 w_k1_ADM_s010
                                 + 1.246 w_k1_AXP_s010 + 1.20233 w_k1_DIS_s010
                                 + 1.08412 w_k1_FDX_s010
                                 + 1.10783 w_k1_IBM_s010
                                 + 1.11827 w_k1_INTC_s010
                                 + 0.988365 w_k1_KO_s010
                                 + 1.15383 w_k1_MAT_s010
                                 + 1.11211 w_k1_MSFT_s010
                                 + 0.999444 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.00323 w_k1_ADM_s011
                                 + 0.936007 w_k1_AXP_s011
                                 + 0.82463 w_k1_DIS_s011
                                 + 0.88288 w_k1_FDX_s011
                                 + 0.97745 w_k1_IBM_s011
                                 + 0.818565 w_k1_INTC_s011
                                 + 0.932152 w_k1_KO_s011
                                 + 0.939312 w_k1_MAT_s011
                                 + 0.872949 w_k1_MSFT_s011
                                 + 0.996795 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.10294 w_k1_ADM_s011
                                 + 1.29426 w_k1_AXP_s011
                                 + 1.15344 w_k1_DIS_s011
                                 + 1.21421 w_k1_FDX_s011
                                 + 0.813135 w_k1_IBM_s011
                                 + 1.07371 w_k1_INTC_s011 + 1.0545 w_k1_KO_s011
                                 + 0.798252 w_k1_MAT_s011
                                 + 1.11721 w_k1_MSFT_s011
                                 + 1.10312 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.05724 w_k1_ADM_s011
                                 + 1.02048 w_k1_AXP_s011
                                 + 0.928136 w_k1_DIS_s011
                                 + 1.00614 w_k1_FDX_s011
                                 + 0.701568 w_k1_IBM_s011
                                 + 1.15506 w_k1_INTC_s011
                                 + 0.820083 w_k1_KO_s011
                                 + 0.527396 w_k1_MAT_s011
                                 + 0.950272 w_k1_MSFT_s011
                                 + 0.997496 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.88718 w_k1_ADM_s011
                                 + 0.694158 w_k1_AXP_s011
                                 + 0.568998 w_k1_DIS_s011
                                 + 0.684989 w_k1_FDX_s011
                                 + 0.730703 w_k1_IBM_s011
                                 + 0.683765 w_k1_INTC_s011
                                 + 0.713489 w_k1_KO_s011
                                 + 0.980443 w_k1_MAT_s011
                                 + 0.675599 w_k1_MSFT_s011
                                 + 0.858622 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.29223 w_k1_ADM_s011
                                 + 1.5205 w_k1_AXP_s011 + 2.02223 w_k1_DIS_s011
                                 + 1.35316 w_k1_FDX_s011
                                 + 1.52403 w_k1_IBM_s011
                                 + 1.27061 w_k1_INTC_s011
                                 + 1.49908 w_k1_KO_s011 + 1.43091 w_k1_MAT_s011
                                 + 1.42822 w_k1_MSFT_s011
                                 + 1.30243 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.07259 w_k1_ADM_s011
                                 + 1.2855 w_k1_AXP_s011 + 1.15516 w_k1_DIS_s011
                                 + 1.13846 w_k1_FDX_s011
                                 + 1.11282 w_k1_IBM_s011
                                 + 1.23474 w_k1_INTC_s011
                                 + 1.15088 w_k1_KO_s011 + 1.53729 w_k1_MAT_s011
                                 + 1.12023 w_k1_MSFT_s011
                                 + 1.20668 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.01306 w_k1_ADM_s011
                                 + 0.947848 w_k1_AXP_s011
                                 + 1.01304 w_k1_DIS_s011
                                 + 0.932862 w_k1_FDX_s011
                                 + 1.20166 w_k1_IBM_s011
                                 + 0.912689 w_k1_INTC_s011
                                 + 1.21781 w_k1_KO_s011 + 1.37917 w_k1_MAT_s011
                                 + 0.926773 w_k1_MSFT_s011
                                 + 0.975022 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.05629 w_k1_ADM_s011
                                 + 1.12808 w_k1_AXP_s011
                                 + 1.33802 w_k1_DIS_s011
                                 + 1.00708 w_k1_FDX_s011
                                 + 1.78477 w_k1_IBM_s011
                                 + 1.01559 w_k1_INTC_s011
                                 + 1.40151 w_k1_KO_s011 + 1.6195 w_k1_MAT_s011
                                 + 1.14848 w_k1_MSFT_s011
                                 + 1.26017 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.08622 w_k1_ADM_s012
                                 + 1.15019 w_k1_AXP_s012
                                 + 1.37727 w_k1_DIS_s012 + 1.1377 w_k1_FDX_s012
                                 + 1.29954 w_k1_IBM_s012
                                 + 1.15502 w_k1_INTC_s012
                                 + 1.25072 w_k1_KO_s012 + 1.2416 w_k1_MAT_s012
                                 + 1.16993 w_k1_MSFT_s012
                                 + 1.26831 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.10052 w_k1_ADM_s012
                                 + 0.952825 w_k1_AXP_s012
                                 + 0.792069 w_k1_DIS_s012
                                 + 0.929286 w_k1_FDX_s012
                                 + 0.931128 w_k1_IBM_s012
                                 + 0.870497 w_k1_INTC_s012
                                 + 0.824664 w_k1_KO_s012
                                 + 0.928731 w_k1_MAT_s012
                                 + 0.807859 w_k1_MSFT_s012
                                 + 0.967861 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.25081 w_k1_ADM_s012
                                 + 1.20609 w_k1_AXP_s012
                                 + 1.12126 w_k1_DIS_s012
                                 + 1.20744 w_k1_FDX_s012
                                 + 0.876629 w_k1_IBM_s012
                                 + 1.10182 w_k1_INTC_s012
                                 + 0.97819 w_k1_KO_s012
                                 + 0.862141 w_k1_MAT_s012
                                 + 1.23222 w_k1_MSFT_s012
                                 + 1.06327 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.998498 w_k1_ADM_s012
                                 + 0.879463 w_k1_AXP_s012
                                 + 1.08519 w_k1_DIS_s012
                                 + 0.973394 w_k1_FDX_s012
                                 + 0.821134 w_k1_IBM_s012
                                 + 0.846155 w_k1_INTC_s012
                                 + 0.909751 w_k1_KO_s012
                                 + 0.779465 w_k1_MAT_s012
                                 + 0.952696 w_k1_MSFT_s012
                                 + 1.01709 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.733162 w_k1_ADM_s012
                                 + 0.759464 w_k1_AXP_s012
                                 + 0.788591 w_k1_DIS_s012
                                 + 0.830242 w_k1_FDX_s012
                                 + 1.15208 w_k1_IBM_s012
                                 + 0.789476 w_k1_INTC_s012
                                 + 0.827572 w_k1_KO_s012
                                 + 1.06104 w_k1_MAT_s012
                                 + 0.904088 w_k1_MSFT_s012
                                 + 0.951713 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.841639 w_k1_ADM_s012
                                 + 0.647914 w_k1_AXP_s012
                                 + 0.712112 w_k1_DIS_s012
                                 + 0.797855 w_k1_FDX_s012
                                 + 0.94021 w_k1_IBM_s012
                                 + 0.739137 w_k1_INTC_s012
                                 + 0.814381 w_k1_KO_s012
                                 + 0.767005 w_k1_MAT_s012
                                 + 0.817137 w_k1_MSFT_s012
                                 + 0.939041 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.707607 w_k1_ADM_s012
                                 + 0.634663 w_k1_AXP_s012
                                 + 0.63291 w_k1_DIS_s012
                                 + 0.750816 w_k1_FDX_s012
                                 + 0.733394 w_k1_IBM_s012
                                 + 0.718703 w_k1_INTC_s012
                                 + 0.643265 w_k1_KO_s012
                                 + 0.710562 w_k1_MAT_s012
                                 + 0.680421 w_k1_MSFT_s012
                                 + 0.77965 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.20626 w_k1_ADM_s012
                                 + 0.946835 w_k1_AXP_s012
                                 + 0.87829 w_k1_DIS_s012
                                 + 1.14502 w_k1_FDX_s012
                                 + 0.664592 w_k1_IBM_s012
                                 + 0.888381 w_k1_INTC_s012
                                 + 0.803351 w_k1_KO_s012
                                 + 0.654723 w_k1_MAT_s012
                                 + 1.05911 w_k1_MSFT_s012
                                 + 0.989129 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.77447 w_k1_ADM_s013
                                 + 1.01833 w_k1_AXP_s013
                                 + 1.05683 w_k1_DIS_s013
                                 + 0.970594 w_k1_FDX_s013
                                 + 1.30892 w_k1_IBM_s013
                                 + 0.874144 w_k1_INTC_s013
                                 + 1.10651 w_k1_KO_s013 + 1.33299 w_k1_MAT_s013
                                 + 0.990025 w_k1_MSFT_s013
                                 + 0.99704 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.487305 w_k1_ADM_s013
                                 + 0.528925 w_k1_AXP_s013
                                 + 0.484376 w_k1_DIS_s013
                                 + 0.613568 w_k1_FDX_s013
                                 + 0.768054 w_k1_IBM_s013
                                 + 0.827971 w_k1_INTC_s013
                                 + 0.731372 w_k1_KO_s013
                                 + 0.835926 w_k1_MAT_s013
                                 + 0.749281 w_k1_MSFT_s013
                                 + 0.76076 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.787503 w_k1_ADM_s013
                                 + 0.948192 w_k1_AXP_s013
                                 + 1.04799 w_k1_DIS_s013
                                 + 1.04869 w_k1_FDX_s013
                                 + 0.692706 w_k1_IBM_s013
                                 + 1.20866 w_k1_INTC_s013
                                 + 0.993149 w_k1_KO_s013
                                 + 0.914556 w_k1_MAT_s013
                                 + 1.19014 w_k1_MSFT_s013
                                 + 0.959991 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.546908 w_k1_ADM_s013
                                 + 0.736536 w_k1_AXP_s013
                                 + 0.679096 w_k1_DIS_s013
                                 + 0.787141 w_k1_FDX_s013
                                 + 0.655986 w_k1_IBM_s013
                                 + 0.869723 w_k1_INTC_s013
                                 + 0.679785 w_k1_KO_s013
                                 + 0.703961 w_k1_MAT_s013
                                 + 0.82934 w_k1_MSFT_s013
                                 + 0.763647 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.752695 w_k1_ADM_s013
                                 + 0.690887 w_k1_AXP_s013
                                 + 0.690008 w_k1_DIS_s013
                                 + 0.753127 w_k1_FDX_s013
                                 + 0.962725 w_k1_IBM_s013
                                 + 1.06698 w_k1_INTC_s013
                                 + 0.907832 w_k1_KO_s013
                                 + 0.87216 w_k1_MAT_s013
                                 + 0.870383 w_k1_MSFT_s013
                                 + 0.912594 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.14956 w_k1_ADM_s013
                                 + 1.5697 w_k1_AXP_s013 + 2.00125 w_k1_DIS_s013
                                 + 1.51612 w_k1_FDX_s013 + 1.1704 w_k1_IBM_s013
                                 + 1.53185 w_k1_INTC_s013
                                 + 1.60226 w_k1_KO_s013 + 1.4017 w_k1_MAT_s013
                                 + 1.58086 w_k1_MSFT_s013 + 1.3069 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.91679 w_k1_ADM_s013
                                 + 1.42895 w_k1_AXP_s013
                                 + 1.28676 w_k1_DIS_s013
                                 + 1.16152 w_k1_FDX_s013
                                 + 1.66205 w_k1_IBM_s013
                                 + 1.33834 w_k1_INTC_s013
                                 + 1.62745 w_k1_KO_s013 + 2.46595 w_k1_MAT_s013
                                 + 1.41097 w_k1_MSFT_s013
                                 + 1.29971 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.764385 w_k1_ADM_s013
                                 + 0.960428 w_k1_AXP_s013
                                 + 1.05906 w_k1_DIS_s013
                                 + 1.06924 w_k1_FDX_s013
                                 + 0.890799 w_k1_IBM_s013
                                 + 1.14724 w_k1_INTC_s013 + 1.0882 w_k1_KO_s013
                                 + 0.960783 w_k1_MAT_s013
                                 + 1.17614 w_k1_MSFT_s013
                                 + 1.05021 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.64281 w_k1_ADM_s014
                                 + 0.5233 w_k1_AXP_s014
                                 + 0.666049 w_k1_DIS_s014
                                 + 0.710394 w_k1_FDX_s014
                                 + 1.05287 w_k1_IBM_s014
                                 + 0.859491 w_k1_INTC_s014
                                 + 1.01739 w_k1_KO_s014 + 1.16019 w_k1_MAT_s014
                                 + 0.778181 w_k1_MSFT_s014
                                 + 0.859568 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.42971 w_k1_ADM_s014
                                 + 1.07029 w_k1_AXP_s014
                                 + 1.91281 w_k1_DIS_s014
                                 + 1.35373 w_k1_FDX_s014
                                 + 1.60996 w_k1_IBM_s014
                                 + 1.12932 w_k1_INTC_s014
                                 + 1.59899 w_k1_KO_s014 + 1.72778 w_k1_MAT_s014
                                 + 1.30909 w_k1_MSFT_s014
                                 + 1.16841 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.736284 w_k1_ADM_s014
                                 + 0.448614 w_k1_AXP_s014
                                 + 0.619819 w_k1_DIS_s014
                                 + 0.738743 w_k1_FDX_s014
                                 + 0.572475 w_k1_IBM_s014
                                 + 0.728824 w_k1_INTC_s014
                                 + 0.629216 w_k1_KO_s014
                                 + 0.41827 w_k1_MAT_s014
                                 + 0.772403 w_k1_MSFT_s014
                                 + 0.75566 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.20802 w_k1_ADM_s014
                                 + 1.11549 w_k1_AXP_s014
                                 + 1.57375 w_k1_DIS_s014
                                 + 1.38522 w_k1_FDX_s014
                                 + 0.872757 w_k1_IBM_s014
                                 + 0.930527 w_k1_INTC_s014
                                 + 1.18677 w_k1_KO_s014 + 1.12467 w_k1_MAT_s014
                                 + 1.30132 w_k1_MSFT_s014
                                 + 1.22663 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.897504 w_k1_ADM_s014
                                 + 0.845768 w_k1_AXP_s014
                                 + 0.907384 w_k1_DIS_s014
                                 + 1.0079 w_k1_FDX_s014 + 0.70221 w_k1_IBM_s014
                                 + 1.04938 w_k1_INTC_s014
                                 + 0.84982 w_k1_KO_s014
                                 + 0.849521 w_k1_MAT_s014
                                 + 1.0884 w_k1_MSFT_s014
                                 + 0.962029 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.02796 w_k1_ADM_s014
                                 + 0.921574 w_k1_AXP_s014
                                 + 1.38005 w_k1_DIS_s014
                                 + 1.05607 w_k1_FDX_s014
                                 + 1.19412 w_k1_IBM_s014
                                 + 1.10181 w_k1_INTC_s014
                                 + 1.35114 w_k1_KO_s014 + 1.40495 w_k1_MAT_s014
                                 + 1.28305 w_k1_MSFT_s014
                                 + 1.10703 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.0379 w_k1_ADM_s014
                                 + 1.01775 w_k1_AXP_s014
                                 + 1.02596 w_k1_DIS_s014
                                 + 0.939069 w_k1_FDX_s014
                                 + 1.50201 w_k1_IBM_s014
                                 + 0.918088 w_k1_INTC_s014
                                 + 1.13289 w_k1_KO_s014 + 1.91847 w_k1_MAT_s014
                                 + 0.978915 w_k1_MSFT_s014
                                 + 0.959497 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.0856 w_k1_ADM_s014
                                 + 0.946368 w_k1_AXP_s014
                                 + 1.04936 w_k1_DIS_s014
                                 + 0.970486 w_k1_FDX_s014
                                 + 0.993203 w_k1_IBM_s014
                                 + 0.917077 w_k1_INTC_s014
                                 + 1.1989 w_k1_KO_s014 + 1.05388 w_k1_MAT_s014
                                 + 1.08193 w_k1_MSFT_s014
                                 + 1.05993 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.82658 w_k1_ADM_s015
                                 + 1.43023 w_k1_AXP_s015
                                 + 1.04353 w_k1_DIS_s015
                                 + 1.11288 w_k1_FDX_s015
                                 + 1.11975 w_k1_IBM_s015
                                 + 1.23368 w_k1_INTC_s015
                                 + 0.96369 w_k1_KO_s015 + 1.13766 w_k1_MAT_s015
                                 + 1.09387 w_k1_MSFT_s015
                                 + 1.04139 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.00547 w_k1_ADM_s015
                                 + 1.08109 w_k1_AXP_s015 + 0.782 w_k1_DIS_s015
                                 + 0.914154 w_k1_FDX_s015
                                 + 0.904472 w_k1_IBM_s015
                                 + 1.02416 w_k1_INTC_s015
                                 + 0.942072 w_k1_KO_s015
                                 + 1.22547 w_k1_MAT_s015
                                 + 0.904224 w_k1_MSFT_s015
                                 + 1.02584 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.22541 w_k1_ADM_s015
                                 + 1.54707 w_k1_AXP_s015
                                 + 1.56749 w_k1_DIS_s015
                                 + 1.47541 w_k1_FDX_s015
                                 + 1.05161 w_k1_IBM_s015
                                 + 1.45867 w_k1_INTC_s015
                                 + 1.47817 w_k1_KO_s015 + 1.05086 w_k1_MAT_s015
                                 + 1.43029 w_k1_MSFT_s015
                                 + 1.30056 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.793841 w_k1_ADM_s015
                                 + 0.667031 w_k1_AXP_s015
                                 + 0.794351 w_k1_DIS_s015
                                 + 0.797514 w_k1_FDX_s015
                                 + 0.932412 w_k1_IBM_s015
                                 + 0.865066 w_k1_INTC_s015
                                 + 0.923994 w_k1_KO_s015
                                 + 1.07458 w_k1_MAT_s015
                                 + 0.888703 w_k1_MSFT_s015
                                 + 0.907507 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.932931 w_k1_ADM_s015
                                 + 0.895256 w_k1_AXP_s015
                                 + 0.929676 w_k1_DIS_s015
                                 + 0.959298 w_k1_FDX_s015
                                 + 0.632538 w_k1_IBM_s015
                                 + 0.975483 w_k1_INTC_s015
                                 + 0.862993 w_k1_KO_s015
                                 + 0.874445 w_k1_MAT_s015
                                 + 0.982407 w_k1_MSFT_s015
                                 + 0.970995 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.898566 w_k1_ADM_s015
                                 + 1.03197 w_k1_AXP_s015
                                 + 1.24342 w_k1_DIS_s015
                                 + 0.995157 w_k1_FDX_s015
                                 + 1.3458 w_k1_IBM_s015
                                 + 1.19537 w_k1_INTC_s015
                                 + 1.36124 w_k1_KO_s015 + 1.658 w_k1_MAT_s015
                                 + 1.12392 w_k1_MSFT_s015
                                 + 0.967158 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.964685 w_k1_ADM_s015
                                 + 0.88902 w_k1_AXP_s015
                                 + 1.29791 w_k1_DIS_s015
                                 + 1.01285 w_k1_FDX_s015
                                 + 0.958067 w_k1_IBM_s015
                                 + 1.29118 w_k1_INTC_s015
                                 + 1.11554 w_k1_KO_s015
                                 + 0.621863 w_k1_MAT_s015
                                 + 1.14638 w_k1_MSFT_s015
                                 + 1.01173 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.939308 w_k1_ADM_s015
                                 + 1.11955 w_k1_AXP_s015
                                 + 1.20754 w_k1_DIS_s015 + 1.1541 w_k1_FDX_s015
                                 + 0.918984 w_k1_IBM_s015
                                 + 1.43638 w_k1_INTC_s015
                                 + 1.22246 w_k1_KO_s015
                                 + 0.906702 w_k1_MAT_s015
                                 + 1.1907 w_k1_MSFT_s015 + 1.01263 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.58303 w_k1_ADM_s016
                                 + 1.78003 w_k1_AXP_s016
                                 + 1.49976 w_k1_DIS_s016
                                 + 1.50059 w_k1_FDX_s016
                                 + 1.05268 w_k1_IBM_s016
                                 + 1.36042 w_k1_INTC_s016
                                 + 1.15337 w_k1_KO_s016 + 1.46613 w_k1_MAT_s016
                                 + 1.3452 w_k1_MSFT_s016 + 1.19864 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.19179 w_k1_ADM_s016
                                 + 1.09565 w_k1_AXP_s016
                                 + 1.33637 w_k1_DIS_s016
                                 + 1.19378 w_k1_FDX_s016 + 1.5266 w_k1_IBM_s016
                                 + 1.28533 w_k1_INTC_s016
                                 + 1.51263 w_k1_KO_s016 + 1.16016 w_k1_MAT_s016
                                 + 1.07634 w_k1_MSFT_s016
                                 + 1.08628 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.969159 w_k1_ADM_s016
                                 + 1.17788 w_k1_AXP_s016
                                 + 1.19365 w_k1_DIS_s016
                                 + 1.17631 w_k1_FDX_s016
                                 + 1.05028 w_k1_IBM_s016
                                 + 1.19575 w_k1_INTC_s016
                                 + 1.22307 w_k1_KO_s016 + 1.0982 w_k1_MAT_s016
                                 + 1.26819 w_k1_MSFT_s016
                                 + 1.10213 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.07489 w_k1_ADM_s016
                                 + 0.887739 w_k1_AXP_s016
                                 + 1.00007 w_k1_DIS_s016
                                 + 1.15282 w_k1_FDX_s016 + 0.7193 w_k1_IBM_s016
                                 + 1.29366 w_k1_INTC_s016
                                 + 0.920634 w_k1_KO_s016
                                 + 0.636674 w_k1_MAT_s016
                                 + 1.12045 w_k1_MSFT_s016
                                 + 0.943388 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.11012 w_k1_ADM_s016
                                 + 0.742571 w_k1_AXP_s016
                                 + 0.687481 w_k1_DIS_s016
                                 + 0.910209 w_k1_FDX_s016
                                 + 0.756089 w_k1_IBM_s016
                                 + 0.892197 w_k1_INTC_s016
                                 + 0.749138 w_k1_KO_s016
                                 + 0.654979 w_k1_MAT_s016
                                 + 0.954133 w_k1_MSFT_s016
                                 + 0.869692 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.906364 w_k1_ADM_s016
                                 + 0.962421 w_k1_AXP_s016
                                 + 0.784152 w_k1_DIS_s016
                                 + 0.766998 w_k1_FDX_s016
                                 + 1.37037 w_k1_IBM_s016
                                 + 0.928024 w_k1_INTC_s016
                                 + 1.00422 w_k1_KO_s016 + 1.13284 w_k1_MAT_s016
                                 + 0.929099 w_k1_MSFT_s016
                                 + 0.850364 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.939346 w_k1_ADM_s016
                                 + 0.790776 w_k1_AXP_s016
                                 + 0.640213 w_k1_DIS_s016
                                 + 0.746727 w_k1_FDX_s016
                                 + 0.811291 w_k1_IBM_s016
                                 + 0.868431 w_k1_INTC_s016
                                 + 0.727103 w_k1_KO_s016
                                 + 0.711684 w_k1_MAT_s016
                                 + 0.797898 w_k1_MSFT_s016
                                 + 0.86974 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.07472 w_k1_ADM_s016
                                 + 0.824993 w_k1_AXP_s016
                                 + 0.889056 w_k1_DIS_s016
                                 + 0.944967 w_k1_FDX_s016
                                 + 1.07319 w_k1_IBM_s016
                                 + 0.847858 w_k1_INTC_s016
                                 + 0.995998 w_k1_KO_s016
                                 + 0.793821 w_k1_MAT_s016
                                 + 0.88396 w_k1_MSFT_s016
                                 + 0.916069 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.13219 w_k1_ADM_s017
                                 + 1.09297 w_k1_AXP_s017
                                 + 1.80243 w_k1_DIS_s017
                                 + 1.19213 w_k1_FDX_s017
                                 + 1.43522 w_k1_IBM_s017
                                 + 1.2772 w_k1_INTC_s017 + 1.53084 w_k1_KO_s017
                                 + 1.73207 w_k1_MAT_s017
                                 + 1.35934 w_k1_MSFT_s017
                                 + 1.17632 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.12053 w_k1_ADM_s017
                                 + 0.884957 w_k1_AXP_s017
                                 + 1.01848 w_k1_DIS_s017
                                 + 1.00735 w_k1_FDX_s017
                                 + 0.981827 w_k1_IBM_s017
                                 + 1.01788 w_k1_INTC_s017
                                 + 1.27015 w_k1_KO_s017
                                 + 0.929017 w_k1_MAT_s017
                                 + 1.11912 w_k1_MSFT_s017
                                 + 0.970415 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.02149 w_k1_ADM_s017
                                 + 0.862867 w_k1_AXP_s017
                                 + 1.05013 w_k1_DIS_s017
                                 + 1.14083 w_k1_FDX_s017
                                 + 0.670607 w_k1_IBM_s017
                                 + 1.07894 w_k1_INTC_s017
                                 + 0.905358 w_k1_KO_s017
                                 + 0.762776 w_k1_MAT_s017
                                 + 1.18508 w_k1_MSFT_s017
                                 + 0.924245 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.840255 w_k1_ADM_s017
                                 + 0.6558 w_k1_AXP_s017
                                 + 0.711279 w_k1_DIS_s017
                                 + 0.706482 w_k1_FDX_s017
                                 + 0.766948 w_k1_IBM_s017
                                 + 0.762914 w_k1_INTC_s017
                                 + 0.830157 w_k1_KO_s017
                                 + 0.744022 w_k1_MAT_s017
                                 + 0.786369 w_k1_MSFT_s017
                                 + 0.77727 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.01509 w_k1_ADM_s017
                                 + 0.710369 w_k1_AXP_s017
                                 + 0.947351 w_k1_DIS_s017
                                 + 0.938944 w_k1_FDX_s017
                                 + 0.911724 w_k1_IBM_s017
                                 + 0.825354 w_k1_INTC_s017
                                 + 0.951989 w_k1_KO_s017
                                 + 0.803728 w_k1_MAT_s017
                                 + 0.923272 w_k1_MSFT_s017
                                 + 0.796767 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.938661 w_k1_ADM_s017
                                 + 0.782993 w_k1_AXP_s017
                                 + 0.791829 w_k1_DIS_s017
                                 + 0.790723 w_k1_FDX_s017
                                 + 1.13912 w_k1_IBM_s017
                                 + 0.896762 w_k1_INTC_s017
                                 + 1.04627 w_k1_KO_s017
                                 + 0.962514 w_k1_MAT_s017
                                 + 0.75529 w_k1_MSFT_s017
                                 + 0.909883 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.14817 w_k1_ADM_s017
                                 + 0.737199 w_k1_AXP_s017
                                 + 0.843368 w_k1_DIS_s017
                                 + 0.942459 w_k1_FDX_s017
                                 + 0.705107 w_k1_IBM_s017
                                 + 0.977934 w_k1_INTC_s017
                                 + 0.845081 w_k1_KO_s017
                                 + 0.563146 w_k1_MAT_s017
                                 + 0.892538 w_k1_MSFT_s017
                                 + 0.847996 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.4459 w_k1_ADM_s017
                                 + 1.20081 w_k1_AXP_s017
                                 + 1.22159 w_k1_DIS_s017
                                 + 1.11452 w_k1_FDX_s017
                                 + 1.13499 w_k1_IBM_s017
                                 + 1.17895 w_k1_INTC_s017
                                 + 1.23975 w_k1_KO_s017 + 1.19552 w_k1_MAT_s017
                                 + 1.04966 w_k1_MSFT_s017
                                 + 1.06277 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.904329 w_k1_ADM_s100
                                 + 0.893617 w_k1_AXP_s100
                                 + 0.967203 w_k1_DIS_s100
                                 + 0.993455 w_k1_FDX_s100
                                 + 1.22554 w_k1_IBM_s100
                                 + 1.12324 w_k1_INTC_s100
                                 + 1.21061 w_k1_KO_s100
                                 + 0.899229 w_k1_MAT_s100
                                 + 1.12984 w_k1_MSFT_s100
                                 + 1.05395 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.885797 w_k1_ADM_s100
                                 + 0.543438 w_k1_AXP_s100
                                 + 0.768486 w_k1_DIS_s100
                                 + 0.796764 w_k1_FDX_s100
                                 + 0.816813 w_k1_IBM_s100
                                 + 0.957045 w_k1_INTC_s100
                                 + 0.849701 w_k1_KO_s100
                                 + 0.452575 w_k1_MAT_s100
                                 + 0.940491 w_k1_MSFT_s100
                                 + 0.794901 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.36743 w_k1_ADM_s100
                                 + 1.3652 w_k1_AXP_s100 + 1.87242 w_k1_DIS_s100
                                 + 1.5759 w_k1_FDX_s100 + 1.01247 w_k1_IBM_s100
                                 + 1.44102 w_k1_INTC_s100
                                 + 1.21272 w_k1_KO_s100
                                 + 0.718128 w_k1_MAT_s100
                                 + 1.53834 w_k1_MSFT_s100 + 1.2993 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.861283 w_k1_ADM_s100
                                 + 0.579301 w_k1_AXP_s100
                                 + 0.774471 w_k1_DIS_s100
                                 + 0.795787 w_k1_FDX_s100
                                 + 1.18214 w_k1_IBM_s100
                                 + 0.93976 w_k1_INTC_s100
                                 + 1.05689 w_k1_KO_s100
                                 + 0.904642 w_k1_MAT_s100
                                 + 0.990301 w_k1_MSFT_s100
                                 + 0.848293 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.62278 w_k1_ADM_s100
                                 + 1.42326 w_k1_AXP_s100
                                 + 1.47185 w_k1_DIS_s100
                                 + 1.49359 w_k1_FDX_s100
                                 + 0.701793 w_k1_IBM_s100
                                 + 1.22506 w_k1_INTC_s100
                                 + 0.974986 w_k1_KO_s100 + 1.06 w_k1_MAT_s100
                                 + 1.60284 w_k1_MSFT_s100
                                 + 1.36591 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.25045 w_k1_ADM_s100
                                 + 1.57022 w_k1_AXP_s100
                                 + 1.46657 w_k1_DIS_s100
                                 + 1.26588 w_k1_FDX_s100
                                 + 1.49142 w_k1_IBM_s100
                                 + 1.0811 w_k1_INTC_s100 + 1.46417 w_k1_KO_s100
                                 + 1.60064 w_k1_MAT_s100
                                 + 1.20629 w_k1_MSFT_s100
                                 + 1.51419 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.964861 w_k1_ADM_s100
                                 + 1.0535 w_k1_AXP_s100
                                 + 0.866503 w_k1_DIS_s100
                                 + 0.853554 w_k1_FDX_s100
                                 + 1.76727 w_k1_IBM_s100
                                 + 0.954937 w_k1_INTC_s100
                                 + 1.30877 w_k1_KO_s100 + 1.87011 w_k1_MAT_s100
                                 + 1.04613 w_k1_MSFT_s100
                                 + 1.15159 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.963057 w_k1_ADM_s100
                                 + 1.01057 w_k1_AXP_s100
                                 + 1.14229 w_k1_DIS_s100
                                 + 1.10999 w_k1_FDX_s100
                                 + 0.704903 w_k1_IBM_s100
                                 + 1.00277 w_k1_INTC_s100
                                 + 0.82421 w_k1_KO_s100
                                 + 0.553548 w_k1_MAT_s100
                                 + 1.10226 w_k1_MSFT_s100
                                 + 1.05278 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.01497 w_k1_ADM_s101
                                 + 1.18859 w_k1_AXP_s101
                                 + 1.36164 w_k1_DIS_s101
                                 + 1.25763 w_k1_FDX_s101
                                 + 0.760486 w_k1_IBM_s101
                                 + 0.975504 w_k1_INTC_s101
                                 + 1.09998 w_k1_KO_s101 + 1.13345 w_k1_MAT_s101
                                 + 1.24947 w_k1_MSFT_s101
                                 + 1.17536 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.962034 w_k1_ADM_s101
                                 + 0.876914 w_k1_AXP_s101
                                 + 0.95849 w_k1_DIS_s101
                                 + 0.99156 w_k1_FDX_s101
                                 + 0.993721 w_k1_IBM_s101
                                 + 1.02653 w_k1_INTC_s101
                                 + 0.950473 w_k1_KO_s101
                                 + 0.983466 w_k1_MAT_s101
                                 + 1.10339 w_k1_MSFT_s101
                                 + 0.963606 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.858273 w_k1_ADM_s101
                                 + 1.0329 w_k1_AXP_s101 + 0.81722 w_k1_DIS_s101
                                 + 1.05068 w_k1_FDX_s101
                                 + 0.612667 w_k1_IBM_s101
                                 + 0.811287 w_k1_INTC_s101
                                 + 0.714664 w_k1_KO_s101
                                 + 0.717944 w_k1_MAT_s101
                                 + 0.921161 w_k1_MSFT_s101
                                 + 0.967516 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.533153 w_k1_ADM_s101
                                 + 0.486265 w_k1_AXP_s101
                                 + 0.598323 w_k1_DIS_s101
                                 + 0.61288 w_k1_FDX_s101
                                 + 0.894414 w_k1_IBM_s101
                                 + 0.932788 w_k1_INTC_s101
                                 + 0.8026 w_k1_KO_s101 + 0.680474 w_k1_MAT_s101
                                 + 0.874213 w_k1_MSFT_s101
                                 + 0.837921 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.700947 w_k1_ADM_s101
                                 + 0.761974 w_k1_AXP_s101
                                 + 0.672057 w_k1_DIS_s101
                                 + 0.838932 w_k1_FDX_s101
                                 + 0.658182 w_k1_IBM_s101
                                 + 0.871535 w_k1_INTC_s101
                                 + 0.581044 w_k1_KO_s101
                                 + 0.495006 w_k1_MAT_s101
                                 + 0.725893 w_k1_MSFT_s101
                                 + 0.810933 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.731044 w_k1_ADM_s101
                                 + 0.886449 w_k1_AXP_s101
                                 + 0.692482 w_k1_DIS_s101
                                 + 0.841556 w_k1_FDX_s101
                                 + 0.83188 w_k1_IBM_s101
                                 + 1.00806 w_k1_INTC_s101
                                 + 0.775589 w_k1_KO_s101
                                 + 1.01265 w_k1_MAT_s101
                                 + 0.944156 w_k1_MSFT_s101
                                 + 0.798102 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.20218 w_k1_ADM_s101
                                 + 1.16789 w_k1_AXP_s101
                                 + 1.09207 w_k1_DIS_s101
                                 + 1.10607 w_k1_FDX_s101
                                 + 1.13251 w_k1_IBM_s101
                                 + 0.963937 w_k1_INTC_s101 + 1.181 w_k1_KO_s101
                                 + 1.29676 w_k1_MAT_s101
                                 + 1.10157 w_k1_MSFT_s101
                                 + 1.04202 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.1522 w_k1_ADM_s101
                                 + 1.39508 w_k1_AXP_s101 + 1.3555 w_k1_DIS_s101
                                 + 1.06672 w_k1_FDX_s101
                                 + 1.60927 w_k1_IBM_s101
                                 + 1.12899 w_k1_INTC_s101
                                 + 1.33203 w_k1_KO_s101 + 2.13581 w_k1_MAT_s101
                                 + 1.25412 w_k1_MSFT_s101
                                 + 1.18768 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.11003 w_k1_ADM_s102
                                 + 1.00483 w_k1_AXP_s102
                                 + 0.927539 w_k1_DIS_s102
                                 + 1.04481 w_k1_FDX_s102
                                 + 0.810891 w_k1_IBM_s102
                                 + 1.21722 w_k1_INTC_s102
                                 + 1.06534 w_k1_KO_s102
                                 + 0.696876 w_k1_MAT_s102
                                 + 1.12553 w_k1_MSFT_s102
                                 + 1.05195 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.34807 w_k1_ADM_s102
                                 + 1.15819 w_k1_AXP_s102
                                 + 1.16246 w_k1_DIS_s102
                                 + 1.14637 w_k1_FDX_s102
                                 + 1.16499 w_k1_IBM_s102
                                 + 1.22817 w_k1_INTC_s102
                                 + 1.11464 w_k1_KO_s102
                                 + 0.870252 w_k1_MAT_s102
                                 + 1.17737 w_k1_MSFT_s102
                                 + 1.04294 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.944561 w_k1_ADM_s102
                                 + 0.721215 w_k1_AXP_s102
                                 + 0.710955 w_k1_DIS_s102
                                 + 0.866158 w_k1_FDX_s102
                                 + 0.811103 w_k1_IBM_s102
                                 + 0.952679 w_k1_INTC_s102
                                 + 0.64317 w_k1_KO_s102
                                 + 0.478139 w_k1_MAT_s102
                                 + 0.842331 w_k1_MSFT_s102
                                 + 0.82572 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.942593 w_k1_ADM_s102
                                 + 0.942401 w_k1_AXP_s102
                                 + 0.689139 w_k1_DIS_s102
                                 + 0.823588 w_k1_FDX_s102
                                 + 1.09266 w_k1_IBM_s102
                                 + 1.12399 w_k1_INTC_s102
                                 + 0.946723 w_k1_KO_s102
                                 + 0.750541 w_k1_MAT_s102
                                 + 0.879026 w_k1_MSFT_s102
                                 + 0.925372 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.07469 w_k1_ADM_s102
                                 + 1.64781 w_k1_AXP_s102 + 1.6691 w_k1_DIS_s102
                                 + 1.55426 w_k1_FDX_s102
                                 + 1.10581 w_k1_IBM_s102
                                 + 2.07386 w_k1_INTC_s102 + 1.1561 w_k1_KO_s102
                                 + 0.735241 w_k1_MAT_s102
                                 + 1.46105 w_k1_MSFT_s102
                                 + 1.15717 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.964928 w_k1_ADM_s102
                                 + 0.711488 w_k1_AXP_s102
                                 + 0.74417 w_k1_DIS_s102
                                 + 0.97974 w_k1_FDX_s102
                                 + 0.660484 w_k1_IBM_s102
                                 + 0.956193 w_k1_INTC_s102
                                 + 0.719609 w_k1_KO_s102
                                 + 0.477952 w_k1_MAT_s102
                                 + 0.977168 w_k1_MSFT_s102
                                 + 0.866274 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.14021 w_k1_ADM_s102
                                 + 1.12186 w_k1_AXP_s102
                                 + 0.999014 w_k1_DIS_s102
                                 + 0.952864 w_k1_FDX_s102
                                 + 1.57156 w_k1_IBM_s102
                                 + 1.21853 w_k1_INTC_s102
                                 + 1.29053 w_k1_KO_s102 + 1.31961 w_k1_MAT_s102
                                 + 1.13429 w_k1_MSFT_s102
                                 + 1.06619 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.06098 w_k1_ADM_s102
                                 + 1.07767 w_k1_AXP_s102
                                 + 1.08953 w_k1_DIS_s102
                                 + 1.16144 w_k1_FDX_s102
                                 + 0.815322 w_k1_IBM_s102
                                 + 1.13763 w_k1_INTC_s102
                                 + 0.950486 w_k1_KO_s102
                                 + 0.679493 w_k1_MAT_s102
                                 + 1.18242 w_k1_MSFT_s102
                                 + 1.05505 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.14988 w_k1_ADM_s103
                                 + 0.815543 w_k1_AXP_s103
                                 + 0.840956 w_k1_DIS_s103
                                 + 0.892889 w_k1_FDX_s103
                                 + 1.18612 w_k1_IBM_s103
                                 + 0.802029 w_k1_INTC_s103
                                 + 0.828904 w_k1_KO_s103 + 1.2031 w_k1_MAT_s103
                                 + 0.950163 w_k1_MSFT_s103
                                 + 0.791157 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.800675 w_k1_ADM_s103
                                 + 0.993575 w_k1_AXP_s103
                                 + 1.06908 w_k1_DIS_s103
                                 + 0.882925 w_k1_FDX_s103
                                 + 1.32146 w_k1_IBM_s103
                                 + 0.889497 w_k1_INTC_s103
                                 + 1.11542 w_k1_KO_s103 + 2.02896 w_k1_MAT_s103
                                 + 1.00069 w_k1_MSFT_s103
                                 + 0.998126 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.23993 w_k1_ADM_s103
                                 + 1.51791 w_k1_AXP_s103
                                 + 1.42887 w_k1_DIS_s103
                                 + 1.19143 w_k1_FDX_s103
                                 + 1.82702 w_k1_IBM_s103
                                 + 1.36504 w_k1_INTC_s103
                                 + 1.29268 w_k1_KO_s103 + 2.51177 w_k1_MAT_s103
                                 + 1.25552 w_k1_MSFT_s103
                                 + 0.988459 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.25478 w_k1_ADM_s103
                                 + 0.945063 w_k1_AXP_s103
                                 + 1.50879 w_k1_DIS_s103
                                 + 1.26846 w_k1_FDX_s103
                                 + 1.19315 w_k1_IBM_s103
                                 + 1.20053 w_k1_INTC_s103
                                 + 1.33282 w_k1_KO_s103 + 1.4147 w_k1_MAT_s103
                                 + 1.39074 w_k1_MSFT_s103
                                 + 1.19397 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.34211 w_k1_ADM_s103
                                 + 1.69857 w_k1_AXP_s103
                                 + 1.68624 w_k1_DIS_s103
                                 + 1.67067 w_k1_FDX_s103
                                 + 0.963551 w_k1_IBM_s103
                                 + 1.36302 w_k1_INTC_s103
                                 + 1.22035 w_k1_KO_s103 + 1.57802 w_k1_MAT_s103
                                 + 1.57319 w_k1_MSFT_s103
                                 + 1.20961 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.17743 w_k1_ADM_s103
                                 + 0.921959 w_k1_AXP_s103
                                 + 1.21269 w_k1_DIS_s103
                                 + 1.19567 w_k1_FDX_s103
                                 + 0.88995 w_k1_IBM_s103
                                 + 1.3556 w_k1_INTC_s103 + 1.03942 w_k1_KO_s103
                                 + 0.696383 w_k1_MAT_s103
                                 + 1.25842 w_k1_MSFT_s103
                                 + 0.902625 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.07922 w_k1_ADM_s103
                                 + 1.40246 w_k1_AXP_s103
                                 + 0.970039 w_k1_DIS_s103
                                 + 1.13706 w_k1_FDX_s103
                                 + 0.743496 w_k1_IBM_s103
                                 + 1.54409 w_k1_INTC_s103
                                 + 0.789511 w_k1_KO_s103
                                 + 0.938711 w_k1_MAT_s103
                                 + 1.13451 w_k1_MSFT_s103
                                 + 0.97741 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.972374 w_k1_ADM_s103
                                 + 0.788304 w_k1_AXP_s103
                                 + 0.854738 w_k1_DIS_s103
                                 + 0.90907 w_k1_FDX_s103
                                 + 0.848377 w_k1_IBM_s103
                                 + 1.0627 w_k1_INTC_s103 + 0.84765 w_k1_KO_s103
                                 + 1.17318 w_k1_MAT_s103
                                 + 1.0211 w_k1_MSFT_s103
                                 + 0.923419 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.29969 w_k1_ADM_s104
                                 + 1.40004 w_k1_AXP_s104
                                 + 1.43913 w_k1_DIS_s104
                                 + 1.31272 w_k1_FDX_s104
                                 + 1.23679 w_k1_IBM_s104
                                 + 0.913723 w_k1_INTC_s104
                                 + 1.23072 w_k1_KO_s104 + 1.39785 w_k1_MAT_s104
                                 + 1.02442 w_k1_MSFT_s104
                                 + 1.22784 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.708411 w_k1_ADM_s104
                                 + 0.772684 w_k1_AXP_s104
                                 + 0.767176 w_k1_DIS_s104
                                 + 0.867231 w_k1_FDX_s104
                                 + 0.537201 w_k1_IBM_s104
                                 + 0.930812 w_k1_INTC_s104
                                 + 0.696939 w_k1_KO_s104
                                 + 0.41118 w_k1_MAT_s104
                                 + 0.910206 w_k1_MSFT_s104
                                 + 0.717622 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.901218 w_k1_ADM_s104
                                 + 0.707537 w_k1_AXP_s104
                                 + 1.25937 w_k1_DIS_s104
                                 + 1.01247 w_k1_FDX_s104
                                 + 0.826066 w_k1_IBM_s104
                                 + 0.931577 w_k1_INTC_s104
                                 + 1.05641 w_k1_KO_s104
                                 + 0.571052 w_k1_MAT_s104
                                 + 1.03608 w_k1_MSFT_s104
                                 + 0.915861 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.04767 w_k1_ADM_s104
                                 + 1.12719 w_k1_AXP_s104
                                 + 1.02516 w_k1_DIS_s104
                                 + 1.11016 w_k1_FDX_s104
                                 + 0.962862 w_k1_IBM_s104
                                 + 1.12005 w_k1_INTC_s104
                                 + 1.00737 w_k1_KO_s104 + 1.12737 w_k1_MAT_s104
                                 + 1.05079 w_k1_MSFT_s104
                                 + 1.02537 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.73586 w_k1_ADM_s104
                                 + 1.08683 w_k1_AXP_s104
                                 + 0.951431 w_k1_DIS_s104
                                 + 0.845609 w_k1_FDX_s104
                                 + 1.49652 w_k1_IBM_s104
                                 + 0.960678 w_k1_INTC_s104
                                 + 1.2154 w_k1_KO_s104 + 1.79916 w_k1_MAT_s104
                                 + 0.839999 w_k1_MSFT_s104
                                 + 1.07752 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.990863 w_k1_ADM_s104
                                 + 0.86581 w_k1_AXP_s104
                                 + 0.769478 w_k1_DIS_s104
                                 + 0.915316 w_k1_FDX_s104
                                 + 0.761595 w_k1_IBM_s104
                                 + 0.910755 w_k1_INTC_s104
                                 + 0.856039 w_k1_KO_s104
                                 + 0.718285 w_k1_MAT_s104
                                 + 0.786764 w_k1_MSFT_s104
                                 + 0.860863 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.70626 w_k1_ADM_s104
                                 + 0.585455 w_k1_AXP_s104
                                 + 0.711063 w_k1_DIS_s104
                                 + 0.684436 w_k1_FDX_s104
                                 + 0.936898 w_k1_IBM_s104
                                 + 0.674494 w_k1_INTC_s104
                                 + 0.894051 w_k1_KO_s104
                                 + 0.994001 w_k1_MAT_s104
                                 + 0.700674 w_k1_MSFT_s104
                                 + 0.749149 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.15865 w_k1_ADM_s104
                                 + 1.25473 w_k1_AXP_s104 + 1.1032 w_k1_DIS_s104
                                 + 1.07904 w_k1_FDX_s104
                                 + 1.31201 w_k1_IBM_s104
                                 + 0.808581 w_k1_INTC_s104
                                 + 1.02267 w_k1_KO_s104 + 1.49482 w_k1_MAT_s104
                                 + 0.870788 w_k1_MSFT_s104
                                 + 1.01032 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.998302 w_k1_ADM_s105
                                 + 0.935346 w_k1_AXP_s105
                                 + 0.693427 w_k1_DIS_s105
                                 + 0.906721 w_k1_FDX_s105
                                 + 0.849533 w_k1_IBM_s105
                                 + 0.832652 w_k1_INTC_s105
                                 + 0.714659 w_k1_KO_s105
                                 + 0.790501 w_k1_MAT_s105
                                 + 0.735323 w_k1_MSFT_s105
                                 + 0.840854 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.30238 w_k1_ADM_s105
                                 + 0.828872 w_k1_AXP_s105
                                 + 1.11342 w_k1_DIS_s105
                                 + 1.10747 w_k1_FDX_s105
                                 + 0.807618 w_k1_IBM_s105
                                 + 1.13692 w_k1_INTC_s105
                                 + 0.867174 w_k1_KO_s105
                                 + 0.523735 w_k1_MAT_s105
                                 + 1.08933 w_k1_MSFT_s105
                                 + 0.996568 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.22946 w_k1_ADM_s105
                                 + 1.2876 w_k1_AXP_s105 + 1.10832 w_k1_DIS_s105
                                 + 1.08912 w_k1_FDX_s105
                                 + 1.20112 w_k1_IBM_s105
                                 + 0.815874 w_k1_INTC_s105
                                 + 1.07282 w_k1_KO_s105 + 1.33537 w_k1_MAT_s105
                                 + 0.884718 w_k1_MSFT_s105
                                 + 1.15352 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.717871 w_k1_ADM_s105
                                 + 0.705676 w_k1_AXP_s105
                                 + 0.979552 w_k1_DIS_s105
                                 + 0.922662 w_k1_FDX_s105
                                 + 0.92457 w_k1_IBM_s105
                                 + 1.00778 w_k1_INTC_s105
                                 + 0.991661 w_k1_KO_s105
                                 + 0.648995 w_k1_MAT_s105
                                 + 0.96449 w_k1_MSFT_s105
                                 + 1.09534 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.22979 w_k1_ADM_s105
                                 + 1.22417 w_k1_AXP_s105
                                 + 1.54036 w_k1_DIS_s105 + 1.3308 w_k1_FDX_s105
                                 + 1.03814 w_k1_IBM_s105
                                 + 1.17752 w_k1_INTC_s105
                                 + 1.33036 w_k1_KO_s105 + 1.08354 w_k1_MAT_s105
                                 + 1.15661 w_k1_MSFT_s105 + 1.4325 w_k1_PG_s105
                                 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.747608 w_k1_ADM_s105
                                 + 0.823813 w_k1_AXP_s105
                                 + 0.872623 w_k1_DIS_s105
                                 + 0.873968 w_k1_FDX_s105
                                 + 1.1512 w_k1_IBM_s105
                                 + 0.858001 w_k1_INTC_s105
                                 + 0.993758 w_k1_KO_s105
                                 + 1.08557 w_k1_MAT_s105
                                 + 0.902674 w_k1_MSFT_s105
                                 + 1.04746 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.884129 w_k1_ADM_s105
                                 + 0.590399 w_k1_AXP_s105
                                 + 0.574513 w_k1_DIS_s105
                                 + 0.717574 w_k1_FDX_s105
                                 + 0.793828 w_k1_IBM_s105
                                 + 0.680574 w_k1_INTC_s105
                                 + 0.817003 w_k1_KO_s105
                                 + 0.767668 w_k1_MAT_s105
                                 + 0.737877 w_k1_MSFT_s105
                                 + 0.900296 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.852652 w_k1_ADM_s105
                                 + 0.788051 w_k1_AXP_s105
                                 + 0.723589 w_k1_DIS_s105
                                 + 0.89365 w_k1_FDX_s105
                                 + 0.689841 w_k1_IBM_s105
                                 + 1.02858 w_k1_INTC_s105
                                 + 0.770751 w_k1_KO_s105
                                 + 0.497703 w_k1_MAT_s105
                                 + 0.82757 w_k1_MSFT_s105
                                 + 0.947608 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.15561 w_k1_ADM_s106
                                 + 1.37018 w_k1_AXP_s106
                                 + 1.20425 w_k1_DIS_s106
                                 + 1.18427 w_k1_FDX_s106
                                 + 0.761319 w_k1_IBM_s106
                                 + 1.41468 w_k1_INTC_s106
                                 + 1.02649 w_k1_KO_s106 + 1.13968 w_k1_MAT_s106
                                 + 1.21841 w_k1_MSFT_s106
                                 + 1.12534 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.09673 w_k1_ADM_s106
                                 + 1.81815 w_k1_AXP_s106 + 1.3991 w_k1_DIS_s106
                                 + 1.27609 w_k1_FDX_s106
                                 + 1.53268 w_k1_IBM_s106
                                 + 1.62262 w_k1_INTC_s106
                                 + 1.38598 w_k1_KO_s106 + 1.93984 w_k1_MAT_s106
                                 + 1.22083 w_k1_MSFT_s106
                                 + 1.23878 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.833943 w_k1_ADM_s106
                                 + 0.8882 w_k1_AXP_s106 + 1.00423 w_k1_DIS_s106
                                 + 0.930679 w_k1_FDX_s106
                                 + 1.26872 w_k1_IBM_s106
                                 + 1.14352 w_k1_INTC_s106
                                 + 1.26193 w_k1_KO_s106 + 1.48405 w_k1_MAT_s106
                                 + 0.995481 w_k1_MSFT_s106
                                 + 0.996038 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.940428 w_k1_ADM_s106
                                 + 0.898875 w_k1_AXP_s106
                                 + 0.760626 w_k1_DIS_s106
                                 + 0.912438 w_k1_FDX_s106
                                 + 0.647462 w_k1_IBM_s106
                                 + 0.896688 w_k1_INTC_s106
                                 + 0.728184 w_k1_KO_s106
                                 + 0.706666 w_k1_MAT_s106
                                 + 0.752833 w_k1_MSFT_s106
                                 + 0.855533 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.822164 w_k1_ADM_s106
                                 + 0.774729 w_k1_AXP_s106
                                 + 0.663353 w_k1_DIS_s106
                                 + 0.72182 w_k1_FDX_s106
                                 + 1.07235 w_k1_IBM_s106
                                 + 0.901654 w_k1_INTC_s106
                                 + 0.886895 w_k1_KO_s106
                                 + 1.14968 w_k1_MAT_s106
                                 + 0.858134 w_k1_MSFT_s106
                                 + 0.937929 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.846744 w_k1_ADM_s106
                                 + 0.912416 w_k1_AXP_s106
                                 + 0.600784 w_k1_DIS_s106
                                 + 0.782022 w_k1_FDX_s106
                                 + 0.635536 w_k1_IBM_s106
                                 + 0.912203 w_k1_INTC_s106
                                 + 0.740711 w_k1_KO_s106
                                 + 0.98045 w_k1_MAT_s106
                                 + 0.924064 w_k1_MSFT_s106
                                 + 0.892872 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.938314 w_k1_ADM_s106
                                 + 1.18071 w_k1_AXP_s106
                                 + 0.785523 w_k1_DIS_s106
                                 + 0.882064 w_k1_FDX_s106
                                 + 0.91635 w_k1_IBM_s106
                                 + 0.972758 w_k1_INTC_s106
                                 + 0.814144 w_k1_KO_s106
                                 + 1.21851 w_k1_MAT_s106
                                 + 0.811405 w_k1_MSFT_s106
                                 + 0.871776 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.911591 w_k1_ADM_s106
                                 + 1.43045 w_k1_AXP_s106
                                 + 1.19806 w_k1_DIS_s106
                                 + 1.08727 w_k1_FDX_s106
                                 + 1.10019 w_k1_IBM_s106
                                 + 1.47875 w_k1_INTC_s106
                                 + 1.13402 w_k1_KO_s106 + 1.46393 w_k1_MAT_s106
                                 + 1.16843 w_k1_MSFT_s106
                                 + 0.942108 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.842305 w_k1_ADM_s107
                                 + 0.923589 w_k1_AXP_s107
                                 + 0.744943 w_k1_DIS_s107
                                 + 0.812266 w_k1_FDX_s107
                                 + 1.39816 w_k1_IBM_s107
                                 + 0.990558 w_k1_INTC_s107
                                 + 1.00046 w_k1_KO_s107 + 0.83089 w_k1_MAT_s107
                                 + 0.808846 w_k1_MSFT_s107
                                 + 0.932122 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.45522 w_k1_ADM_s107
                                 + 1.09546 w_k1_AXP_s107
                                 + 1.19082 w_k1_DIS_s107
                                 + 1.20996 w_k1_FDX_s107
                                 + 0.742356 w_k1_IBM_s107
                                 + 1.12468 w_k1_INTC_s107
                                 + 0.978743 w_k1_KO_s107
                                 + 0.645348 w_k1_MAT_s107
                                 + 1.26736 w_k1_MSFT_s107
                                 + 1.10018 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.817877 w_k1_ADM_s107
                                 + 1.09331 w_k1_AXP_s107
                                 + 0.812343 w_k1_DIS_s107
                                 + 0.807069 w_k1_FDX_s107
                                 + 1.14912 w_k1_IBM_s107
                                 + 0.896296 w_k1_INTC_s107
                                 + 1.00171 w_k1_KO_s107 + 1.14866 w_k1_MAT_s107
                                 + 0.959874 w_k1_MSFT_s107
                                 + 1.08848 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.34793 w_k1_ADM_s107
                                 + 1.25619 w_k1_AXP_s107 + 1.1445 w_k1_DIS_s107
                                 + 1.13203 w_k1_FDX_s107
                                 + 1.21804 w_k1_IBM_s107
                                 + 1.23661 w_k1_INTC_s107
                                 + 1.26345 w_k1_KO_s107 + 1.31332 w_k1_MAT_s107
                                 + 1.14752 w_k1_MSFT_s107
                                 + 1.12941 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.44535 w_k1_ADM_s107
                                 + 1.52716 w_k1_AXP_s107
                                 + 1.46597 w_k1_DIS_s107
                                 + 1.32911 w_k1_FDX_s107
                                 + 1.13099 w_k1_IBM_s107
                                 + 1.34877 w_k1_INTC_s107
                                 + 1.38098 w_k1_KO_s107 + 1.03955 w_k1_MAT_s107
                                 + 1.33624 w_k1_MSFT_s107
                                 + 1.31126 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.39241 w_k1_ADM_s107
                                 + 1.33095 w_k1_AXP_s107
                                 + 1.17903 w_k1_DIS_s107
                                 + 0.95531 w_k1_FDX_s107
                                 + 1.70503 w_k1_IBM_s107
                                 + 1.20938 w_k1_INTC_s107
                                 + 1.49073 w_k1_KO_s107 + 1.2036 w_k1_MAT_s107
                                 + 1.17814 w_k1_MSFT_s107
                                 + 1.27135 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.22554 w_k1_ADM_s107
                                 + 0.919013 w_k1_AXP_s107
                                 + 0.612129 w_k1_DIS_s107
                                 + 0.817216 w_k1_FDX_s107
                                 + 0.685832 w_k1_IBM_s107
                                 + 0.789836 w_k1_INTC_s107
                                 + 0.768653 w_k1_KO_s107
                                 + 0.538651 w_k1_MAT_s107
                                 + 0.832945 w_k1_MSFT_s107
                                 + 0.889744 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.847905 w_k1_ADM_s107
                                 + 0.948025 w_k1_AXP_s107
                                 + 0.931429 w_k1_DIS_s107
                                 + 0.921944 w_k1_FDX_s107
                                 + 0.847911 w_k1_IBM_s107
                                 + 1.15861 w_k1_INTC_s107
                                 + 1.03813 w_k1_KO_s107
                                 + 0.801008 w_k1_MAT_s107
                                 + 1.15287 w_k1_MSFT_s107
                                 + 1.09819 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.00198 w_k1_ADM_s110
                                 + 0.78331 w_k1_AXP_s110
                                 + 0.918246 w_k1_DIS_s110
                                 + 0.94158 w_k1_FDX_s110
                                 + 0.89877 w_k1_IBM_s110
                                 + 0.925492 w_k1_INTC_s110
                                 + 0.951951 w_k1_KO_s110
                                 + 0.893097 w_k1_MAT_s110
                                 + 0.98025 w_k1_MSFT_s110
                                 + 0.941484 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.02992 w_k1_ADM_s110
                                 + 0.918828 w_k1_AXP_s110
                                 + 1.36819 w_k1_DIS_s110
                                 + 1.05555 w_k1_FDX_s110
                                 + 1.16498 w_k1_IBM_s110
                                 + 0.973651 w_k1_INTC_s110
                                 + 1.18687 w_k1_KO_s110 + 1.021 w_k1_MAT_s110
                                 + 1.12641 w_k1_MSFT_s110
                                 + 1.00945 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.76528 w_k1_ADM_s110
                                 + 1.04159 w_k1_AXP_s110
                                 + 1.07126 w_k1_DIS_s110
                                 + 0.993177 w_k1_FDX_s110
                                 + 1.35273 w_k1_IBM_s110
                                 + 0.998276 w_k1_INTC_s110
                                 + 1.16701 w_k1_KO_s110 + 1.35824 w_k1_MAT_s110
                                 + 0.887092 w_k1_MSFT_s110
                                 + 0.94262 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.851555 w_k1_ADM_s110
                                 + 0.779369 w_k1_AXP_s110
                                 + 0.857196 w_k1_DIS_s110
                                 + 0.929563 w_k1_FDX_s110
                                 + 0.777749 w_k1_IBM_s110
                                 + 0.774626 w_k1_INTC_s110
                                 + 0.723351 w_k1_KO_s110
                                 + 0.787406 w_k1_MAT_s110
                                 + 0.848566 w_k1_MSFT_s110
                                 + 0.757622 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.03034 w_k1_ADM_s110
                                 + 1.5022 w_k1_AXP_s110 + 1.64324 w_k1_DIS_s110
                                 + 1.24051 w_k1_FDX_s110
                                 + 1.61146 w_k1_IBM_s110
                                 + 1.25567 w_k1_INTC_s110
                                 + 1.32217 w_k1_KO_s110 + 2.14098 w_k1_MAT_s110
                                 + 1.19672 w_k1_MSFT_s110
                                 + 1.17052 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.820427 w_k1_ADM_s110
                                 + 1.00925 w_k1_AXP_s110
                                 + 1.15597 w_k1_DIS_s110
                                 + 1.06701 w_k1_FDX_s110
                                 + 0.967997 w_k1_IBM_s110
                                 + 0.853856 w_k1_INTC_s110
                                 + 0.937849 w_k1_KO_s110 + 1.0807 w_k1_MAT_s110
                                 + 0.934484 w_k1_MSFT_s110
                                 + 0.986456 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.22093 w_k1_ADM_s110
                                 + 1.24892 w_k1_AXP_s110
                                 + 1.42559 w_k1_DIS_s110
                                 + 1.28331 w_k1_FDX_s110
                                 + 1.12578 w_k1_IBM_s110
                                 + 0.871062 w_k1_INTC_s110
                                 + 1.13306 w_k1_KO_s110 + 1.18504 w_k1_MAT_s110
                                 + 1.21421 w_k1_MSFT_s110
                                 + 1.12108 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.726853 w_k1_ADM_s110
                                 + 0.985106 w_k1_AXP_s110
                                 + 0.791046 w_k1_DIS_s110
                                 + 0.776512 w_k1_FDX_s110
                                 + 1.07498 w_k1_IBM_s110
                                 + 0.721588 w_k1_INTC_s110
                                 + 0.809692 w_k1_KO_s110
                                 + 0.948746 w_k1_MAT_s110
                                 + 0.70184 w_k1_MSFT_s110 + 0.7588 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.00414 w_k1_ADM_s111
                                 + 0.823057 w_k1_AXP_s111
                                 + 0.927731 w_k1_DIS_s111
                                 + 0.928859 w_k1_FDX_s111
                                 + 0.827027 w_k1_IBM_s111
                                 + 0.946398 w_k1_INTC_s111
                                 + 0.890971 w_k1_KO_s111
                                 + 0.592177 w_k1_MAT_s111
                                 + 1.04396 w_k1_MSFT_s111
                                 + 0.92998 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.838353 w_k1_ADM_s111
                                 + 0.712494 w_k1_AXP_s111
                                 + 0.751482 w_k1_DIS_s111
                                 + 0.811255 w_k1_FDX_s111
                                 + 0.743721 w_k1_IBM_s111
                                 + 0.756768 w_k1_INTC_s111
                                 + 0.817047 w_k1_KO_s111
                                 + 0.555976 w_k1_MAT_s111
                                 + 0.857294 w_k1_MSFT_s111
                                 + 0.91625 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.905949 w_k1_ADM_s111
                                 + 0.933919 w_k1_AXP_s111
                                 + 0.682248 w_k1_DIS_s111
                                 + 0.875617 w_k1_FDX_s111
                                 + 0.95574 w_k1_IBM_s111
                                 + 0.921598 w_k1_INTC_s111
                                 + 0.842565 w_k1_KO_s111
                                 + 1.21119 w_k1_MAT_s111
                                 + 0.869249 w_k1_MSFT_s111
                                 + 0.951754 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.38248 w_k1_ADM_s111
                                 + 0.995776 w_k1_AXP_s111
                                 + 1.43042 w_k1_DIS_s111
                                 + 1.08874 w_k1_FDX_s111
                                 + 1.59901 w_k1_IBM_s111
                                 + 1.03418 w_k1_INTC_s111
                                 + 1.37962 w_k1_KO_s111 + 1.41057 w_k1_MAT_s111
                                 + 1.15326 w_k1_MSFT_s111
                                 + 1.14815 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.05863 w_k1_ADM_s111
                                 + 1.17639 w_k1_AXP_s111
                                 + 1.58058 w_k1_DIS_s111
                                 + 1.44002 w_k1_FDX_s111
                                 + 0.95137 w_k1_IBM_s111
                                 + 1.18917 w_k1_INTC_s111
                                 + 1.23636 w_k1_KO_s111 + 1.0991 w_k1_MAT_s111
                                 + 1.41556 w_k1_MSFT_s111
                                 + 1.17935 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.26343 w_k1_ADM_s111
                                 + 1.06513 w_k1_AXP_s111
                                 + 1.09706 w_k1_DIS_s111
                                 + 1.10883 w_k1_FDX_s111
                                 + 0.909909 w_k1_IBM_s111
                                 + 1.08869 w_k1_INTC_s111
                                 + 0.938763 w_k1_KO_s111
                                 + 1.03253 w_k1_MAT_s111
                                 + 1.07106 w_k1_MSFT_s111
                                 + 1.03044 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.943778 w_k1_ADM_s111
                                 + 1.26045 w_k1_AXP_s111
                                 + 1.16962 w_k1_DIS_s111
                                 + 1.06491 w_k1_FDX_s111
                                 + 1.17467 w_k1_IBM_s111
                                 + 1.11717 w_k1_INTC_s111
                                 + 1.16693 w_k1_KO_s111 + 1.12165 w_k1_MAT_s111
                                 + 1.02739 w_k1_MSFT_s111
                                 + 1.06482 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.02952 w_k1_ADM_s111
                                 + 0.843493 w_k1_AXP_s111
                                 + 0.758517 w_k1_DIS_s111
                                 + 0.742799 w_k1_FDX_s111
                                 + 1.07147 w_k1_IBM_s111
                                 + 0.702552 w_k1_INTC_s111
                                 + 0.814492 w_k1_KO_s111
                                 + 0.918439 w_k1_MAT_s111
                                 + 0.745779 w_k1_MSFT_s111
                                 + 1.0616 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.11126 w_k1_ADM_s112
                                 + 0.910034 w_k1_AXP_s112
                                 + 1.19667 w_k1_DIS_s112
                                 + 0.966083 w_k1_FDX_s112
                                 + 0.940843 w_k1_IBM_s112
                                 + 0.967874 w_k1_INTC_s112
                                 + 0.97671 w_k1_KO_s112 + 1.08658 w_k1_MAT_s112
                                 + 0.951492 w_k1_MSFT_s112
                                 + 1.18321 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.45938 w_k1_ADM_s112
                                 + 1.65512 w_k1_AXP_s112
                                 + 1.74456 w_k1_DIS_s112
                                 + 1.32403 w_k1_FDX_s112
                                 + 1.28652 w_k1_IBM_s112
                                 + 1.34377 w_k1_INTC_s112
                                 + 1.31299 w_k1_KO_s112 + 2.00177 w_k1_MAT_s112
                                 + 1.35336 w_k1_MSFT_s112
                                 + 1.49262 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.961223 w_k1_ADM_s112
                                 + 1.01711 w_k1_AXP_s112
                                 + 0.978747 w_k1_DIS_s112
                                 + 1.10581 w_k1_FDX_s112
                                 + 0.736602 w_k1_IBM_s112
                                 + 0.893035 w_k1_INTC_s112
                                 + 0.749664 w_k1_KO_s112
                                 + 1.07274 w_k1_MAT_s112
                                 + 0.874558 w_k1_MSFT_s112
                                 + 1.0576 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.814694 w_k1_ADM_s112
                                 + 0.782625 w_k1_AXP_s112
                                 + 0.696204 w_k1_DIS_s112
                                 + 0.715987 w_k1_FDX_s112
                                 + 0.974294 w_k1_IBM_s112
                                 + 0.816063 w_k1_INTC_s112
                                 + 0.764449 w_k1_KO_s112
                                 + 0.944651 w_k1_MAT_s112
                                 + 0.713041 w_k1_MSFT_s112
                                 + 0.938073 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.758613 w_k1_ADM_s112
                                 + 0.983661 w_k1_AXP_s112
                                 + 0.70149 w_k1_DIS_s112
                                 + 0.827724 w_k1_FDX_s112
                                 + 0.734566 w_k1_IBM_s112
                                 + 0.696127 w_k1_INTC_s112
                                 + 0.625552 w_k1_KO_s112
                                 + 1.22801 w_k1_MAT_s112
                                 + 0.695834 w_k1_MSFT_s112
                                 + 0.878804 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.07512 w_k1_ADM_s112
                                 + 0.648633 w_k1_AXP_s112
                                 + 0.773872 w_k1_DIS_s112
                                 + 0.864959 w_k1_FDX_s112
                                 + 0.690193 w_k1_IBM_s112
                                 + 0.950101 w_k1_INTC_s112
                                 + 0.593507 w_k1_KO_s112
                                 + 0.380802 w_k1_MAT_s112
                                 + 0.82121 w_k1_MSFT_s112
                                 + 0.904119 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.833478 w_k1_ADM_s112
                                 + 0.990689 w_k1_AXP_s112
                                 + 1.01462 w_k1_DIS_s112
                                 + 0.973927 w_k1_FDX_s112
                                 + 1.22948 w_k1_IBM_s112
                                 + 1.10008 w_k1_INTC_s112
                                 + 0.99937 w_k1_KO_s112 + 1.54947 w_k1_MAT_s112
                                 + 0.951971 w_k1_MSFT_s112
                                 + 1.07708 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.07038 w_k1_ADM_s112
                                 + 1.38508 w_k1_AXP_s112
                                 + 1.51455 w_k1_DIS_s112 + 1.1689 w_k1_FDX_s112
                                 + 1.24261 w_k1_IBM_s112
                                 + 1.18299 w_k1_INTC_s112
                                 + 1.14949 w_k1_KO_s112 + 1.53199 w_k1_MAT_s112
                                 + 1.0885 w_k1_MSFT_s112 + 1.24977 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.940124 w_k1_ADM_s113
                                 + 1.1253 w_k1_AXP_s113 + 1.39025 w_k1_DIS_s113
                                 + 1.14549 w_k1_FDX_s113
                                 + 1.29682 w_k1_IBM_s113
                                 + 1.22361 w_k1_INTC_s113 + 1.3135 w_k1_KO_s113
                                 + 0.808142 w_k1_MAT_s113
                                 + 1.24098 w_k1_MSFT_s113
                                 + 1.06625 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.84516 w_k1_ADM_s113
                                 + 1.35302 w_k1_AXP_s113
                                 + 1.28139 w_k1_DIS_s113
                                 + 1.09012 w_k1_FDX_s113
                                 + 1.69552 w_k1_IBM_s113
                                 + 1.28607 w_k1_INTC_s113
                                 + 1.57655 w_k1_KO_s113 + 1.98586 w_k1_MAT_s113
                                 + 1.21621 w_k1_MSFT_s113
                                 + 1.18307 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.841941 w_k1_ADM_s113
                                 + 0.954502 w_k1_AXP_s113
                                 + 0.898062 w_k1_DIS_s113
                                 + 1.01805 w_k1_FDX_s113
                                 + 0.740111 w_k1_IBM_s113
                                 + 1.0313 w_k1_INTC_s113
                                 + 0.791261 w_k1_KO_s113
                                 + 0.671806 w_k1_MAT_s113
                                 + 0.960014 w_k1_MSFT_s113
                                 + 0.906437 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.03162 w_k1_ADM_s113
                                 + 1.23876 w_k1_AXP_s113
                                 + 1.40968 w_k1_DIS_s113
                                 + 1.28494 w_k1_FDX_s113
                                 + 0.973494 w_k1_IBM_s113
                                 + 1.30135 w_k1_INTC_s113
                                 + 1.13825 w_k1_KO_s113 + 0.80505 w_k1_MAT_s113
                                 + 1.26327 w_k1_MSFT_s113
                                 + 1.05575 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.678541 w_k1_ADM_s113
                                 + 0.82029 w_k1_AXP_s113
                                 + 0.85111 w_k1_DIS_s113
                                 + 0.875063 w_k1_FDX_s113
                                 + 0.946401 w_k1_IBM_s113
                                 + 1.01457 w_k1_INTC_s113
                                 + 1.25906 w_k1_KO_s113 + 0.82335 w_k1_MAT_s113
                                 + 1.16308 w_k1_MSFT_s113
                                 + 0.982635 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.946264 w_k1_ADM_s113
                                 + 1.07883 w_k1_AXP_s113
                                 + 0.906505 w_k1_DIS_s113
                                 + 0.931885 w_k1_FDX_s113
                                 + 1.32016 w_k1_IBM_s113
                                 + 1.0362 w_k1_INTC_s113 + 1.15526 w_k1_KO_s113
                                 + 1.03452 w_k1_MAT_s113
                                 + 1.07146 w_k1_MSFT_s113
                                 + 1.09533 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.925775 w_k1_ADM_s113
                                 + 0.63217 w_k1_AXP_s113
                                 + 0.650639 w_k1_DIS_s113
                                 + 0.900213 w_k1_FDX_s113
                                 + 0.465854 w_k1_IBM_s113
                                 + 0.741424 w_k1_INTC_s113
                                 + 0.684212 w_k1_KO_s113
                                 + 0.350996 w_k1_MAT_s113
                                 + 0.855679 w_k1_MSFT_s113
                                 + 0.771216 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.697303 w_k1_ADM_s113
                                 + 0.76 w_k1_AXP_s113 + 0.576612 w_k1_DIS_s113
                                 + 0.726182 w_k1_FDX_s113
                                 + 0.759793 w_k1_IBM_s113
                                 + 0.867861 w_k1_INTC_s113
                                 + 0.677195 w_k1_KO_s113
                                 + 0.544185 w_k1_MAT_s113
                                 + 0.837679 w_k1_MSFT_s113
                                 + 0.798519 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.27614 w_k1_ADM_s114
                                 + 0.917589 w_k1_AXP_s114
                                 + 1.04994 w_k1_DIS_s114
                                 + 0.92003 w_k1_FDX_s114
                                 + 1.01644 w_k1_IBM_s114
                                 + 0.885283 w_k1_INTC_s114
                                 + 1.03353 w_k1_KO_s114
                                 + 0.903966 w_k1_MAT_s114
                                 + 0.864677 w_k1_MSFT_s114
                                 + 1.08421 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.871454 w_k1_ADM_s114
                                 + 0.601166 w_k1_AXP_s114
                                 + 0.7054 w_k1_DIS_s114 + 0.79073 w_k1_FDX_s114
                                 + 0.540159 w_k1_IBM_s114
                                 + 0.716322 w_k1_INTC_s114
                                 + 0.648191 w_k1_KO_s114
                                 + 0.377368 w_k1_MAT_s114
                                 + 0.850195 w_k1_MSFT_s114
                                 + 0.850858 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.17868 w_k1_ADM_s114
                                 + 0.902997 w_k1_AXP_s114
                                 + 0.922131 w_k1_DIS_s114
                                 + 1.03778 w_k1_FDX_s114
                                 + 0.723948 w_k1_IBM_s114
                                 + 0.889979 w_k1_INTC_s114
                                 + 0.765973 w_k1_KO_s114
                                 + 0.688442 w_k1_MAT_s114
                                 + 0.924911 w_k1_MSFT_s114
                                 + 1.00837 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.28828 w_k1_ADM_s114
                                 + 1.50585 w_k1_AXP_s114 + 1.5293 w_k1_DIS_s114
                                 + 1.19669 w_k1_FDX_s114
                                 + 2.03302 w_k1_IBM_s114
                                 + 1.00067 w_k1_INTC_s114
                                 + 1.40191 w_k1_KO_s114 + 1.91204 w_k1_MAT_s114
                                 + 1.07723 w_k1_MSFT_s114
                                 + 1.29909 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.32427 w_k1_ADM_s114
                                 + 1.12631 w_k1_AXP_s114
                                 + 1.39145 w_k1_DIS_s114
                                 + 1.10503 w_k1_FDX_s114
                                 + 1.45097 w_k1_IBM_s114
                                 + 1.03958 w_k1_INTC_s114
                                 + 1.30971 w_k1_KO_s114 + 1.19524 w_k1_MAT_s114
                                 + 1.05583 w_k1_MSFT_s114
                                 + 1.28632 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.935486 w_k1_ADM_s114
                                 + 0.560887 w_k1_AXP_s114
                                 + 0.604464 w_k1_DIS_s114
                                 + 0.673728 w_k1_FDX_s114
                                 + 0.808367 w_k1_IBM_s114
                                 + 0.717681 w_k1_INTC_s114
                                 + 0.752137 w_k1_KO_s114
                                 + 0.500245 w_k1_MAT_s114
                                 + 0.678228 w_k1_MSFT_s114
                                 + 0.897741 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.848604 w_k1_ADM_s114
                                 + 0.685436 w_k1_AXP_s114
                                 + 0.934581 w_k1_DIS_s114
                                 + 0.805979 w_k1_FDX_s114
                                 + 1.1493 w_k1_IBM_s114
                                 + 0.997115 w_k1_INTC_s114
                                 + 1.09353 w_k1_KO_s114
                                 + 0.999202 w_k1_MAT_s114
                                 + 0.942007 w_k1_MSFT_s114
                                 + 1.00074 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.51266 w_k1_ADM_s114
                                 + 1.29034 w_k1_AXP_s114
                                 + 1.32912 w_k1_DIS_s114
                                 + 1.38269 w_k1_FDX_s114
                                 + 0.929857 w_k1_IBM_s114
                                 + 1.07133 w_k1_INTC_s114
                                 + 1.12646 w_k1_KO_s114
                                 + 0.904018 w_k1_MAT_s114
                                 + 1.20226 w_k1_MSFT_s114
                                 + 1.23297 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.858419 w_k1_ADM_s115
                                 + 0.711436 w_k1_AXP_s115
                                 + 0.565837 w_k1_DIS_s115
                                 + 0.793505 w_k1_FDX_s115
                                 + 0.649018 w_k1_IBM_s115
                                 + 0.787594 w_k1_INTC_s115
                                 + 0.658177 w_k1_KO_s115
                                 + 0.587859 w_k1_MAT_s115
                                 + 0.576409 w_k1_MSFT_s115
                                 + 0.767436 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.04806 w_k1_ADM_s115
                                 + 0.849369 w_k1_AXP_s115
                                 + 0.864034 w_k1_DIS_s115
                                 + 1.07979 w_k1_FDX_s115
                                 + 0.719636 w_k1_IBM_s115
                                 + 1.20035 w_k1_INTC_s115
                                 + 0.823767 w_k1_KO_s115
                                 + 0.636173 w_k1_MAT_s115
                                 + 1.02646 w_k1_MSFT_s115
                                 + 0.923783 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.00209 w_k1_ADM_s115
                                 + 0.798382 w_k1_AXP_s115
                                 + 1.45783 w_k1_DIS_s115
                                 + 1.27102 w_k1_FDX_s115
                                 + 0.968721 w_k1_IBM_s115
                                 + 1.50365 w_k1_INTC_s115
                                 + 1.37004 w_k1_KO_s115
                                 + 0.835362 w_k1_MAT_s115
                                 + 1.26807 w_k1_MSFT_s115
                                 + 1.11875 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.01661 w_k1_ADM_s115
                                 + 0.960732 w_k1_AXP_s115
                                 + 1.12478 w_k1_DIS_s115
                                 + 1.02323 w_k1_FDX_s115
                                 + 1.37441 w_k1_IBM_s115
                                 + 1.26525 w_k1_INTC_s115
                                 + 1.41179 w_k1_KO_s115 + 1.1363 w_k1_MAT_s115
                                 + 1.18182 w_k1_MSFT_s115
                                 + 0.954072 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04248 w_k1_ADM_s115
                                 + 0.931536 w_k1_AXP_s115
                                 + 0.936676 w_k1_DIS_s115
                                 + 0.967186 w_k1_FDX_s115
                                 + 0.978179 w_k1_IBM_s115
                                 + 0.898197 w_k1_INTC_s115
                                 + 1.05293 w_k1_KO_s115 + 1.19105 w_k1_MAT_s115
                                 + 0.95175 w_k1_MSFT_s115
                                 + 1.01926 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.356 w_k1_ADM_s115
                                 + 1.29434 w_k1_AXP_s115
                                 + 1.37782 w_k1_DIS_s115
                                 + 1.34845 w_k1_FDX_s115
                                 + 1.37034 w_k1_IBM_s115
                                 + 1.36082 w_k1_INTC_s115
                                 + 1.40593 w_k1_KO_s115 + 1.55256 w_k1_MAT_s115
                                 + 1.23121 w_k1_MSFT_s115
                                 + 1.33013 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.163 w_k1_ADM_s115
                                 + 1.14308 w_k1_AXP_s115
                                 + 1.08399 w_k1_DIS_s115
                                 + 1.16794 w_k1_FDX_s115
                                 + 0.860979 w_k1_IBM_s115
                                 + 1.30637 w_k1_INTC_s115
                                 + 1.00267 w_k1_KO_s115
                                 + 0.863046 w_k1_MAT_s115
                                 + 1.18833 w_k1_MSFT_s115
                                 + 1.05759 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.649913 w_k1_ADM_s115
                                 + 0.732094 w_k1_AXP_s115
                                 + 0.771003 w_k1_DIS_s115
                                 + 0.796008 w_k1_FDX_s115
                                 + 1.16284 w_k1_IBM_s115
                                 + 0.980437 w_k1_INTC_s115
                                 + 1.06543 w_k1_KO_s115
                                 + 0.900713 w_k1_MAT_s115
                                 + 0.85734 w_k1_MSFT_s115
                                 + 0.880494 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.16033 w_k1_ADM_s116
                                 + 0.798285 w_k1_AXP_s116
                                 + 1.08564 w_k1_DIS_s116
                                 + 1.19835 w_k1_FDX_s116
                                 + 0.746859 w_k1_IBM_s116
                                 + 1.15445 w_k1_INTC_s116
                                 + 0.934645 w_k1_KO_s116
                                 + 0.660196 w_k1_MAT_s116
                                 + 1.40125 w_k1_MSFT_s116
                                 + 0.963666 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.39466 w_k1_ADM_s116
                                 + 1.0732 w_k1_AXP_s116 + 1.17399 w_k1_DIS_s116
                                 + 1.14941 w_k1_FDX_s116
                                 + 0.807325 w_k1_IBM_s116
                                 + 1.05198 w_k1_INTC_s116
                                 + 0.918183 w_k1_KO_s116
                                 + 1.00909 w_k1_MAT_s116
                                 + 1.2098 w_k1_MSFT_s116
                                 + 0.990767 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.35626 w_k1_ADM_s116
                                 + 1.0358 w_k1_AXP_s116 + 1.28884 w_k1_DIS_s116
                                 + 1.06888 w_k1_FDX_s116
                                 + 1.73738 w_k1_IBM_s116
                                 + 1.17834 w_k1_INTC_s116
                                 + 1.30939 w_k1_KO_s116 + 1.62483 w_k1_MAT_s116
                                 + 1.387 w_k1_MSFT_s116 + 1.11716 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.777652 w_k1_ADM_s116
                                 + 0.612657 w_k1_AXP_s116
                                 + 0.62386 w_k1_DIS_s116
                                 + 0.725633 w_k1_FDX_s116
                                 + 0.750664 w_k1_IBM_s116
                                 + 0.777931 w_k1_INTC_s116
                                 + 0.747673 w_k1_KO_s116
                                 + 0.840691 w_k1_MAT_s116
                                 + 0.911952 w_k1_MSFT_s116
                                 + 0.785746 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.16173 w_k1_ADM_s116
                                 + 1.14606 w_k1_AXP_s116
                                 + 1.80584 w_k1_DIS_s116
                                 + 1.42563 w_k1_FDX_s116
                                 + 1.23437 w_k1_IBM_s116
                                 + 1.07807 w_k1_INTC_s116
                                 + 1.27736 w_k1_KO_s116 + 1.65687 w_k1_MAT_s116
                                 + 1.60219 w_k1_MSFT_s116
                                 + 1.08439 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.942108 w_k1_ADM_s116
                                 + 0.720744 w_k1_AXP_s116
                                 + 0.685891 w_k1_DIS_s116
                                 + 0.906033 w_k1_FDX_s116
                                 + 0.845529 w_k1_IBM_s116
                                 + 1.02939 w_k1_INTC_s116
                                 + 0.779718 w_k1_KO_s116
                                 + 0.956722 w_k1_MAT_s116
                                 + 1.11446 w_k1_MSFT_s116
                                 + 0.799455 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.923411 w_k1_ADM_s116
                                 + 0.896178 w_k1_AXP_s116
                                 + 0.983355 w_k1_DIS_s116
                                 + 0.930819 w_k1_FDX_s116
                                 + 0.915883 w_k1_IBM_s116
                                 + 1.06207 w_k1_INTC_s116
                                 + 0.956752 w_k1_KO_s116
                                 + 0.982777 w_k1_MAT_s116
                                 + 1.037 w_k1_MSFT_s116 + 0.836479 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.23208 w_k1_ADM_s116
                                 + 1.11398 w_k1_AXP_s116
                                 + 1.12885 w_k1_DIS_s116
                                 + 1.07607 w_k1_FDX_s116 + 1.0871 w_k1_IBM_s116
                                 + 0.979505 w_k1_INTC_s116
                                 + 0.998933 w_k1_KO_s116
                                 + 1.30214 w_k1_MAT_s116
                                 + 1.21678 w_k1_MSFT_s116
                                 + 0.971665 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.90342 w_k1_ADM_s117
                                 + 1.30717 w_k1_AXP_s117
                                 + 0.796017 w_k1_DIS_s117
                                 + 0.98734 w_k1_FDX_s117
                                 + 1.13135 w_k1_IBM_s117
                                 + 0.907441 w_k1_INTC_s117
                                 + 0.945995 w_k1_KO_s117
                                 + 1.56468 w_k1_MAT_s117
                                 + 0.743951 w_k1_MSFT_s117
                                 + 0.985352 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.973693 w_k1_ADM_s117
                                 + 0.814243 w_k1_AXP_s117
                                 + 0.861549 w_k1_DIS_s117
                                 + 0.958648 w_k1_FDX_s117
                                 + 0.565672 w_k1_IBM_s117
                                 + 0.937441 w_k1_INTC_s117
                                 + 0.801356 w_k1_KO_s117
                                 + 0.713383 w_k1_MAT_s117
                                 + 0.853596 w_k1_MSFT_s117
                                 + 0.9564 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.88517 w_k1_ADM_s117
                                 + 1.05108 w_k1_AXP_s117
                                 + 1.08032 w_k1_DIS_s117
                                 + 1.01816 w_k1_FDX_s117
                                 + 1.12378 w_k1_IBM_s117
                                 + 0.932469 w_k1_INTC_s117
                                 + 1.09956 w_k1_KO_s117 + 1.43789 w_k1_MAT_s117
                                 + 1.03122 w_k1_MSFT_s117
                                 + 1.11596 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.43068 w_k1_ADM_s117
                                 + 1.47305 w_k1_AXP_s117
                                 + 1.17144 w_k1_DIS_s117
                                 + 1.29166 w_k1_FDX_s117
                                 + 0.993643 w_k1_IBM_s117
                                 + 1.04862 w_k1_INTC_s117
                                 + 1.16104 w_k1_KO_s117 + 1.69847 w_k1_MAT_s117
                                 + 0.905073 w_k1_MSFT_s117
                                 + 1.19541 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.824915 w_k1_ADM_s117
                                 + 1.17636 w_k1_AXP_s117
                                 + 0.953959 w_k1_DIS_s117
                                 + 0.877473 w_k1_FDX_s117
                                 + 1.33236 w_k1_IBM_s117
                                 + 0.894622 w_k1_INTC_s117
                                 + 1.04273 w_k1_KO_s117 + 1.03091 w_k1_MAT_s117
                                 + 0.787912 w_k1_MSFT_s117
                                 + 1.12262 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.776532 w_k1_ADM_s117
                                 + 0.73384 w_k1_AXP_s117
                                 + 0.744302 w_k1_DIS_s117
                                 + 0.770752 w_k1_FDX_s117
                                 + 0.944807 w_k1_IBM_s117
                                 + 0.998264 w_k1_INTC_s117
                                 + 0.976645 w_k1_KO_s117
                                 + 0.668248 w_k1_MAT_s117
                                 + 0.790053 w_k1_MSFT_s117
                                 + 0.932975 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.19627 w_k1_ADM_s117
                                 + 1.44307 w_k1_AXP_s117
                                 + 1.10204 w_k1_DIS_s117 + 1.0809 w_k1_FDX_s117
                                 + 0.861528 w_k1_IBM_s117
                                 + 0.871151 w_k1_INTC_s117
                                 + 1.03204 w_k1_KO_s117 + 1.33257 w_k1_MAT_s117
                                 + 0.866652 w_k1_MSFT_s117
                                 + 1.14816 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.22879 w_k1_ADM_s117
                                 + 1.36646 w_k1_AXP_s117
                                 + 1.59878 w_k1_DIS_s117 + 1.2724 w_k1_FDX_s117
                                 + 1.18181 w_k1_IBM_s117
                                 + 1.38181 w_k1_INTC_s117
                                 + 1.23656 w_k1_KO_s117
                                 + 0.789851 w_k1_MAT_s117
                                 + 1.1591 w_k1_MSFT_s117 + 1.2113 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.117631030760332 w_k1_ADM_s000
                                 + 0.0554111297828284 w_k1_ADM_s001
                                 + 0.0456145481181944 w_k1_ADM_s002
                                 + 0.041108561493667 w_k1_ADM_s003
                                 + 0.0932134011129256 w_k1_ADM_s004
                                 + 0.154698857200557 w_k1_ADM_s005
                                 + 0.0356192681714204 w_k1_ADM_s006
                                 + 0.0492396249663291 w_k1_ADM_s007
                                 + 0.0334149284066409 w_k1_ADM_s010
                                 + 0.0323515837229652 w_k1_ADM_s011
                                 + 0.0572062364086025 w_k1_ADM_s012
                                 + 0.024747056639735 w_k1_ADM_s013
                                 + 0.0180454382126604 w_k1_ADM_s014
                                 + 0.0194431774579792 w_k1_ADM_s015
                                 + 0.0268804757770124 w_k1_ADM_s016
                                 + 0.0252946019775218 w_k1_ADM_s017
                                 + 0.00413097738148459 w_k1_ADM_s100
                                 + 0.0142818870865495 w_k1_ADM_s101
                                 + 0.00366134402846758 w_k1_ADM_s102
                                 + 0.00460289167409588 w_k1_ADM_s103
                                 + 0.00810703055576823 w_k1_ADM_s104
                                 + 0.00611703249362827 w_k1_ADM_s105
                                 + 0.00406667786250992 w_k1_ADM_s106
                                 + 0.00820350526320281 w_k1_ADM_s107
                                 + 0.00779254776958103 w_k1_ADM_s110
                                 + 0.0171876097397434 w_k1_ADM_s111
                                 + 0.00772800764984904 w_k1_ADM_s112
                                 + 0.0141874800581861 w_k1_ADM_s113
                                 + 0.0387530702440058 w_k1_ADM_s114
                                 + 0.0238395706859489 w_k1_ADM_s115
                                 + 0.0142844451490328 w_k1_ADM_s116
                                 + 0.00887502818604749 w_k1_ADM_s117
                                 + 0.119617775651635 w_k1_AXP_s000
                                 + 0.0641310836362161 w_k1_AXP_s001
                                 + 0.0408003746376796 w_k1_AXP_s002
                                 + 0.0411708778119118 w_k1_AXP_s003
                                 + 0.103400612226257 w_k1_AXP_s004
                                 + 0.150285329096411 w_k1_AXP_s005
                                 + 0.0364377371729627 w_k1_AXP_s006
                                 + 0.0442925872175575 w_k1_AXP_s007
                                 + 0.0329425508951459 w_k1_AXP_s010
                                 + 0.0331997622776793 w_k1_AXP_s011
                                 + 0.0541976529789573 w_k1_AXP_s012
                                 + 0.0307366404706551 w_k1_AXP_s013
                                 + 0.0159216269590084 w_k1_AXP_s014
                                 + 0.0213188026671705 w_k1_AXP_s015
                                 + 0.0241198374430618 w_k1_AXP_s016
                                 + 0.0205539666025552 w_k1_AXP_s017
                                 + 0.0038944073243783 w_k1_AXP_s100
                                 + 0.0154888554507441 w_k1_AXP_s101
                                 + 0.0034261875897444 w_k1_AXP_s102
                                 + 0.00431433712238418 w_k1_AXP_s103
                                 + 0.00861408294126408 w_k1_AXP_s104
                                 + 0.00580616418154745 w_k1_AXP_s105
                                 + 0.00502635706886761 w_k1_AXP_s106
                                 + 0.00807999085914266 w_k1_AXP_s107
                                 + 0.0086894660247877 w_k1_AXP_s110
                                 + 0.0159624533279646 w_k1_AXP_s111
                                 + 0.00839324251475423 w_k1_AXP_s112
                                 + 0.0166739997424023 w_k1_AXP_s113
                                 + 0.0309296025246885 w_k1_AXP_s114
                                 + 0.0221975761060734 w_k1_AXP_s115
                                 + 0.0118458274018501 w_k1_AXP_s116
                                 + 0.00985271981503744 w_k1_AXP_s117
                                 + 0.121005461837567 w_k1_DIS_s000
                                 + 0.0622734919282099 w_k1_DIS_s001
                                 + 0.0384776646966377 w_k1_DIS_s002
                                 + 0.041397023167999 w_k1_DIS_s003
                                 + 0.104547106697057 w_k1_DIS_s004
                                 + 0.159383695537792 w_k1_DIS_s005
                                 + 0.03594672646727 w_k1_DIS_s006
                                 + 0.045966756896033 w_k1_DIS_s007
                                 + 0.0361280293916742 w_k1_DIS_s010
                                 + 0.0319248835645169 w_k1_DIS_s011
                                 + 0.0586834221772322 w_k1_DIS_s012
                                 + 0.0328448426217429 w_k1_DIS_s013
                                 + 0.0197408525105076 w_k1_DIS_s014
                                 + 0.0226969007370918 w_k1_DIS_s015
                                 + 0.0244474692981221 w_k1_DIS_s016
                                 + 0.0249932555611929 w_k1_DIS_s017
                                 + 0.00433728026671376 w_k1_DIS_s100
                                 + 0.0151895582788916 w_k1_DIS_s101
                                 + 0.0032191400831951 w_k1_DIS_s102
                                 + 0.00464112182535307 w_k1_DIS_s103
                                 + 0.00842426369350156 w_k1_DIS_s104
                                 + 0.00591205836667428 w_k1_DIS_s105
                                 + 0.00425814721860079 w_k1_DIS_s106
                                 + 0.00712932356079684 w_k1_DIS_s107
                                 + 0.00968020897393566 w_k1_DIS_s110
                                 + 0.0175377126962261 w_k1_DIS_s111
                                 + 0.0086957154670102 w_k1_DIS_s112
                                 + 0.0171625835616676 w_k1_DIS_s113
                                 + 0.0349578543692784 w_k1_DIS_s114
                                 + 0.0235426489705073 w_k1_DIS_s115
                                 + 0.013618542306842 w_k1_DIS_s116
                                 + 0.00920825791743309 w_k1_DIS_s117
                                 + 0.122109045444035 w_k1_FDX_s000
                                 + 0.0597227626415749 w_k1_FDX_s001
                                 + 0.0411457885441765 w_k1_FDX_s002
                                 + 0.0408426894229392 w_k1_FDX_s003
                                 + 0.0968828894824304 w_k1_FDX_s004
                                 + 0.149889234476484 w_k1_FDX_s005
                                 + 0.0355323173188354 w_k1_FDX_s006
                                 + 0.0473482865241592 w_k1_FDX_s007
                                 + 0.0330751196863948 w_k1_FDX_s010
                                 + 0.0310994700091438 w_k1_FDX_s011
                                 + 0.0572898036032026 w_k1_FDX_s012
                                 + 0.0313361664756031 w_k1_FDX_s013
                                 + 0.0181705353046363 w_k1_FDX_s014
                                 + 0.0213167653355671 w_k1_FDX_s015
                                 + 0.0258580112079651 w_k1_FDX_s016
                                 + 0.0227265295079712 w_k1_FDX_s017
                                 + 0.00420024107955462 w_k1_FDX_s100
                                 + 0.0149569017782983 w_k1_FDX_s101
                                 + 0.00344841767805626 w_k1_FDX_s102
                                 + 0.00452855411371687 w_k1_FDX_s103
                                 + 0.00835146743994322 w_k1_FDX_s104
                                 + 0.00594236084695006 w_k1_FDX_s105
                                 + 0.00427054826686673 w_k1_FDX_s106
                                 + 0.00714048974420075 w_k1_FDX_s107
                                 + 0.00882773362390171 w_k1_FDX_s110
                                 + 0.0164787017922573 w_k1_FDX_s111
                                 + 0.00799022507706667 w_k1_FDX_s112
                                 + 0.0169269211811559 w_k1_FDX_s113
                                 + 0.0326019053549353 w_k1_FDX_s114
                                 + 0.0247366042181099 w_k1_FDX_s115
                                 + 0.0134738841842822 w_k1_FDX_s116
                                 + 0.00907584653318175 w_k1_FDX_s117
                                 + 0.118162624605969 w_k1_IBM_s000
                                 + 0.0603418256515156 w_k1_IBM_s001
                                 + 0.0401395081879832 w_k1_IBM_s002
                                 + 0.0410374419832102 w_k1_IBM_s003
                                 + 0.103077071013221 w_k1_IBM_s004
                                 + 0.151319555433635 w_k1_IBM_s005
                                 + 0.0354947216077624 w_k1_IBM_s006
                                 + 0.0459889024889313 w_k1_IBM_s007
                                 + 0.0345850949373933 w_k1_IBM_s010
                                 + 0.0320523134842085 w_k1_IBM_s011
                                 + 0.0581638093380793 w_k1_IBM_s012
                                 + 0.0321324255315368 w_k1_IBM_s013
                                 + 0.0183788117013551 w_k1_IBM_s014
                                 + 0.0212886753453692 w_k1_IBM_s015
                                 + 0.025091053315449 w_k1_IBM_s016
                                 + 0.0223277010555738 w_k1_IBM_s017
                                 + 0.00420938701501894 w_k1_IBM_s100
                                 + 0.0148879755639293 w_k1_IBM_s101
                                 + 0.00341747329115859 w_k1_IBM_s102
                                 + 0.00444450149114681 w_k1_IBM_s103
                                 + 0.00875393207804717 w_k1_IBM_s104
                                 + 0.00588450306798084 w_k1_IBM_s105
                                 + 0.00441376890000959 w_k1_IBM_s106
                                 + 0.00747114907160058 w_k1_IBM_s107
                                 + 0.00903731950570662 w_k1_IBM_s110
                                 + 0.0167547333960312 w_k1_IBM_s111
                                 + 0.0083078925917001 w_k1_IBM_s112
                                 + 0.0170029578213727 w_k1_IBM_s113
                                 + 0.0334634517875026 w_k1_IBM_s114
                                 + 0.024188723036627 w_k1_IBM_s115
                                 + 0.0132485970538793 w_k1_IBM_s116
                                 + 0.0092500734505192 w_k1_IBM_s117
                                 + 0.126272678501295 w_k1_INTC_s000
                                 + 0.0574620905192859 w_k1_INTC_s001
                                 + 0.0372378122247892 w_k1_INTC_s002
                                 + 0.0393925414623933 w_k1_INTC_s003
                                 + 0.094593070345831 w_k1_INTC_s004
                                 + 0.159932918558528 w_k1_INTC_s005
                                 + 0.0343149202383338 w_k1_INTC_s006
                                 + 0.0479776883541932 w_k1_INTC_s007
                                 + 0.034404202918477 w_k1_INTC_s010
                                 + 0.031588970911663 w_k1_INTC_s011
                                 + 0.0541923340687153 w_k1_INTC_s012
                                 + 0.0348815018497806 w_k1_INTC_s013
                                 + 0.0168407590934421 w_k1_INTC_s014
                                 + 0.023800050884131 w_k1_INTC_s015
                                 + 0.0264056236184273 w_k1_INTC_s016
                                 + 0.0232445615839107 w_k1_INTC_s017
                                 + 0.00423495216364656 w_k1_INTC_s100
                                 + 0.0144379626012852 w_k1_INTC_s101
                                 + 0.00393843188630968 w_k1_INTC_s102
                                 + 0.00481535073706258 w_k1_INTC_s103
                                 + 0.00774110826007735 w_k1_INTC_s104
                                 + 0.00541670752478982 w_k1_INTC_s105
                                 + 0.00514630430862013 w_k1_INTC_s106
                                 + 0.00786128961210338 w_k1_INTC_s107
                                 + 0.00748506069901601 w_k1_INTC_s110
                                 + 0.0159864666963016 w_k1_INTC_s111
                                 + 0.00812761509322315 w_k1_INTC_s112
                                 + 0.0181957780023623 w_k1_INTC_s113
                                 + 0.0297957386188274 w_k1_INTC_s114
                                 + 0.0273413307182325 w_k1_INTC_s115
                                 + 0.0135332090508419 w_k1_INTC_s116
                                 + 0.00917113890431958 w_k1_INTC_s117
                                 + 0.107643288182121 w_k1_KO_s000
                                 + 0.0561724974321298 w_k1_KO_s001
                                 + 0.0376749204770592 w_k1_KO_s002
                                 + 0.0378362040385895 w_k1_KO_s003
                                 + 0.107240436219857 w_k1_KO_s004
                                 + 0.149646168819585 w_k1_KO_s005
                                 + 0.0356921971457743 w_k1_KO_s006
                                 + 0.0543157397298323 w_k1_KO_s007
                                 + 0.0314807286842402 w_k1_KO_s010
                                 + 0.032533892227384 w_k1_KO_s011
                                 + 0.0552703340868915 w_k1_KO_s012
                                 + 0.0342734544645429 w_k1_KO_s013
                                 + 0.0195411102122357 w_k1_KO_s014
                                 + 0.0234806659005481 w_k1_KO_s015
                                 + 0.0255028279839395 w_k1_KO_s016
                                 + 0.0251150043903456 w_k1_KO_s017
                                 + 0.00423785162454483 w_k1_KO_s100
                                 + 0.0148559699263877 w_k1_KO_s101
                                 + 0.00328408688759116 w_k1_KO_s102
                                 + 0.00419257218032533 w_k1_KO_s103
                                 + 0.00849604482477127 w_k1_KO_s104
                                 + 0.00580637220391571 w_k1_KO_s105
                                 + 0.00444656421831329 w_k1_KO_s106
                                 + 0.00786301507517823 w_k1_KO_s107
                                 + 0.00838089988309206 w_k1_KO_s110
                                 + 0.0167327151793408 w_k1_KO_s111
                                 + 0.00742390273861645 w_k1_KO_s112
                                 + 0.0185979372693562 w_k1_KO_s113
                                 + 0.032954635049743 w_k1_KO_s114
                                 + 0.0255767988018395 w_k1_KO_s115
                                 + 0.0127178530879448 w_k1_KO_s116
                                 + 0.00942139283764255 w_k1_KO_s117
                                 + 0.105131245987591 w_k1_MAT_s000
                                 + 0.0571157112466533 w_k1_MAT_s001
                                 + 0.0389237693656198 w_k1_MAT_s002
                                 + 0.0403706319608593 w_k1_MAT_s003
                                 + 0.0844747514875034 w_k1_MAT_s004
                                 + 0.174940242347964 w_k1_MAT_s005
                                 + 0.0356681777873717 w_k1_MAT_s006
                                 + 0.0492981002441613 w_k1_MAT_s007
                                 + 0.0350057197548509 w_k1_MAT_s010
                                 + 0.0346632751930715 w_k1_MAT_s011
                                 + 0.0550812333298468 w_k1_MAT_s012
                                 + 0.0356328876274519 w_k1_MAT_s013
                                 + 0.0212978314211697 w_k1_MAT_s014
                                 + 0.0242097723853025 w_k1_MAT_s015
                                 + 0.0220875566981782 w_k1_MAT_s016
                                 + 0.0228550591150113 w_k1_MAT_s017
                                 + 0.00356079542214767 w_k1_MAT_s100
                                 + 0.0176826077988494 w_k1_MAT_s101
                                 + 0.00254112326236301 w_k1_MAT_s102
                                 + 0.00562433331942442 w_k1_MAT_s103
                                 + 0.00962149769663113 w_k1_MAT_s104
                                 + 0.00563939484124539 w_k1_MAT_s105
                                 + 0.00555592658369488 w_k1_MAT_s106
                                 + 0.00658246780311294 w_k1_MAT_s107
                                 + 0.00961063309409682 w_k1_MAT_s110
                                 + 0.0156151236586369 w_k1_MAT_s111
                                 + 0.0103243926779102 w_k1_MAT_s112
                                 + 0.0142715931026706 w_k1_MAT_s113
                                 + 0.0289413994066586 w_k1_MAT_s114
                                 + 0.0228337701190801 w_k1_MAT_s115
                                 + 0.0146492211714479 w_k1_MAT_s116
                                 + 0.00967411326299468 w_k1_MAT_s117
                                 + 0.119948035743958 w_k1_MSFT_s000
                                 + 0.0569555287740653 w_k1_MSFT_s001
                                 + 0.0368981660786839 w_k1_MSFT_s002
                                 + 0.0423723031863201 w_k1_MSFT_s003
                                 + 0.0975081045832445 w_k1_MSFT_s004
                                 + 0.157661406275572 w_k1_MSFT_s005
                                 + 0.0345742442264056 w_k1_MSFT_s006
                                 + 0.0471760064820141 w_k1_MSFT_s007
                                 + 0.0326824279904835 w_k1_MSFT_s010
                                 + 0.0306931532185116 w_k1_MSFT_s011
                                 + 0.0569071465871326 w_k1_MSFT_s012
                                 + 0.0344107931217519 w_k1_MSFT_s013
                                 + 0.0191054744973199 w_k1_MSFT_s014
                                 + 0.0225329370045731 w_k1_MSFT_s015
                                 + 0.0256564874933391 w_k1_MSFT_s016
                                 + 0.0233603467756418 w_k1_MSFT_s017
                                 + 0.00453195569636506 w_k1_MSFT_s100
                                 + 0.0157522037045668 w_k1_MSFT_s101
                                 + 0.00357013684269948 w_k1_MSFT_s102
                                 + 0.00480987648147376 w_k1_MSFT_s103
                                 + 0.00758508995238906 w_k1_MSFT_s104
                                 + 0.00538480810975045 w_k1_MSFT_s105
                                 + 0.00431717593995682 w_k1_MSFT_s106
                                 + 0.00787074356358335 w_k1_MSFT_s107
                                 + 0.00824648108105385 w_k1_MSFT_s110
                                 + 0.0167981507338833 w_k1_MSFT_s111
                                 + 0.00747699733621653 w_k1_MSFT_s112
                                 + 0.0185889397368374 w_k1_MSFT_s113
                                 + 0.0310667059364311 w_k1_MSFT_s114
                                 + 0.0242996310661026 w_k1_MSFT_s115
                                 + 0.0158283443650122 w_k1_MSFT_s116
                                 + 0.00825336820418439 w_k1_MSFT_s117
                                 + 0.116987171133464 w_k1_PG_s000
                                 + 0.0564852528408415 w_k1_PG_s001
                                 + 0.0393967651814906 w_k1_PG_s002
                                 + 0.0427071996421324 w_k1_PG_s003
                                 + 0.0894196036226338 w_k1_PG_s004
                                 + 0.146860742900632 w_k1_PG_s005
                                 + 0.0351508489212812 w_k1_PG_s006
                                 + 0.0462716083917795 w_k1_PG_s007
                                 + 0.0338821849216963 w_k1_PG_s010
                                 + 0.0329504365989086 w_k1_PG_s011
                                 + 0.0604654077718057 w_k1_PG_s012
                                 + 0.0318948131371663 w_k1_PG_s013
                                 + 0.0180871439207444 w_k1_PG_s014
                                 + 0.0212059499546505 w_k1_PG_s015
                                 + 0.0240817086596207 w_k1_PG_s016
                                 + 0.0214080788273576 w_k1_PG_s017
                                 + 0.0043183544269254 w_k1_PG_s100
                                 + 0.0147920107244138 w_k1_PG_s101
                                 + 0.00330831499596551 w_k1_PG_s102
                                 + 0.00402652235742416 w_k1_PG_s103
                                 + 0.00817268510227178 w_k1_PG_s104
                                 + 0.00638951088685638 w_k1_PG_s105
                                 + 0.00420705626014177 w_k1_PG_s106
                                 + 0.00795492946800986 w_k1_PG_s107
                                 + 0.0080406699490505 w_k1_PG_s110
                                 + 0.016650605678328 w_k1_PG_s111
                                 + 0.00878653099385485 w_k1_PG_s112
                                 + 0.0166235645436074 w_k1_PG_s113
                                 + 0.0354310601548096 w_k1_PG_s114
                                 + 0.023745931502282 w_k1_PG_s115
                                 + 0.0121373794482026 w_k1_PG_s116
                                 + 0.00979423374915966 w_k1_PG_s117 >= 1.209
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
