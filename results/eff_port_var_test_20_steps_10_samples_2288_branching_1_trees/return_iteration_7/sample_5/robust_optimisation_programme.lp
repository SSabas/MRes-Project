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
 w_k1_ADM_s0_return_constraint:  1.09038 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.943498 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.352 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.04649 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.927717 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.07187 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.01749 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.22238 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.11398 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.81244 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.04119 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.761186 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.03675 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.920697 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.97036 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.35064 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.02937 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.739601 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.874545 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.07172 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.894191 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.958216 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.14869 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.921144 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.708944 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.32712 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.57832 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.851494 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.893024 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.951266 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.755848 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.20221 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.985591 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.44287 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.950198 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.07004 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.916329 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.1296 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.2535 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.858024 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.34865 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.891707 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.977312 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.2991 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.12752 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.59915 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.19495 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.840752 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.26677 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.759687 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.09383 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.745515 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.33096 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.59303 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.43063 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.698188 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.874682 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.24253 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.961129 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.14874 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.14069 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.842701 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.527527 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.29166 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.89869 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.825629 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.01993 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.910691 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.752158 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.48109 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.14455 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.52049 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.05091 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.26771 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.11617 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.959774 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.27098 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.849765 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.50759 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.910774 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.963316 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.33585 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.955616 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.69779 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.07786 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.04808 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.03502 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.700139 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.49636 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.779796 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.13039 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.91708 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.64825 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.670706 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.971919 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.03493 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.0654 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.16965 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.08565 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.860859 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.495469 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.48411 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.97881 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.719777 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.899465 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.88006 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.710596 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.46896 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.30806 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.91993 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.1877 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.21613 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.935521 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.973785 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16353 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.897646 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.25503 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.956154 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.977855 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.14367 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.961694 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.53928 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.25097 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.956487 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.09673 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.738511 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.14182 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.9134 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.0033 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.53692 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.24932 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.754983 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.00363 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.11031 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.865922 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.920142 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.21301 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.988219 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.625092 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.27569 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.67532 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.80295 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.913667 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.860819 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.765757 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.37531 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.00771 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.4167 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.25453 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.031 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.940099 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.993839 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.14663 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.935291 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.383 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.926665 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.899677 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.59652 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.06826 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.24217 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.7972 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.30596 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.07639 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.18477 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.994382 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.592912 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.43549 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.21911 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.42922 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.735215 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.956428 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.779609 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.30951 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.92459 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.846399 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.613668 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.760109 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.43881 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.14012 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.823911 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.48936 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.14444 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.818722 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.895422 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.26892 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.18814 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.694887 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.7211 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.1848 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.869138 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.13161 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.92124 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.0812 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.89742 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.10272 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.22249 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.05132 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.38275 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.07971 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.11964 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.27933 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.867666 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.11826 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.11755 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.07788 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.32493 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.37199 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.915906 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.11337 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.04016 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.92823 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.23754 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.04499 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.872905 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.614598 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.02072 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.23913 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.757336 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.810913 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.839255 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.880614 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.19183 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.08579 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.15247 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.14029 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.985861 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.895596 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.81428 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.15238 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.922984 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.32896 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.931429 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.999486 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.42492 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.0235 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.51307 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.974122 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.28235 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.12566 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.03558 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.30222 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.831906 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.24566 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.34478 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.39342 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.769527 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.874494 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.9368 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.14612 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.52525 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.946719 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.630599 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.587911 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.27563 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.33036 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.742562 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.05143 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.973092 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.79517 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.06518 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.1142 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.31624 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.879444 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.24657 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.896156 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.898941 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.26321 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.80274 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.39962 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.811774 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.937124 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.82601 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.30531 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.41169 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.753533 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.19066 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.35145 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.970763 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.937384 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.499658 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.76228 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.18961 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.89601 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.9334 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.936838 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.04175 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.30546 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.95777 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.929033 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.605635 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.637472 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.40652 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.3997 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.880405 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.78359 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.972593 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.924036 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.838089 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.04675 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.5509 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.807252 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 2.06122 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.36481 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.822076 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1838 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.866612 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.13186 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.903544 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.05419 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.2935 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.00564 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.47459 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.08722 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.974863 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.2349 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.813606 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.29261 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.959286 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.08512 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.49314 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.31743 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.757966 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.05822 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.08977 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.961862 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.03942 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.987631 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.928499 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.607213 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.22255 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.45017 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.858349 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.920706 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.871953 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.767988 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.07928 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.03196 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.33213 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.13164 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.03004 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.956828 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.897922 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.1006 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.932658 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.25216 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.901009 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.981938 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.23144 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.951803 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.20818 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.958766 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.9891 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.10154 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.865694 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.98235 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.878365 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.2703 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.371 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.27003 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.859582 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.02513 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.00306 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.10304 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.27811 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.06212 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.945086 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.701718 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.17721 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.3554 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.851475 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.05313 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.944083 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.829526 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.0358 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.01677 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.2182 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.05585 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.0773 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.980338 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.92385 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.314000339675644 z_k1_s0000
                                 - 0.0900002732159061 z_k1_s0001
                                 - 0.101000380773656 z_k1_s0002
                                 - 0.353001024417783 z_k1_s0003
                                 - 0.167001026120157 z_k1_s0004
                                 - 0.201000144967285 z_k1_s0005
                                 - 0.136000419601836 z_k1_s0006
                                 - 0.154999998581355 z_k1_s0007
                                 - 0.180999965840095 z_k1_s0010
                                 - 0.154999572206562 z_k1_s0011
                                 - 0.175999225835239 z_k1_s0012
                                 - 0.0709998400945322 z_k1_s0013
                                 - 0.188999930650787 z_k1_s0014
                                 - 0.0529999192704736 z_k1_s0015
                                 - 0.0969997257292835 z_k1_s0016
                                 - 0.0939997897251518 z_k1_s0017
                                 - 0.531999933670946 z_k1_s0020
                                 - 0.233999098705867 z_k1_s0021
                                 - 0.185000588629249 z_k1_s0022
                                 - 0.108999815619991 z_k1_s0023
                                 - 0.156999829015844 z_k1_s0024
                                 - 0.147999826504121 z_k1_s0025
                                 - 0.111999816457231 z_k1_s0026
                                 - 0.133999929996877 z_k1_s0027
                                 - 0.259000727626013 z_k1_s0030
                                 - 0.157000517611185 z_k1_s0031
                                 - 0.133000151678079 z_k1_s0032
                                 - 0.209000319265605 z_k1_s0033
                                 - 0.233000118798344 z_k1_s0034
                                 - 0.0840000063650447 z_k1_s0035
                                 - 0.0849999862122345 z_k1_s0036
                                 - 0.256000221684077 z_k1_s0037
                                 - 0.0609999676148929 z_k1_s0040
                                 - 0.174000298262049 z_k1_s0041
                                 - 0.0959998242136355 z_k1_s0042
                                 - 0.0689999529660292 z_k1_s0043
                                 - 0.0840000224368837 z_k1_s0044
                                 - 0.0559999679579351 z_k1_s0045
                                 - 0.0990003034076814 z_k1_s0046
                                 - 0.0659999672718507 z_k1_s0047
                                 - 0.262998894266105 z_k1_s0050
                                 - 0.069999921540026 z_k1_s0051
                                 - 0.102999735602016 z_k1_s0052
                                 - 0.214999014267113 z_k1_s0053
                                 - 0.325000049470817 z_k1_s0054
                                 - 0.258998766349767 z_k1_s0055
                                 - 0.260000039576653 z_k1_s0056
                                 - 0.15999965516321 z_k1_s0057
                                 - 0.189999609690277 z_k1_s0060
                                 - 0.0939998850467527 z_k1_s0061
                                 - 0.0789999744711927 z_k1_s0062
                                 - 0.223999666834827 z_k1_s0063
                                 - 0.0950002688450448 z_k1_s0064
                                 - 0.061000026100532 z_k1_s0065
                                 - 0.0989999480385873 z_k1_s0066
                                 - 0.0859999698597799 z_k1_s0067
                                 - 0.173999826605661 z_k1_s0070
                                 - 0.171998921923401 z_k1_s0071
                                 - 0.186999596559643 z_k1_s0072
                                 - 0.0859996831609989 z_k1_s0073
                                 - 0.0679997623947025 z_k1_s0074
                                 - 0.159999048812303 z_k1_s0075
                                 - 0.164999644023214 z_k1_s0076
                                 - 0.0989996753142792 z_k1_s0077
                                 - 0.395998403987653 z_k1_s0100
                                 - 0.277999995920247 z_k1_s0101
                                 - 0.646998916044967 z_k1_s0102
                                 - 0.453999950358979 z_k1_s0103
                                 - 0.292999808744065 z_k1_s0104
                                 - 0.274999854135544 z_k1_s0105
                                 - 0.376000445755289 z_k1_s0106
                                 - 0.26799982200447 z_k1_s0107
                                 - 0.138999807677956 z_k1_s0110
                                 - 0.353999044429093 z_k1_s0111
                                 - 0.186999967681688 z_k1_s0112
                                 - 0.620000193382918 z_k1_s0113
                                 - 0.206999959416632 z_k1_s0114
                                 - 0.403999023766452 z_k1_s0115
                                 - 0.134999944210868 z_k1_s0116
                                 - 0.201999961482896 z_k1_s0117
                                 - 0.150999915428151 z_k1_s0120
                                 - 0.229999462109224 z_k1_s0121
                                 - 0.166000010000453 z_k1_s0122
                                 - 0.272000919711319 z_k1_s0123
                                 - 0.390000754880367 z_k1_s0124
                                 - 0.243999947976589 z_k1_s0125
                                 - 0.384000546651496 z_k1_s0126
                                 - 0.293000583512681 z_k1_s0127
                                 - 1.23999778232907 z_k1_s0130
                                 - 1.19700040803699 z_k1_s0131
                                 - 0.500000312768545 z_k1_s0132
                                 - 0.480999788194782 z_k1_s0133
                                 - 0.422000002855771 z_k1_s0134
                                 - 2.01699958352899 z_k1_s0135
                                 - 1.05899643037487 z_k1_s0136
                                 - 1.04500098804836 z_k1_s0137
                                 - 0.450999405939679 z_k1_s0140
                                 - 0.379999681939935 z_k1_s0141
                                 - 1.14699873139185 z_k1_s0142
                                 - 0.703999694405331 z_k1_s0143
                                 - 0.607999636763776 z_k1_s0144
                                 - 1.82999791248225 z_k1_s0145
                                 - 0.583999258203686 z_k1_s0146
                                 - 1.57899680588063 z_k1_s0147
                                 - 0.859000170873472 z_k1_s0150
                                 - 0.718001067302479 z_k1_s0151
                                 - 1.36300526654901 z_k1_s0152
                                 - 0.88500430375379 z_k1_s0153
                                 - 0.813003575800142 z_k1_s0154
                                 - 0.393000954807292 z_k1_s0155
                                 - 0.925999300271517 z_k1_s0156
                                 - 1.00899936543867 z_k1_s0157
                                 - 0.246998832556174 z_k1_s0160
                                 - 0.472999250521816 z_k1_s0161
                                 - 0.472999250521816 z_k1_s0162
                                 - 0.528000848256075 z_k1_s0163
                                 - 0.187999928803485 z_k1_s0164
                                 - 0.152000024649585 z_k1_s0165
                                 - 0.227999795474497 z_k1_s0166
                                 - 0.125999906038529 z_k1_s0167
                                 - 0.128000074333464 z_k1_s0170
                                 - 0.249000427372987 z_k1_s0171
                                 - 0.396001286147565 z_k1_s0172
                                 - 0.132999977986307 z_k1_s0173
                                 - 0.600000423344383 z_k1_s0174
                                 - 0.239999689937623 z_k1_s0175
                                 - 0.331998971830217 z_k1_s0176
                                 - 0.319000037313046 z_k1_s0177
                                 - 1.14400383847331 z_k1_s1000
                                 - 1.34200530648013 z_k1_s1001
                                 - 1.04700206399284 z_k1_s1002
                                 - 1.00500301896395 z_k1_s1003
                                 - 0.468001000419611 z_k1_s1004
                                 - 0.802000669641927 z_k1_s1005
                                 - 0.406001216136177 z_k1_s1006
                                 - 0.751002824251963 z_k1_s1007
                                 - 0.914000761896089 z_k1_s1010
                                 - 0.317999839408215 z_k1_s1011
                                 - 0.971999139172426 z_k1_s1012
                                 - 0.28800012716205 z_k1_s1013
                                 - 0.326999907882004 z_k1_s1014
                                 - 0.627001674341833 z_k1_s1015
                                 - 0.616000558652051 z_k1_s1016
                                 - 1.09800112980506 z_k1_s1017
                                 - 0.679999429953996 z_k1_s1020
                                 - 0.648000765862701 z_k1_s1021
                                 - 0.848000364437288 z_k1_s1022
                                 - 0.690001992984024 z_k1_s1023
                                 - 0.267999938970186 z_k1_s1024
                                 - 0.275000143490407 z_k1_s1025
                                 - 0.284000065816233 z_k1_s1026
                                 - 0.281000091707625 z_k1_s1027
                                 - 1.20600037003719 z_k1_s1030
                                 - 0.745000255104897 z_k1_s1031
                                 - 1.31399987284655 z_k1_s1032
                                 - 2.33900025363852 z_k1_s1033
                                 - 0.981997782247793 z_k1_s1034
                                 - 0.864999801370837 z_k1_s1035
                                 - 0.683999812125613 z_k1_s1036
                                 - 0.748000043894066 z_k1_s1037
                                 - 0.257000392956523 z_k1_s1040
                                 - 0.496999451754683 z_k1_s1041
                                 - 0.598001315393521 z_k1_s1042
                                 - 0.169000329063331 z_k1_s1043
                                 - 0.247999109224873 z_k1_s1044
                                 - 0.186000190819774 z_k1_s1045
                                 - 0.122000278206718 z_k1_s1046
                                 - 0.377001149357202 z_k1_s1047
                                 - 0.458999681550771 z_k1_s1050
                                 - 0.584999399957037 z_k1_s1051
                                 - 0.472999870329049 z_k1_s1052
                                 - 1.60699799618486 z_k1_s1053
                                 - 0.492999291184486 z_k1_s1054
                                 - 0.829996101582773 z_k1_s1055
                                 - 0.988998528507259 z_k1_s1056
                                 - 1.16599799146381 z_k1_s1057
                                 - 0.321000409319919 z_k1_s1060
                                 - 1.21499995822169 z_k1_s1061
                                 - 0.279000717126459 z_k1_s1062
                                 - 0.889999660659702 z_k1_s1063
                                 - 0.838002844816363 z_k1_s1064
                                 - 0.821001444208763 z_k1_s1065
                                 - 0.464000886507897 z_k1_s1066
                                 - 0.943000632547074 z_k1_s1067
                                 - 0.758001363899551 z_k1_s1070
                                 - 0.448001122487165 z_k1_s1071
                                 - 0.529001560648226 z_k1_s1072
                                 - 0.897005121384362 z_k1_s1073
                                 - 0.850004373689524 z_k1_s1074
                                 - 0.895999822567542 z_k1_s1075
                                 - 0.908002848199699 z_k1_s1076
                                 - 0.770004389531708 z_k1_s1077
                                 - 0.3909990594951 z_k1_s1100
                                 - 0.291999070341786 z_k1_s1101
                                 - 0.317000862934663 z_k1_s1102
                                 - 0.371999959242757 z_k1_s1103
                                 - 0.123999896647655 z_k1_s1104
                                 - 0.263999701775714 z_k1_s1105
                                 - 0.330999066068849 z_k1_s1106
                                 - 0.602000831709356 z_k1_s1107
                                 - 0.128000111487067 z_k1_s1110
                                 - 0.107999895661372 z_k1_s1111
                                 - 0.115000107250085 z_k1_s1112
                                 - 0.0609998873199594 z_k1_s1113
                                 - 0.0359999349871861 z_k1_s1114
                                 - 0.166999217199882 z_k1_s1115
                                 - 0.118999969015598 z_k1_s1116
                                 - 0.172999916846283 z_k1_s1117
                                 - 0.0499998949368647 z_k1_s1120
                                 - 0.0629999005963819 z_k1_s1121
                                 - 0.135999615300475 z_k1_s1122
                                 - 0.0879997106650958 z_k1_s1123
                                 - 0.0649998908978678 z_k1_s1124
                                 - 0.180999397083906 z_k1_s1125
                                 - 0.0669998811993536 z_k1_s1126
                                 - 0.0369999579772066 z_k1_s1127
                                 - 0.114999897307745 z_k1_s1130
                                 - 0.203999903180552 z_k1_s1131
                                 - 0.0669998712359319 z_k1_s1132
                                 - 0.0739998420068766 z_k1_s1133
                                 - 0.0830000309262831 z_k1_s1134
                                 - 0.0479999505719392 z_k1_s1135
                                 - 0.0539999255184632 z_k1_s1136
                                 - 0.109999540686274 z_k1_s1137
                                 - 0.0469999049055459 z_k1_s1140
                                 - 0.119999985949958 z_k1_s1141
                                 - 0.0990000090836818 z_k1_s1142
                                 - 0.18099940680045 z_k1_s1143
                                 - 0.0930001338359823 z_k1_s1144
                                 - 0.0699998953209657 z_k1_s1145
                                 - 0.10999991820424 z_k1_s1146
                                 - 0.106999567081063 z_k1_s1147
                                 - 0.0759998543183013 z_k1_s1150
                                 - 0.0729998631359939 z_k1_s1151
                                 - 0.0999997060769104 z_k1_s1152
                                 - 0.11199967080614 z_k1_s1153
                                 - 0.183999459181515 z_k1_s1154
                                 - 0.0599999013459956 z_k1_s1155
                                 - 0.106999685502294 z_k1_s1156
                                 - 0.0649998866498411 z_k1_s1157
                                 - 0.0839998819755747 z_k1_s1160
                                 - 0.0879998412696945 z_k1_s1161
                                 - 0.134000416950739 z_k1_s1162
                                 - 0.234999604026989 z_k1_s1163
                                 - 0.319999199526447 z_k1_s1164
                                 - 0.0839998819755747 z_k1_s1165
                                 - 0.120999843145752 z_k1_s1166
                                 - 0.162000009209735 z_k1_s1167
                                 - 0.110999852018396 z_k1_s1170
                                 - 0.48199949888161 z_k1_s1171
                                 - 0.273000339936763 z_k1_s1172
                                 - 0.172999859073669 z_k1_s1173
                                 - 0.0989998938947721 z_k1_s1174
                                 - 0.217999654162285 z_k1_s1175
                                 - 0.295999477715791 z_k1_s1176
                                 - 0.263000215250494 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.910574 w_k1_ADM_s000
                                 + 0.800295 w_k1_AXP_s000
                                 + 0.930471 w_k1_DIS_s000
                                 + 1.03747 w_k1_FDX_s000
                                 + 0.650139 w_k1_IBM_s000
                                 + 1.10952 w_k1_INTC_s000
                                 + 0.87578 w_k1_KO_s000
                                 + 0.460003 w_k1_MAT_s000
                                 + 1.0426 w_k1_MSFT_s000
                                 + 0.865575 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.847324 w_k1_ADM_s000
                                 + 1.0737 w_k1_AXP_s000 + 1.12741 w_k1_DIS_s000
                                 + 1.0049 w_k1_FDX_s000 + 1.58495 w_k1_IBM_s000
                                 + 0.956344 w_k1_INTC_s000
                                 + 1.20297 w_k1_KO_s000 + 1.58701 w_k1_MAT_s000
                                 + 1.07915 w_k1_MSFT_s000 + 1.1277 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.764387 w_k1_ADM_s000
                                 + 0.881164 w_k1_AXP_s000
                                 + 0.923965 w_k1_DIS_s000
                                 + 0.945413 w_k1_FDX_s000
                                 + 1.29718 w_k1_IBM_s000
                                 + 1.1039 w_k1_INTC_s000 + 1.01652 w_k1_KO_s000
                                 + 1.01838 w_k1_MAT_s000
                                 + 0.915736 w_k1_MSFT_s000
                                 + 0.960509 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.987949 w_k1_ADM_s000
                                 + 0.899066 w_k1_AXP_s000
                                 + 1.01408 w_k1_DIS_s000
                                 + 0.984442 w_k1_FDX_s000
                                 + 1.05323 w_k1_IBM_s000
                                 + 0.918674 w_k1_INTC_s000
                                 + 1.00548 w_k1_KO_s000
                                 + 0.868362 w_k1_MAT_s000
                                 + 0.876149 w_k1_MSFT_s000
                                 + 0.950133 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.08203 w_k1_ADM_s000
                                 + 1.21308 w_k1_AXP_s000
                                 + 0.987539 w_k1_DIS_s000
                                 + 1.18754 w_k1_FDX_s000
                                 + 0.832721 w_k1_IBM_s000
                                 + 1.06373 w_k1_INTC_s000
                                 + 0.874185 w_k1_KO_s000
                                 + 0.707355 w_k1_MAT_s000
                                 + 0.982458 w_k1_MSFT_s000
                                 + 1.12139 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.07699 w_k1_ADM_s000
                                 + 1.12732 w_k1_AXP_s000
                                 + 1.30878 w_k1_DIS_s000
                                 + 1.37227 w_k1_FDX_s000
                                 + 0.991772 w_k1_IBM_s000
                                 + 1.19962 w_k1_INTC_s000
                                 + 1.17245 w_k1_KO_s000
                                 + 0.684104 w_k1_MAT_s000
                                 + 1.22747 w_k1_MSFT_s000
                                 + 1.13316 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.74086 w_k1_ADM_s000
                                 + 0.66415 w_k1_AXP_s000
                                 + 0.556053 w_k1_DIS_s000
                                 + 0.830912 w_k1_FDX_s000
                                 + 0.630958 w_k1_IBM_s000
                                 + 0.880837 w_k1_INTC_s000
                                 + 0.667724 w_k1_KO_s000
                                 + 0.405705 w_k1_MAT_s000
                                 + 0.780554 w_k1_MSFT_s000
                                 + 0.854501 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.3782 w_k1_ADM_s000
                                 + 1.43444 w_k1_AXP_s000
                                 + 1.78955 w_k1_DIS_s000
                                 + 1.45564 w_k1_FDX_s000 + 1.7518 w_k1_IBM_s000
                                 + 1.33371 w_k1_INTC_s000
                                 + 1.40059 w_k1_KO_s000 + 1.15356 w_k1_MAT_s000
                                 + 1.198 w_k1_MSFT_s000 + 1.30935 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.13209 w_k1_ADM_s001
                                 + 1.1698 w_k1_AXP_s001 + 1.23752 w_k1_DIS_s001
                                 + 1.18323 w_k1_FDX_s001
                                 + 0.816464 w_k1_IBM_s001
                                 + 1.43903 w_k1_INTC_s001
                                 + 0.995393 w_k1_KO_s001
                                 + 0.598765 w_k1_MAT_s001
                                 + 1.14665 w_k1_MSFT_s001
                                 + 1.01944 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.00776 w_k1_ADM_s001
                                 + 0.797963 w_k1_AXP_s001
                                 + 0.65827 w_k1_DIS_s001
                                 + 0.832705 w_k1_FDX_s001
                                 + 0.500484 w_k1_IBM_s001
                                 + 0.929973 w_k1_INTC_s001
                                 + 0.679465 w_k1_KO_s001
                                 + 0.626687 w_k1_MAT_s001
                                 + 0.838854 w_k1_MSFT_s001
                                 + 0.859908 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.93807 w_k1_ADM_s001
                                 + 0.829111 w_k1_AXP_s001
                                 + 0.947304 w_k1_DIS_s001
                                 + 0.871968 w_k1_FDX_s001
                                 + 0.83097 w_k1_IBM_s001
                                 + 0.926579 w_k1_INTC_s001
                                 + 0.938365 w_k1_KO_s001
                                 + 0.992645 w_k1_MAT_s001
                                 + 0.915443 w_k1_MSFT_s001
                                 + 0.961329 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.12952 w_k1_ADM_s001
                                 + 1.14708 w_k1_AXP_s001
                                 + 1.14356 w_k1_DIS_s001
                                 + 1.00203 w_k1_FDX_s001
                                 + 1.70481 w_k1_IBM_s001
                                 + 1.44604 w_k1_INTC_s001
                                 + 1.29498 w_k1_KO_s001 + 1.20913 w_k1_MAT_s001
                                 + 1.11892 w_k1_MSFT_s001
                                 + 1.24617 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.21975 w_k1_ADM_s001
                                 + 1.25243 w_k1_AXP_s001
                                 + 1.51909 w_k1_DIS_s001
                                 + 1.14522 w_k1_FDX_s001
                                 + 1.28122 w_k1_IBM_s001
                                 + 1.35908 w_k1_INTC_s001
                                 + 1.28833 w_k1_KO_s001 + 1.15564 w_k1_MAT_s001
                                 + 1.3275 w_k1_MSFT_s001 + 1.32088 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.87702 w_k1_ADM_s001
                                 + 0.73335 w_k1_AXP_s001
                                 + 1.12541 w_k1_DIS_s001
                                 + 1.00552 w_k1_FDX_s001
                                 + 0.557217 w_k1_IBM_s001
                                 + 1.07773 w_k1_INTC_s001
                                 + 0.878927 w_k1_KO_s001
                                 + 0.589054 w_k1_MAT_s001
                                 + 1.08493 w_k1_MSFT_s001
                                 + 0.914513 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.28761 w_k1_ADM_s001
                                 + 0.835742 w_k1_AXP_s001
                                 + 0.982908 w_k1_DIS_s001
                                 + 1.03412 w_k1_FDX_s001
                                 + 0.765314 w_k1_IBM_s001
                                 + 0.916345 w_k1_INTC_s001
                                 + 0.675504 w_k1_KO_s001
                                 + 0.745586 w_k1_MAT_s001
                                 + 0.895937 w_k1_MSFT_s001
                                 + 0.919187 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.701033 w_k1_ADM_s001
                                 + 1.13399 w_k1_AXP_s001
                                 + 0.827879 w_k1_DIS_s001
                                 + 0.757873 w_k1_FDX_s001
                                 + 1.51704 w_k1_IBM_s001
                                 + 1.30125 w_k1_INTC_s001
                                 + 1.06521 w_k1_KO_s001 + 1.41742 w_k1_MAT_s001
                                 + 0.85585 w_k1_MSFT_s001
                                 + 1.11875 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.800175 w_k1_ADM_s002
                                 + 0.943363 w_k1_AXP_s002
                                 + 1.0237 w_k1_DIS_s002 + 1.01605 w_k1_FDX_s002
                                 + 0.777641 w_k1_IBM_s002
                                 + 1.14777 w_k1_INTC_s002
                                 + 0.96955 w_k1_KO_s002 + 1.07205 w_k1_MAT_s002
                                 + 1.05649 w_k1_MSFT_s002
                                 + 0.936623 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.909308 w_k1_ADM_s002
                                 + 0.721609 w_k1_AXP_s002
                                 + 0.75358 w_k1_DIS_s002
                                 + 0.819943 w_k1_FDX_s002
                                 + 0.680691 w_k1_IBM_s002
                                 + 0.876762 w_k1_INTC_s002
                                 + 0.793563 w_k1_KO_s002
                                 + 0.825664 w_k1_MAT_s002
                                 + 0.838419 w_k1_MSFT_s002
                                 + 0.950316 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.20682 w_k1_ADM_s002
                                 + 1.21336 w_k1_AXP_s002
                                 + 1.34669 w_k1_DIS_s002
                                 + 1.16689 w_k1_FDX_s002
                                 + 1.26048 w_k1_IBM_s002
                                 + 1.41043 w_k1_INTC_s002
                                 + 1.23695 w_k1_KO_s002 + 1.58819 w_k1_MAT_s002
                                 + 1.21858 w_k1_MSFT_s002
                                 + 1.09706 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.65195 w_k1_ADM_s002
                                 + 0.536374 w_k1_AXP_s002
                                 + 0.564156 w_k1_DIS_s002
                                 + 0.712889 w_k1_FDX_s002
                                 + 0.561485 w_k1_IBM_s002
                                 + 0.904939 w_k1_INTC_s002
                                 + 0.598353 w_k1_KO_s002
                                 + 0.416722 w_k1_MAT_s002
                                 + 0.783785 w_k1_MSFT_s002
                                 + 0.736692 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.79094 w_k1_ADM_s002
                                 + 0.82471 w_k1_AXP_s002
                                 + 0.805192 w_k1_DIS_s002
                                 + 0.831622 w_k1_FDX_s002
                                 + 0.995452 w_k1_IBM_s002
                                 + 0.91706 w_k1_INTC_s002
                                 + 0.960268 w_k1_KO_s002
                                 + 1.26722 w_k1_MAT_s002
                                 + 1.07038 w_k1_MSFT_s002
                                 + 0.96488 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.946975 w_k1_ADM_s002
                                 + 1.09301 w_k1_AXP_s002
                                 + 1.09212 w_k1_DIS_s002
                                 + 0.949155 w_k1_FDX_s002
                                 + 1.39548 w_k1_IBM_s002
                                 + 1.18131 w_k1_INTC_s002
                                 + 1.20723 w_k1_KO_s002 + 1.53827 w_k1_MAT_s002
                                 + 1.06591 w_k1_MSFT_s002
                                 + 1.21933 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.30114 w_k1_ADM_s002
                                 + 1.52473 w_k1_AXP_s002 + 1.503 w_k1_DIS_s002
                                 + 1.25674 w_k1_FDX_s002
                                 + 1.72024 w_k1_IBM_s002
                                 + 1.40751 w_k1_INTC_s002
                                 + 1.60623 w_k1_KO_s002 + 2.37541 w_k1_MAT_s002
                                 + 1.38687 w_k1_MSFT_s002
                                 + 1.51491 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.08591 w_k1_ADM_s002
                                 + 1.26265 w_k1_AXP_s002
                                 + 1.62317 w_k1_DIS_s002 + 1.2953 w_k1_FDX_s002
                                 + 0.869262 w_k1_IBM_s002
                                 + 1.24634 w_k1_INTC_s002
                                 + 1.15195 w_k1_KO_s002 + 1.33839 w_k1_MAT_s002
                                 + 1.69445 w_k1_MSFT_s002
                                 + 1.14658 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.02483 w_k1_ADM_s003
                                 + 1.18902 w_k1_AXP_s003
                                 + 1.23061 w_k1_DIS_s003
                                 + 1.14737 w_k1_FDX_s003
                                 + 1.23008 w_k1_IBM_s003
                                 + 0.943367 w_k1_INTC_s003
                                 + 1.02601 w_k1_KO_s003 + 1.28618 w_k1_MAT_s003
                                 + 0.996404 w_k1_MSFT_s003
                                 + 1.02465 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.828317 w_k1_ADM_s003
                                 + 0.775532 w_k1_AXP_s003
                                 + 0.726329 w_k1_DIS_s003
                                 + 0.724631 w_k1_FDX_s003
                                 + 0.872758 w_k1_IBM_s003
                                 + 0.790054 w_k1_INTC_s003
                                 + 0.805473 w_k1_KO_s003
                                 + 0.96914 w_k1_MAT_s003
                                 + 0.854444 w_k1_MSFT_s003
                                 + 0.923879 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.14129 w_k1_ADM_s003
                                 + 0.711986 w_k1_AXP_s003
                                 + 0.805714 w_k1_DIS_s003
                                 + 0.879743 w_k1_FDX_s003
                                 + 0.567316 w_k1_IBM_s003
                                 + 0.758706 w_k1_INTC_s003
                                 + 0.690655 w_k1_KO_s003
                                 + 0.733262 w_k1_MAT_s003
                                 + 1.05415 w_k1_MSFT_s003
                                 + 0.903292 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.32852 w_k1_ADM_s003
                                 + 1.23125 w_k1_AXP_s003
                                 + 1.44069 w_k1_DIS_s003
                                 + 1.16818 w_k1_FDX_s003
                                 + 1.00767 w_k1_IBM_s003
                                 + 1.32184 w_k1_INTC_s003
                                 + 1.15966 w_k1_KO_s003
                                 + 0.799331 w_k1_MAT_s003
                                 + 1.34048 w_k1_MSFT_s003
                                 + 1.06115 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.17016 w_k1_ADM_s003
                                 + 1.12267 w_k1_AXP_s003
                                 + 0.95714 w_k1_DIS_s003
                                 + 1.02764 w_k1_FDX_s003
                                 + 0.716542 w_k1_IBM_s003
                                 + 0.933082 w_k1_INTC_s003
                                 + 0.84718 w_k1_KO_s003
                                 + 0.747108 w_k1_MAT_s003
                                 + 1.10532 w_k1_MSFT_s003
                                 + 0.914097 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.00009 w_k1_ADM_s003
                                 + 0.91392 w_k1_AXP_s003
                                 + 0.991934 w_k1_DIS_s003
                                 + 0.842116 w_k1_FDX_s003
                                 + 0.898793 w_k1_IBM_s003
                                 + 0.848285 w_k1_INTC_s003
                                 + 0.851562 w_k1_KO_s003
                                 + 0.874406 w_k1_MAT_s003
                                 + 1.1216 w_k1_MSFT_s003
                                 + 0.968734 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.829395 w_k1_ADM_s003
                                 + 0.731961 w_k1_AXP_s003
                                 + 0.626448 w_k1_DIS_s003
                                 + 0.754893 w_k1_FDX_s003
                                 + 0.526925 w_k1_IBM_s003
                                 + 0.783956 w_k1_INTC_s003
                                 + 0.602029 w_k1_KO_s003
                                 + 0.592545 w_k1_MAT_s003
                                 + 0.766089 w_k1_MSFT_s003
                                 + 0.834964 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.35548 w_k1_ADM_s003
                                 + 1.13058 w_k1_AXP_s003
                                 + 1.11101 w_k1_DIS_s003
                                 + 0.940411 w_k1_FDX_s003
                                 + 1.63888 w_k1_IBM_s003
                                 + 1.03767 w_k1_INTC_s003
                                 + 1.08856 w_k1_KO_s003 + 1.12796 w_k1_MAT_s003
                                 + 1.01519 w_k1_MSFT_s003
                                 + 1.05441 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.934125 w_k1_ADM_s004
                                 + 0.906587 w_k1_AXP_s004
                                 + 0.856311 w_k1_DIS_s004
                                 + 0.924698 w_k1_FDX_s004
                                 + 0.922799 w_k1_IBM_s004
                                 + 0.743725 w_k1_INTC_s004
                                 + 0.894863 w_k1_KO_s004
                                 + 1.11787 w_k1_MAT_s004
                                 + 0.774925 w_k1_MSFT_s004
                                 + 0.907959 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.25202 w_k1_ADM_s004
                                 + 1.37766 w_k1_AXP_s004
                                 + 1.29063 w_k1_DIS_s004
                                 + 1.16751 w_k1_FDX_s004
                                 + 1.10069 w_k1_IBM_s004
                                 + 0.991618 w_k1_INTC_s004
                                 + 1.03791 w_k1_KO_s004 + 1.2916 w_k1_MAT_s004
                                 + 0.9944 w_k1_MSFT_s004 + 1.02477 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.839834 w_k1_ADM_s004
                                 + 0.718325 w_k1_AXP_s004
                                 + 0.705041 w_k1_DIS_s004
                                 + 0.837599 w_k1_FDX_s004
                                 + 0.61586 w_k1_IBM_s004
                                 + 0.645787 w_k1_INTC_s004
                                 + 0.698701 w_k1_KO_s004
                                 + 0.805707 w_k1_MAT_s004
                                 + 0.788039 w_k1_MSFT_s004
                                 + 0.774688 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.932699 w_k1_ADM_s004
                                 + 0.739127 w_k1_AXP_s004
                                 + 0.937653 w_k1_DIS_s004
                                 + 0.888706 w_k1_FDX_s004
                                 + 0.866364 w_k1_IBM_s004
                                 + 0.851169 w_k1_INTC_s004
                                 + 0.901077 w_k1_KO_s004
                                 + 0.981884 w_k1_MAT_s004
                                 + 0.933885 w_k1_MSFT_s004
                                 + 0.937238 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.15651 w_k1_ADM_s004
                                 + 1.0204 w_k1_AXP_s004
                                 + 0.860434 w_k1_DIS_s004
                                 + 1.02539 w_k1_FDX_s004
                                 + 0.619525 w_k1_IBM_s004
                                 + 1.01961 w_k1_INTC_s004
                                 + 0.776488 w_k1_KO_s004
                                 + 0.688497 w_k1_MAT_s004
                                 + 0.857157 w_k1_MSFT_s004
                                 + 0.88615 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.00376 w_k1_ADM_s004
                                 + 0.804077 w_k1_AXP_s004
                                 + 1.27415 w_k1_DIS_s004
                                 + 1.08538 w_k1_FDX_s004
                                 + 0.990108 w_k1_IBM_s004
                                 + 1.05214 w_k1_INTC_s004
                                 + 1.27039 w_k1_KO_s004
                                 + 0.971624 w_k1_MAT_s004
                                 + 1.237 w_k1_MSFT_s004 + 0.998449 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.41146 w_k1_ADM_s004
                                 + 1.44452 w_k1_AXP_s004
                                 + 1.20804 w_k1_DIS_s004
                                 + 0.985279 w_k1_FDX_s004
                                 + 1.72248 w_k1_IBM_s004
                                 + 0.882749 w_k1_INTC_s004
                                 + 1.46087 w_k1_KO_s004 + 1.77027 w_k1_MAT_s004
                                 + 1.03952 w_k1_MSFT_s004
                                 + 1.18792 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.684453 w_k1_ADM_s004
                                 + 0.892821 w_k1_AXP_s004
                                 + 0.820294 w_k1_DIS_s004
                                 + 0.815809 w_k1_FDX_s004
                                 + 1.15563 w_k1_IBM_s004
                                 + 0.678344 w_k1_INTC_s004
                                 + 0.952469 w_k1_KO_s004
                                 + 1.01053 w_k1_MAT_s004
                                 + 0.714507 w_k1_MSFT_s004
                                 + 0.909103 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.00008 w_k1_ADM_s005
                                 + 1.09787 w_k1_AXP_s005
                                 + 1.42325 w_k1_DIS_s005
                                 + 1.24518 w_k1_FDX_s005
                                 + 0.986826 w_k1_IBM_s005
                                 + 1.13309 w_k1_INTC_s005
                                 + 1.11227 w_k1_KO_s005 + 1.15582 w_k1_MAT_s005
                                 + 1.20427 w_k1_MSFT_s005
                                 + 1.14931 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.768695 w_k1_ADM_s005
                                 + 0.853673 w_k1_AXP_s005
                                 + 0.91487 w_k1_DIS_s005
                                 + 0.759941 w_k1_FDX_s005
                                 + 1.38356 w_k1_IBM_s005
                                 + 0.785155 w_k1_INTC_s005
                                 + 0.940027 w_k1_KO_s005
                                 + 1.35594 w_k1_MAT_s005
                                 + 0.822302 w_k1_MSFT_s005
                                 + 0.872054 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.891171 w_k1_ADM_s005
                                 + 0.979653 w_k1_AXP_s005
                                 + 0.861692 w_k1_DIS_s005
                                 + 0.962809 w_k1_FDX_s005
                                 + 1.12158 w_k1_IBM_s005
                                 + 1.15581 w_k1_INTC_s005
                                 + 1.02312 w_k1_KO_s005 + 1.28339 w_k1_MAT_s005
                                 + 1.0198 w_k1_MSFT_s005
                                 + 0.968685 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.37124 w_k1_ADM_s005
                                 + 1.47567 w_k1_AXP_s005
                                 + 1.34741 w_k1_DIS_s005
                                 + 1.17898 w_k1_FDX_s005
                                 + 1.87158 w_k1_IBM_s005
                                 + 1.09077 w_k1_INTC_s005
                                 + 1.26598 w_k1_KO_s005 + 2.23028 w_k1_MAT_s005
                                 + 1.10033 w_k1_MSFT_s005
                                 + 1.22037 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.06926 w_k1_ADM_s005
                                 + 1.06267 w_k1_AXP_s005
                                 + 0.828341 w_k1_DIS_s005
                                 + 1.02476 w_k1_FDX_s005
                                 + 0.555376 w_k1_IBM_s005
                                 + 0.859728 w_k1_INTC_s005
                                 + 0.603717 w_k1_KO_s005
                                 + 0.744019 w_k1_MAT_s005
                                 + 0.923131 w_k1_MSFT_s005
                                 + 0.88114 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.13003 w_k1_ADM_s005
                                 + 0.778795 w_k1_AXP_s005
                                 + 0.659237 w_k1_DIS_s005
                                 + 0.821622 w_k1_FDX_s005
                                 + 0.698172 w_k1_IBM_s005
                                 + 0.825008 w_k1_INTC_s005
                                 + 0.694947 w_k1_KO_s005
                                 + 0.584265 w_k1_MAT_s005
                                 + 0.888352 w_k1_MSFT_s005
                                 + 0.897798 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.835184 w_k1_ADM_s005
                                 + 0.821146 w_k1_AXP_s005
                                 + 0.720026 w_k1_DIS_s005
                                 + 0.860195 w_k1_FDX_s005
                                 + 0.957594 w_k1_IBM_s005
                                 + 0.966954 w_k1_INTC_s005
                                 + 0.819993 w_k1_KO_s005
                                 + 1.08937 w_k1_MAT_s005
                                 + 0.785265 w_k1_MSFT_s005
                                 + 0.949965 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.93183 w_k1_ADM_s005
                                 + 1.50674 w_k1_AXP_s005
                                 + 1.14514 w_k1_DIS_s005
                                 + 1.12176 w_k1_FDX_s005 + 1.2016 w_k1_IBM_s005
                                 + 1.23303 w_k1_INTC_s005
                                 + 0.980481 w_k1_KO_s005
                                 + 1.33589 w_k1_MAT_s005
                                 + 1.03628 w_k1_MSFT_s005
                                 + 1.05648 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.09052 w_k1_ADM_s006
                                 + 0.78369 w_k1_AXP_s006
                                 + 0.848994 w_k1_DIS_s006
                                 + 0.923923 w_k1_FDX_s006
                                 + 0.874812 w_k1_IBM_s006
                                 + 1.01222 w_k1_INTC_s006
                                 + 0.867314 w_k1_KO_s006
                                 + 0.946817 w_k1_MAT_s006
                                 + 0.949214 w_k1_MSFT_s006
                                 + 1.01197 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.979274 w_k1_ADM_s006
                                 + 1.06804 w_k1_AXP_s006
                                 + 0.882297 w_k1_DIS_s006
                                 + 0.966664 w_k1_FDX_s006
                                 + 1.47057 w_k1_IBM_s006
                                 + 0.983229 w_k1_INTC_s006
                                 + 0.93521 w_k1_KO_s006 + 1.58977 w_k1_MAT_s006
                                 + 0.901429 w_k1_MSFT_s006
                                 + 1.03467 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.986625 w_k1_ADM_s006
                                 + 1.0843 w_k1_AXP_s006 + 1.34067 w_k1_DIS_s006
                                 + 1.04285 w_k1_FDX_s006
                                 + 1.92872 w_k1_IBM_s006
                                 + 1.19235 w_k1_INTC_s006
                                 + 1.19926 w_k1_KO_s006 + 1.55989 w_k1_MAT_s006
                                 + 1.02696 w_k1_MSFT_s006
                                 + 1.10755 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.02046 w_k1_ADM_s006
                                 + 0.662284 w_k1_AXP_s006
                                 + 0.716236 w_k1_DIS_s006
                                 + 0.898897 w_k1_FDX_s006
                                 + 0.694622 w_k1_IBM_s006
                                 + 0.770195 w_k1_INTC_s006
                                 + 0.646166 w_k1_KO_s006
                                 + 0.654848 w_k1_MAT_s006
                                 + 0.775416 w_k1_MSFT_s006
                                 + 0.942451 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.840362 w_k1_ADM_s006
                                 + 1.09229 w_k1_AXP_s006
                                 + 1.26328 w_k1_DIS_s006
                                 + 1.20048 w_k1_FDX_s006
                                 + 0.839255 w_k1_IBM_s006
                                 + 1.16013 w_k1_INTC_s006
                                 + 0.955256 w_k1_KO_s006
                                 + 0.820793 w_k1_MAT_s006
                                 + 1.03308 w_k1_MSFT_s006
                                 + 1.03549 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.949073 w_k1_ADM_s006
                                 + 0.673678 w_k1_AXP_s006
                                 + 0.918236 w_k1_DIS_s006
                                 + 0.902755 w_k1_FDX_s006
                                 + 1.29481 w_k1_IBM_s006
                                 + 1.01723 w_k1_INTC_s006
                                 + 1.03976 w_k1_KO_s006 + 1.51945 w_k1_MAT_s006
                                 + 0.999029 w_k1_MSFT_s006
                                 + 1.05098 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.178 w_k1_ADM_s006
                                 + 1.22158 w_k1_AXP_s006
                                 + 1.17574 w_k1_DIS_s006
                                 + 1.25292 w_k1_FDX_s006
                                 + 1.02072 w_k1_IBM_s006
                                 + 1.03803 w_k1_INTC_s006
                                 + 0.898019 w_k1_KO_s006
                                 + 1.00988 w_k1_MAT_s006
                                 + 1.02612 w_k1_MSFT_s006
                                 + 1.20384 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.23102 w_k1_ADM_s006
                                 + 1.44356 w_k1_AXP_s006
                                 + 1.41347 w_k1_DIS_s006
                                 + 1.27402 w_k1_FDX_s006
                                 + 1.52473 w_k1_IBM_s006 + 1.31 w_k1_INTC_s006
                                 + 1.37044 w_k1_KO_s006 + 1.76988 w_k1_MAT_s006
                                 + 1.10974 w_k1_MSFT_s006
                                 + 1.32881 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.834044 w_k1_ADM_s007
                                 + 0.778901 w_k1_AXP_s007
                                 + 0.802096 w_k1_DIS_s007
                                 + 0.924087 w_k1_FDX_s007
                                 + 0.719428 w_k1_IBM_s007
                                 + 0.850578 w_k1_INTC_s007
                                 + 0.791526 w_k1_KO_s007
                                 + 0.760824 w_k1_MAT_s007
                                 + 0.950112 w_k1_MSFT_s007
                                 + 0.750553 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.81664 w_k1_ADM_s007
                                 + 0.848115 w_k1_AXP_s007
                                 + 0.80309 w_k1_DIS_s007
                                 + 0.858653 w_k1_FDX_s007
                                 + 1.16182 w_k1_IBM_s007
                                 + 0.844461 w_k1_INTC_s007
                                 + 0.96162 w_k1_KO_s007 + 1.17515 w_k1_MAT_s007
                                 + 0.973262 w_k1_MSFT_s007
                                 + 0.815028 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.783171 w_k1_ADM_s007
                                 + 1.09378 w_k1_AXP_s007
                                 + 0.955142 w_k1_DIS_s007
                                 + 0.956059 w_k1_FDX_s007
                                 + 1.46812 w_k1_IBM_s007
                                 + 1.16035 w_k1_INTC_s007
                                 + 0.984311 w_k1_KO_s007
                                 + 1.47855 w_k1_MAT_s007
                                 + 1.05134 w_k1_MSFT_s007
                                 + 0.990806 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.25589 w_k1_ADM_s007
                                 + 1.35943 w_k1_AXP_s007
                                 + 1.63379 w_k1_DIS_s007
                                 + 1.40772 w_k1_FDX_s007
                                 + 1.32993 w_k1_IBM_s007
                                 + 1.31579 w_k1_INTC_s007
                                 + 1.34774 w_k1_KO_s007 + 1.79521 w_k1_MAT_s007
                                 + 1.45163 w_k1_MSFT_s007
                                 + 1.28699 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.23445 w_k1_ADM_s007
                                 + 1.33818 w_k1_AXP_s007
                                 + 0.927765 w_k1_DIS_s007
                                 + 1.17604 w_k1_FDX_s007
                                 + 0.927973 w_k1_IBM_s007
                                 + 0.823038 w_k1_INTC_s007
                                 + 0.885501 w_k1_KO_s007
                                 + 1.15336 w_k1_MAT_s007
                                 + 0.899975 w_k1_MSFT_s007
                                 + 0.921013 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.981996 w_k1_ADM_s007
                                 + 1.05214 w_k1_AXP_s007
                                 + 1.31205 w_k1_DIS_s007
                                 + 1.19274 w_k1_FDX_s007
                                 + 1.06724 w_k1_IBM_s007
                                 + 1.15967 w_k1_INTC_s007
                                 + 1.08801 w_k1_KO_s007
                                 + 0.685715 w_k1_MAT_s007
                                 + 1.24781 w_k1_MSFT_s007
                                 + 1.05961 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.881132 w_k1_ADM_s007
                                 + 1.06459 w_k1_AXP_s007
                                 + 0.999943 w_k1_DIS_s007
                                 + 1.13857 w_k1_FDX_s007
                                 + 1.19073 w_k1_IBM_s007
                                 + 1.06129 w_k1_INTC_s007 + 1.0101 w_k1_KO_s007
                                 + 1.47403 w_k1_MAT_s007
                                 + 1.06762 w_k1_MSFT_s007
                                 + 1.10367 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.706777 w_k1_ADM_s007
                                 + 0.800559 w_k1_AXP_s007
                                 + 0.624234 w_k1_DIS_s007
                                 + 0.758962 w_k1_FDX_s007
                                 + 0.727237 w_k1_IBM_s007
                                 + 0.686468 w_k1_INTC_s007
                                 + 0.720023 w_k1_KO_s007
                                 + 0.951805 w_k1_MAT_s007
                                 + 0.824522 w_k1_MSFT_s007
                                 + 0.796526 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.895766 w_k1_ADM_s010
                                 + 0.920121 w_k1_AXP_s010
                                 + 0.709421 w_k1_DIS_s010
                                 + 0.901102 w_k1_FDX_s010
                                 + 0.678937 w_k1_IBM_s010
                                 + 0.918577 w_k1_INTC_s010
                                 + 0.667001 w_k1_KO_s010
                                 + 0.762715 w_k1_MAT_s010
                                 + 0.800054 w_k1_MSFT_s010
                                 + 0.815895 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.1596 w_k1_ADM_s010
                                 + 1.13373 w_k1_AXP_s010
                                 + 0.957176 w_k1_DIS_s010
                                 + 1.07294 w_k1_FDX_s010
                                 + 0.864385 w_k1_IBM_s010
                                 + 1.13714 w_k1_INTC_s010
                                 + 0.917478 w_k1_KO_s010
                                 + 0.743326 w_k1_MAT_s010
                                 + 0.933579 w_k1_MSFT_s010
                                 + 0.927187 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.907485 w_k1_ADM_s010
                                 + 0.734461 w_k1_AXP_s010
                                 + 0.816954 w_k1_DIS_s010
                                 + 0.915639 w_k1_FDX_s010
                                 + 0.975241 w_k1_IBM_s010
                                 + 1.11422 w_k1_INTC_s010
                                 + 0.901996 w_k1_KO_s010
                                 + 0.80997 w_k1_MAT_s010
                                 + 0.93242 w_k1_MSFT_s010
                                 + 0.865763 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.10691 w_k1_ADM_s010
                                 + 1.09882 w_k1_AXP_s010
                                 + 1.25043 w_k1_DIS_s010
                                 + 1.20696 w_k1_FDX_s010
                                 + 1.17023 w_k1_IBM_s010
                                 + 1.1298 w_k1_INTC_s010 + 1.14808 w_k1_KO_s010
                                 + 1.04965 w_k1_MAT_s010
                                 + 1.02064 w_k1_MSFT_s010
                                 + 0.990499 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.17287 w_k1_ADM_s010
                                 + 0.92989 w_k1_AXP_s010
                                 + 1.14049 w_k1_DIS_s010
                                 + 1.20758 w_k1_FDX_s010
                                 + 0.734204 w_k1_IBM_s010
                                 + 1.07464 w_k1_INTC_s010
                                 + 1.07589 w_k1_KO_s010
                                 + 0.763656 w_k1_MAT_s010
                                 + 1.10807 w_k1_MSFT_s010
                                 + 0.985109 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.36739 w_k1_ADM_s010
                                 + 1.38335 w_k1_AXP_s010
                                 + 1.49422 w_k1_DIS_s010
                                 + 1.40663 w_k1_FDX_s010
                                 + 1.34461 w_k1_IBM_s010
                                 + 1.16753 w_k1_INTC_s010
                                 + 1.23633 w_k1_KO_s010 + 1.42909 w_k1_MAT_s010
                                 + 1.1304 w_k1_MSFT_s010 + 1.15607 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.953704 w_k1_ADM_s010
                                 + 1.1325 w_k1_AXP_s010
                                 + 0.984435 w_k1_DIS_s010
                                 + 1.02904 w_k1_FDX_s010
                                 + 1.34421 w_k1_IBM_s010
                                 + 1.03451 w_k1_INTC_s010
                                 + 1.18405 w_k1_KO_s010 + 1.47857 w_k1_MAT_s010
                                 + 0.975082 w_k1_MSFT_s010
                                 + 0.96197 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.15531 w_k1_ADM_s010
                                 + 1.20249 w_k1_AXP_s010
                                 + 0.937643 w_k1_DIS_s010
                                 + 1.01621 w_k1_FDX_s010
                                 + 1.04715 w_k1_IBM_s010 + 1.196 w_k1_INTC_s010
                                 + 0.993095 w_k1_KO_s010
                                 + 1.01335 w_k1_MAT_s010
                                 + 0.848666 w_k1_MSFT_s010
                                 + 0.831428 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.22222 w_k1_ADM_s011
                                 + 1.33699 w_k1_AXP_s011
                                 + 1.67386 w_k1_DIS_s011
                                 + 1.11661 w_k1_FDX_s011
                                 + 2.09497 w_k1_IBM_s011
                                 + 1.24363 w_k1_INTC_s011
                                 + 1.71256 w_k1_KO_s011 + 1.95018 w_k1_MAT_s011
                                 + 1.19838 w_k1_MSFT_s011
                                 + 1.21763 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.837493 w_k1_ADM_s011
                                 + 0.920691 w_k1_AXP_s011
                                 + 1.08743 w_k1_DIS_s011
                                 + 1.02568 w_k1_FDX_s011
                                 + 0.835027 w_k1_IBM_s011
                                 + 1.19017 w_k1_INTC_s011
                                 + 0.96276 w_k1_KO_s011
                                 + 0.950459 w_k1_MAT_s011
                                 + 1.03682 w_k1_MSFT_s011
                                 + 1.02344 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.97719 w_k1_ADM_s011
                                 + 1.26026 w_k1_AXP_s011
                                 + 1.42452 w_k1_DIS_s011
                                 + 1.13743 w_k1_FDX_s011
                                 + 1.37138 w_k1_IBM_s011
                                 + 1.56397 w_k1_INTC_s011 + 1.493 w_k1_KO_s011
                                 + 1.70505 w_k1_MAT_s011
                                 + 1.14408 w_k1_MSFT_s011
                                 + 1.22558 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.813389 w_k1_ADM_s011
                                 + 0.789093 w_k1_AXP_s011
                                 + 0.7511 w_k1_DIS_s011
                                 + 0.808107 w_k1_FDX_s011
                                 + 0.91784 w_k1_IBM_s011
                                 + 0.921756 w_k1_INTC_s011
                                 + 0.860716 w_k1_KO_s011
                                 + 1.45839 w_k1_MAT_s011
                                 + 0.861183 w_k1_MSFT_s011
                                 + 0.99751 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.19538 w_k1_ADM_s011
                                 + 1.01012 w_k1_AXP_s011
                                 + 1.05324 w_k1_DIS_s011
                                 + 1.19421 w_k1_FDX_s011
                                 + 0.647566 w_k1_IBM_s011
                                 + 1.05995 w_k1_INTC_s011
                                 + 0.958159 w_k1_KO_s011
                                 + 1.05703 w_k1_MAT_s011
                                 + 1.16383 w_k1_MSFT_s011
                                 + 0.94336 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.12447 w_k1_ADM_s011
                                 + 1.02455 w_k1_AXP_s011
                                 + 0.802702 w_k1_DIS_s011
                                 + 1.00892 w_k1_FDX_s011
                                 + 0.749882 w_k1_IBM_s011
                                 + 1.05033 w_k1_INTC_s011
                                 + 0.899121 w_k1_KO_s011
                                 + 0.764223 w_k1_MAT_s011
                                 + 0.947367 w_k1_MSFT_s011
                                 + 0.902454 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.10158 w_k1_ADM_s011
                                 + 1.76168 w_k1_AXP_s011
                                 + 1.36879 w_k1_DIS_s011
                                 + 1.31983 w_k1_FDX_s011
                                 + 1.07059 w_k1_IBM_s011
                                 + 1.36167 w_k1_INTC_s011
                                 + 1.16132 w_k1_KO_s011 + 1.68642 w_k1_MAT_s011
                                 + 1.15792 w_k1_MSFT_s011
                                 + 1.09323 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.861961 w_k1_ADM_s011
                                 + 1.28163 w_k1_AXP_s011 + 1.0517 w_k1_DIS_s011
                                 + 0.918423 w_k1_FDX_s011
                                 + 1.45372 w_k1_IBM_s011
                                 + 1.13652 w_k1_INTC_s011
                                 + 1.16709 w_k1_KO_s011 + 1.54318 w_k1_MAT_s011
                                 + 1.04919 w_k1_MSFT_s011
                                 + 1.08972 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.4253 w_k1_ADM_s012
                                 + 1.40744 w_k1_AXP_s012
                                 + 1.65537 w_k1_DIS_s012
                                 + 1.34231 w_k1_FDX_s012
                                 + 1.62537 w_k1_IBM_s012
                                 + 1.01504 w_k1_INTC_s012
                                 + 1.58011 w_k1_KO_s012 + 1.31948 w_k1_MAT_s012
                                 + 0.95365 w_k1_MSFT_s012 + 1.169 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.29696 w_k1_ADM_s012
                                 + 0.926062 w_k1_AXP_s012
                                 + 0.904262 w_k1_DIS_s012
                                 + 1.10231 w_k1_FDX_s012
                                 + 0.776868 w_k1_IBM_s012
                                 + 0.991385 w_k1_INTC_s012
                                 + 0.851606 w_k1_KO_s012
                                 + 0.807148 w_k1_MAT_s012
                                 + 0.968106 w_k1_MSFT_s012
                                 + 0.966163 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.2977 w_k1_ADM_s012
                                 + 0.935721 w_k1_AXP_s012
                                 + 0.67007 w_k1_DIS_s012
                                 + 0.881974 w_k1_FDX_s012
                                 + 0.692322 w_k1_IBM_s012
                                 + 0.793521 w_k1_INTC_s012
                                 + 0.751863 w_k1_KO_s012
                                 + 0.440854 w_k1_MAT_s012
                                 + 0.8019 w_k1_MSFT_s012
                                 + 0.862096 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.2509 w_k1_ADM_s012
                                 + 0.783914 w_k1_AXP_s012
                                 + 0.880204 w_k1_DIS_s012
                                 + 0.955107 w_k1_FDX_s012
                                 + 1.00514 w_k1_IBM_s012
                                 + 0.885408 w_k1_INTC_s012
                                 + 0.982958 w_k1_KO_s012
                                 + 0.737481 w_k1_MAT_s012
                                 + 0.871455 w_k1_MSFT_s012
                                 + 1.02737 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.15847 w_k1_ADM_s012
                                 + 1.29133 w_k1_AXP_s012
                                 + 1.30559 w_k1_DIS_s012
                                 + 1.20568 w_k1_FDX_s012
                                 + 0.982319 w_k1_IBM_s012
                                 + 1.00272 w_k1_INTC_s012
                                 + 1.20426 w_k1_KO_s012
                                 + 0.968483 w_k1_MAT_s012
                                 + 1.09046 w_k1_MSFT_s012
                                 + 1.12449 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.984844 w_k1_ADM_s012
                                 + 0.75635 w_k1_AXP_s012
                                 + 0.995875 w_k1_DIS_s012
                                 + 0.877865 w_k1_FDX_s012
                                 + 1.21888 w_k1_IBM_s012
                                 + 0.727472 w_k1_INTC_s012
                                 + 1.14926 w_k1_KO_s012
                                 + 0.865188 w_k1_MAT_s012
                                 + 0.933516 w_k1_MSFT_s012
                                 + 1.04224 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.979772 w_k1_ADM_s012
                                 + 1.10206 w_k1_AXP_s012 + 1.059 w_k1_DIS_s012
                                 + 0.943673 w_k1_FDX_s012
                                 + 1.19821 w_k1_IBM_s012
                                 + 0.986104 w_k1_INTC_s012
                                 + 1.07951 w_k1_KO_s012
                                 + 0.958663 w_k1_MAT_s012
                                 + 1.04042 w_k1_MSFT_s012 + 1.0671 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.826784 w_k1_ADM_s012
                                 + 0.576527 w_k1_AXP_s012
                                 + 0.78369 w_k1_DIS_s012
                                 + 0.786813 w_k1_FDX_s012
                                 + 0.652928 w_k1_IBM_s012
                                 + 0.647709 w_k1_INTC_s012
                                 + 0.842524 w_k1_KO_s012
                                 + 0.614136 w_k1_MAT_s012
                                 + 0.717526 w_k1_MSFT_s012
                                 + 0.900855 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.32557 w_k1_ADM_s013
                                 + 1.0631 w_k1_AXP_s013
                                 + 0.988651 w_k1_DIS_s013
                                 + 0.981086 w_k1_FDX_s013
                                 + 1.42033 w_k1_IBM_s013
                                 + 0.843528 w_k1_INTC_s013
                                 + 1.1606 w_k1_KO_s013 + 0.933862 w_k1_MAT_s013
                                 + 1.00053 w_k1_MSFT_s013
                                 + 1.06521 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.805235 w_k1_ADM_s013
                                 + 0.650445 w_k1_AXP_s013
                                 + 0.612116 w_k1_DIS_s013
                                 + 0.743762 w_k1_FDX_s013
                                 + 0.748695 w_k1_IBM_s013
                                 + 0.756685 w_k1_INTC_s013
                                 + 0.764793 w_k1_KO_s013
                                 + 0.477076 w_k1_MAT_s013
                                 + 0.857607 w_k1_MSFT_s013
                                 + 0.888663 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.34199 w_k1_ADM_s013
                                 + 1.60011 w_k1_AXP_s013
                                 + 1.37549 w_k1_DIS_s013 + 1.1114 w_k1_FDX_s013
                                 + 1.95202 w_k1_IBM_s013
                                 + 1.04058 w_k1_INTC_s013
                                 + 1.52343 w_k1_KO_s013 + 1.88279 w_k1_MAT_s013
                                 + 1.19271 w_k1_MSFT_s013
                                 + 1.17919 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.661624 w_k1_ADM_s013
                                 + 0.71391 w_k1_AXP_s013
                                 + 0.577666 w_k1_DIS_s013
                                 + 0.697296 w_k1_FDX_s013
                                 + 1.33793 w_k1_IBM_s013
                                 + 0.928103 w_k1_INTC_s013
                                 + 0.886894 w_k1_KO_s013
                                 + 1.09214 w_k1_MAT_s013
                                 + 0.830163 w_k1_MSFT_s013
                                 + 0.921308 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.906846 w_k1_ADM_s013
                                 + 0.893405 w_k1_AXP_s013
                                 + 1.18164 w_k1_DIS_s013
                                 + 0.98395 w_k1_FDX_s013
                                 + 1.10088 w_k1_IBM_s013
                                 + 0.934589 w_k1_INTC_s013
                                 + 0.996118 w_k1_KO_s013
                                 + 0.702523 w_k1_MAT_s013
                                 + 1.17796 w_k1_MSFT_s013
                                 + 1.00287 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.0729 w_k1_ADM_s013
                                 + 0.987349 w_k1_AXP_s013
                                 + 1.03512 w_k1_DIS_s013
                                 + 1.18252 w_k1_FDX_s013
                                 + 0.732226 w_k1_IBM_s013
                                 + 1.01686 w_k1_INTC_s013
                                 + 0.960418 w_k1_KO_s013
                                 + 0.559501 w_k1_MAT_s013
                                 + 1.13369 w_k1_MSFT_s013 + 1.0481 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.896837 w_k1_ADM_s013
                                 + 0.909069 w_k1_AXP_s013
                                 + 0.728562 w_k1_DIS_s013
                                 + 0.911762 w_k1_FDX_s013
                                 + 0.967549 w_k1_IBM_s013
                                 + 0.848524 w_k1_INTC_s013
                                 + 0.783558 w_k1_KO_s013
                                 + 0.723908 w_k1_MAT_s013
                                 + 0.888409 w_k1_MSFT_s013
                                 + 0.90155 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.35195 w_k1_ADM_s013
                                 + 1.54569 w_k1_AXP_s013
                                 + 1.48862 w_k1_DIS_s013
                                 + 1.30045 w_k1_FDX_s013
                                 + 1.11014 w_k1_IBM_s013
                                 + 1.17262 w_k1_INTC_s013
                                 + 1.19554 w_k1_KO_s013
                                 + 0.862045 w_k1_MAT_s013
                                 + 1.35599 w_k1_MSFT_s013
                                 + 1.18133 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.796323 w_k1_ADM_s014
                                 + 0.706247 w_k1_AXP_s014
                                 + 0.672835 w_k1_DIS_s014
                                 + 0.679914 w_k1_FDX_s014
                                 + 0.990233 w_k1_IBM_s014
                                 + 0.665659 w_k1_INTC_s014
                                 + 0.882067 w_k1_KO_s014
                                 + 0.702235 w_k1_MAT_s014
                                 + 0.706833 w_k1_MSFT_s014
                                 + 0.790799 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.834253 w_k1_ADM_s014
                                 + 0.674216 w_k1_AXP_s014
                                 + 0.676369 w_k1_DIS_s014
                                 + 0.779394 w_k1_FDX_s014
                                 + 0.605972 w_k1_IBM_s014
                                 + 0.85867 w_k1_INTC_s014
                                 + 0.792491 w_k1_KO_s014
                                 + 0.484707 w_k1_MAT_s014
                                 + 0.883245 w_k1_MSFT_s014
                                 + 0.826486 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.989556 w_k1_ADM_s014
                                 + 1.4226 w_k1_AXP_s014 + 1.33519 w_k1_DIS_s014
                                 + 1.10379 w_k1_FDX_s014
                                 + 1.53515 w_k1_IBM_s014
                                 + 1.12255 w_k1_INTC_s014
                                 + 1.48121 w_k1_KO_s014 + 1.80013 w_k1_MAT_s014
                                 + 1.13545 w_k1_MSFT_s014
                                 + 1.19154 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.03003 w_k1_ADM_s014
                                 + 0.896631 w_k1_AXP_s014
                                 + 0.805612 w_k1_DIS_s014
                                 + 0.769063 w_k1_FDX_s014
                                 + 1.34066 w_k1_IBM_s014
                                 + 0.949833 w_k1_INTC_s014
                                 + 1.15576 w_k1_KO_s014 + 1.21222 w_k1_MAT_s014
                                 + 0.811918 w_k1_MSFT_s014
                                 + 1.05424 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.06798 w_k1_ADM_s014
                                 + 0.738489 w_k1_AXP_s014
                                 + 1.16289 w_k1_DIS_s014
                                 + 1.02141 w_k1_FDX_s014
                                 + 0.665365 w_k1_IBM_s014
                                 + 0.969089 w_k1_INTC_s014
                                 + 1.09765 w_k1_KO_s014
                                 + 0.511045 w_k1_MAT_s014
                                 + 1.04423 w_k1_MSFT_s014
                                 + 0.989686 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.13874 w_k1_ADM_s014
                                 + 1.04143 w_k1_AXP_s014
                                 + 0.94685 w_k1_DIS_s014
                                 + 0.970688 w_k1_FDX_s014
                                 + 1.0731 w_k1_IBM_s014
                                 + 0.879269 w_k1_INTC_s014
                                 + 1.12866 w_k1_KO_s014
                                 + 0.841219 w_k1_MAT_s014
                                 + 0.852557 w_k1_MSFT_s014
                                 + 0.965506 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.866097 w_k1_ADM_s014
                                 + 0.844477 w_k1_AXP_s014
                                 + 0.797843 w_k1_DIS_s014
                                 + 0.797632 w_k1_FDX_s014
                                 + 0.885933 w_k1_IBM_s014
                                 + 0.767015 w_k1_INTC_s014
                                 + 1.05905 w_k1_KO_s014
                                 + 0.793667 w_k1_MAT_s014
                                 + 0.845989 w_k1_MSFT_s014
                                 + 0.916107 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.47677 w_k1_ADM_s014
                                 + 1.30853 w_k1_AXP_s014
                                 + 1.51689 w_k1_DIS_s014
                                 + 1.25516 w_k1_FDX_s014
                                 + 0.928402 w_k1_IBM_s014
                                 + 0.987351 w_k1_INTC_s014
                                 + 1.29381 w_k1_KO_s014 + 1.11093 w_k1_MAT_s014
                                 + 1.2499 w_k1_MSFT_s014 + 1.12904 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.0634 w_k1_ADM_s015
                                 + 0.872655 w_k1_AXP_s015
                                 + 0.828881 w_k1_DIS_s015
                                 + 0.965925 w_k1_FDX_s015
                                 + 1.04895 w_k1_IBM_s015
                                 + 1.01833 w_k1_INTC_s015
                                 + 0.906672 w_k1_KO_s015
                                 + 0.853033 w_k1_MAT_s015
                                 + 0.89871 w_k1_MSFT_s015
                                 + 1.06226 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.03996 w_k1_ADM_s015
                                 + 1.38697 w_k1_AXP_s015
                                 + 1.13246 w_k1_DIS_s015
                                 + 1.08998 w_k1_FDX_s015
                                 + 1.73981 w_k1_IBM_s015
                                 + 1.09764 w_k1_INTC_s015
                                 + 1.38489 w_k1_KO_s015 + 2.08099 w_k1_MAT_s015
                                 + 1.00272 w_k1_MSFT_s015
                                 + 1.26373 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.780178 w_k1_ADM_s015
                                 + 0.784088 w_k1_AXP_s015
                                 + 0.800313 w_k1_DIS_s015
                                 + 0.839012 w_k1_FDX_s015
                                 + 0.9931 w_k1_IBM_s015
                                 + 0.894791 w_k1_INTC_s015
                                 + 0.940293 w_k1_KO_s015
                                 + 0.815563 w_k1_MAT_s015
                                 + 0.920206 w_k1_MSFT_s015 + 1.05 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.10717 w_k1_ADM_s015
                                 + 1.14663 w_k1_AXP_s015
                                 + 0.921422 w_k1_DIS_s015
                                 + 0.994709 w_k1_FDX_s015
                                 + 1.24877 w_k1_IBM_s015
                                 + 0.933607 w_k1_INTC_s015
                                 + 1.01824 w_k1_KO_s015 + 1.08108 w_k1_MAT_s015
                                 + 0.984818 w_k1_MSFT_s015
                                 + 1.13642 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.39763 w_k1_ADM_s015
                                 + 1.07654 w_k1_AXP_s015
                                 + 1.33742 w_k1_DIS_s015
                                 + 1.33069 w_k1_FDX_s015
                                 + 0.88633 w_k1_IBM_s015
                                 + 1.21472 w_k1_INTC_s015
                                 + 1.00075 w_k1_KO_s015
                                 + 0.654971 w_k1_MAT_s015
                                 + 1.18594 w_k1_MSFT_s015
                                 + 1.22556 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.26234 w_k1_ADM_s015
                                 + 1.74106 w_k1_AXP_s015
                                 + 1.98085 w_k1_DIS_s015
                                 + 1.41957 w_k1_FDX_s015 + 1.3538 w_k1_IBM_s015
                                 + 1.58269 w_k1_INTC_s015
                                 + 1.46381 w_k1_KO_s015 + 1.5354 w_k1_MAT_s015
                                 + 1.30812 w_k1_MSFT_s015
                                 + 1.30328 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.843991 w_k1_ADM_s015
                                 + 0.584676 w_k1_AXP_s015
                                 + 0.650608 w_k1_DIS_s015
                                 + 0.768122 w_k1_FDX_s015
                                 + 0.696786 w_k1_IBM_s015
                                 + 0.74998 w_k1_INTC_s015
                                 + 0.685951 w_k1_KO_s015
                                 + 0.613113 w_k1_MAT_s015
                                 + 0.803921 w_k1_MSFT_s015
                                 + 0.900176 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.18741 w_k1_ADM_s015
                                 + 1.16831 w_k1_AXP_s015
                                 + 1.14677 w_k1_DIS_s015 + 1.1955 w_k1_FDX_s015
                                 + 0.93779 w_k1_IBM_s015
                                 + 1.11595 w_k1_INTC_s015
                                 + 1.04357 w_k1_KO_s015 + 1.08064 w_k1_MAT_s015
                                 + 1.02601 w_k1_MSFT_s015 + 1.1801 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.627398 w_k1_ADM_s016
                                 + 0.690505 w_k1_AXP_s016
                                 + 0.612448 w_k1_DIS_s016
                                 + 0.731826 w_k1_FDX_s016
                                 + 0.905964 w_k1_IBM_s016
                                 + 1.01876 w_k1_INTC_s016
                                 + 0.895649 w_k1_KO_s016
                                 + 0.858667 w_k1_MAT_s016
                                 + 0.865751 w_k1_MSFT_s016
                                 + 0.796546 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.33826 w_k1_ADM_s016
                                 + 1.17768 w_k1_AXP_s016
                                 + 1.45095 w_k1_DIS_s016
                                 + 1.27283 w_k1_FDX_s016
                                 + 1.23704 w_k1_IBM_s016
                                 + 1.38825 w_k1_INTC_s016
                                 + 1.37177 w_k1_KO_s016 + 1.21512 w_k1_MAT_s016
                                 + 1.2549 w_k1_MSFT_s016 + 1.16673 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.988457 w_k1_ADM_s016
                                 + 0.938018 w_k1_AXP_s016
                                 + 1.12114 w_k1_DIS_s016 + 1.1849 w_k1_FDX_s016
                                 + 0.756987 w_k1_IBM_s016
                                 + 1.19224 w_k1_INTC_s016
                                 + 0.995446 w_k1_KO_s016
                                 + 0.962207 w_k1_MAT_s016
                                 + 1.08626 w_k1_MSFT_s016
                                 + 0.97713 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.821909 w_k1_ADM_s016
                                 + 0.851513 w_k1_AXP_s016
                                 + 1.04919 w_k1_DIS_s016
                                 + 1.05553 w_k1_FDX_s016
                                 + 0.965825 w_k1_IBM_s016
                                 + 1.14138 w_k1_INTC_s016
                                 + 1.04117 w_k1_KO_s016 + 1.04199 w_k1_MAT_s016
                                 + 1.06733 w_k1_MSFT_s016
                                 + 0.908405 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.09858 w_k1_ADM_s016
                                 + 0.885508 w_k1_AXP_s016
                                 + 0.857036 w_k1_DIS_s016
                                 + 0.912478 w_k1_FDX_s016
                                 + 1.51539 w_k1_IBM_s016
                                 + 0.852256 w_k1_INTC_s016
                                 + 0.954712 w_k1_KO_s016
                                 + 1.64484 w_k1_MAT_s016
                                 + 0.948937 w_k1_MSFT_s016
                                 + 0.921807 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.529413 w_k1_ADM_s016
                                 + 0.730066 w_k1_AXP_s016
                                 + 0.696019 w_k1_DIS_s016
                                 + 0.743299 w_k1_FDX_s016
                                 + 1.10901 w_k1_IBM_s016
                                 + 0.826481 w_k1_INTC_s016
                                 + 0.9258 w_k1_KO_s016 + 1.0098 w_k1_MAT_s016
                                 + 0.822189 w_k1_MSFT_s016
                                 + 0.729691 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.88076 w_k1_ADM_s016
                                 + 0.897051 w_k1_AXP_s016
                                 + 0.884025 w_k1_DIS_s016
                                 + 0.935328 w_k1_FDX_s016
                                 + 1.21072 w_k1_IBM_s016
                                 + 1.23169 w_k1_INTC_s016 + 1.0806 w_k1_KO_s016
                                 + 1.09202 w_k1_MAT_s016
                                 + 0.979491 w_k1_MSFT_s016
                                 + 1.04431 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.836752 w_k1_ADM_s016
                                 + 0.692304 w_k1_AXP_s016
                                 + 0.551088 w_k1_DIS_s016
                                 + 0.766164 w_k1_FDX_s016
                                 + 0.629979 w_k1_IBM_s016
                                 + 0.906429 w_k1_INTC_s016
                                 + 0.672709 w_k1_KO_s016
                                 + 0.61557 w_k1_MAT_s016
                                 + 0.755324 w_k1_MSFT_s016
                                 + 0.786524 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.12847 w_k1_ADM_s017
                                 + 1.03841 w_k1_AXP_s017
                                 + 1.75874 w_k1_DIS_s017
                                 + 1.45056 w_k1_FDX_s017
                                 + 0.91533 w_k1_IBM_s017
                                 + 1.28409 w_k1_INTC_s017
                                 + 1.26707 w_k1_KO_s017 + 1.24817 w_k1_MAT_s017
                                 + 1.60038 w_k1_MSFT_s017
                                 + 1.22025 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.953014 w_k1_ADM_s017
                                 + 1.03814 w_k1_AXP_s017
                                 + 1.03769 w_k1_DIS_s017
                                 + 0.965513 w_k1_FDX_s017
                                 + 1.08829 w_k1_IBM_s017
                                 + 0.888479 w_k1_INTC_s017
                                 + 0.997725 w_k1_KO_s017
                                 + 1.20252 w_k1_MAT_s017
                                 + 1.04962 w_k1_MSFT_s017
                                 + 0.983506 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.84208 w_k1_ADM_s017
                                 + 0.71598 w_k1_AXP_s017
                                 + 0.691207 w_k1_DIS_s017
                                 + 0.752426 w_k1_FDX_s017
                                 + 0.952071 w_k1_IBM_s017
                                 + 0.551038 w_k1_INTC_s017
                                 + 0.741489 w_k1_KO_s017
                                 + 0.747639 w_k1_MAT_s017
                                 + 0.714755 w_k1_MSFT_s017
                                 + 0.750321 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.38582 w_k1_ADM_s017
                                 + 1.66357 w_k1_AXP_s017
                                 + 1.27257 w_k1_DIS_s017
                                 + 1.40778 w_k1_FDX_s017
                                 + 0.955943 w_k1_IBM_s017
                                 + 1.21017 w_k1_INTC_s017
                                 + 0.961261 w_k1_KO_s017
                                 + 0.990652 w_k1_MAT_s017
                                 + 1.25682 w_k1_MSFT_s017
                                 + 1.00946 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.929174 w_k1_ADM_s017
                                 + 0.92045 w_k1_AXP_s017
                                 + 0.88826 w_k1_DIS_s017
                                 + 0.917226 w_k1_FDX_s017
                                 + 0.693639 w_k1_IBM_s017
                                 + 0.853088 w_k1_INTC_s017
                                 + 0.687243 w_k1_KO_s017
                                 + 0.671675 w_k1_MAT_s017
                                 + 1.01309 w_k1_MSFT_s017
                                 + 0.791156 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.886104 w_k1_ADM_s017
                                 + 0.830974 w_k1_AXP_s017
                                 + 0.874675 w_k1_DIS_s017
                                 + 0.858168 w_k1_FDX_s017
                                 + 1.43633 w_k1_IBM_s017
                                 + 0.755714 w_k1_INTC_s017
                                 + 1.04743 w_k1_KO_s017 + 1.39615 w_k1_MAT_s017
                                 + 0.893478 w_k1_MSFT_s017
                                 + 1.0008 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.23877 w_k1_ADM_s017
                                 + 1.27744 w_k1_AXP_s017
                                 + 1.71634 w_k1_DIS_s017
                                 + 1.29423 w_k1_FDX_s017
                                 + 1.49647 w_k1_IBM_s017
                                 + 1.14027 w_k1_INTC_s017
                                 + 1.28519 w_k1_KO_s017 + 1.70365 w_k1_MAT_s017
                                 + 1.33559 w_k1_MSFT_s017
                                 + 1.15484 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.22528 w_k1_ADM_s017
                                 + 1.02798 w_k1_AXP_s017
                                 + 1.26872 w_k1_DIS_s017
                                 + 1.15236 w_k1_FDX_s017
                                 + 1.01079 w_k1_IBM_s017
                                 + 0.921025 w_k1_INTC_s017
                                 + 1.0402 w_k1_KO_s017 + 1.06308 w_k1_MAT_s017
                                 + 1.04344 w_k1_MSFT_s017
                                 + 0.959712 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.31907 w_k1_ADM_s100
                                 + 1.30964 w_k1_AXP_s100
                                 + 1.25025 w_k1_DIS_s100 + 1.2764 w_k1_FDX_s100
                                 + 0.936218 w_k1_IBM_s100
                                 + 1.10506 w_k1_INTC_s100
                                 + 0.998132 w_k1_KO_s100
                                 + 1.38114 w_k1_MAT_s100
                                 + 1.38629 w_k1_MSFT_s100
                                 + 1.10288 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.05709 w_k1_ADM_s100
                                 + 1.01363 w_k1_AXP_s100
                                 + 1.02893 w_k1_DIS_s100
                                 + 0.924544 w_k1_FDX_s100
                                 + 1.28024 w_k1_IBM_s100
                                 + 0.942642 w_k1_INTC_s100
                                 + 1.05127 w_k1_KO_s100 + 1.19109 w_k1_MAT_s100
                                 + 1.03144 w_k1_MSFT_s100
                                 + 0.95148 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.02204 w_k1_ADM_s100
                                 + 1.26277 w_k1_AXP_s100
                                 + 1.56012 w_k1_DIS_s100
                                 + 1.16798 w_k1_FDX_s100
                                 + 1.36932 w_k1_IBM_s100
                                 + 1.08754 w_k1_INTC_s100
                                 + 1.28817 w_k1_KO_s100 + 1.70216 w_k1_MAT_s100
                                 + 1.3267 w_k1_MSFT_s100 + 1.04931 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.00104 w_k1_ADM_s100
                                 + 0.990299 w_k1_AXP_s100
                                 + 1.09896 w_k1_DIS_s100
                                 + 1.07204 w_k1_FDX_s100
                                 + 0.858778 w_k1_IBM_s100
                                 + 0.998214 w_k1_INTC_s100
                                 + 1.01268 w_k1_KO_s100 + 1.01868 w_k1_MAT_s100
                                 + 1.19941 w_k1_MSFT_s100
                                 + 0.969834 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.07181 w_k1_ADM_s100
                                 + 1.08745 w_k1_AXP_s100
                                 + 0.721447 w_k1_DIS_s100
                                 + 0.934244 w_k1_FDX_s100
                                 + 0.677157 w_k1_IBM_s100
                                 + 0.818381 w_k1_INTC_s100
                                 + 0.688331 w_k1_KO_s100
                                 + 0.773315 w_k1_MAT_s100
                                 + 0.870274 w_k1_MSFT_s100
                                 + 0.888499 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.841742 w_k1_ADM_s100
                                 + 0.717415 w_k1_AXP_s100
                                 + 0.68523 w_k1_DIS_s100
                                 + 0.702887 w_k1_FDX_s100
                                 + 0.873824 w_k1_IBM_s100
                                 + 0.752162 w_k1_INTC_s100
                                 + 0.834039 w_k1_KO_s100
                                 + 1.14583 w_k1_MAT_s100
                                 + 0.867154 w_k1_MSFT_s100
                                 + 0.868336 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.700585 w_k1_ADM_s100
                                 + 0.770543 w_k1_AXP_s100
                                 + 0.852549 w_k1_DIS_s100
                                 + 0.911792 w_k1_FDX_s100
                                 + 0.635409 w_k1_IBM_s100
                                 + 0.780899 w_k1_INTC_s100
                                 + 0.753051 w_k1_KO_s100
                                 + 0.785475 w_k1_MAT_s100
                                 + 1.10365 w_k1_MSFT_s100
                                 + 0.808752 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.908874 w_k1_ADM_s100
                                 + 0.886006 w_k1_AXP_s100
                                 + 0.732146 w_k1_DIS_s100
                                 + 0.795482 w_k1_FDX_s100
                                 + 0.85175 w_k1_IBM_s100
                                 + 0.858465 w_k1_INTC_s100
                                 + 0.80782 w_k1_KO_s100 + 1.0955 w_k1_MAT_s100
                                 + 0.845884 w_k1_MSFT_s100
                                 + 0.902471 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.00953 w_k1_ADM_s101
                                 + 0.68903 w_k1_AXP_s101
                                 + 0.652971 w_k1_DIS_s101
                                 + 0.829966 w_k1_FDX_s101
                                 + 0.773381 w_k1_IBM_s101
                                 + 0.840865 w_k1_INTC_s101
                                 + 0.976187 w_k1_KO_s101
                                 + 0.848533 w_k1_MAT_s101
                                 + 0.930557 w_k1_MSFT_s101
                                 + 0.866676 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.16683 w_k1_ADM_s101
                                 + 1.0997 w_k1_AXP_s101
                                 + 0.934409 w_k1_DIS_s101
                                 + 1.02364 w_k1_FDX_s101
                                 + 1.29787 w_k1_IBM_s101
                                 + 0.881183 w_k1_INTC_s101
                                 + 1.17075 w_k1_KO_s101
                                 + 0.993247 w_k1_MAT_s101
                                 + 0.882983 w_k1_MSFT_s101
                                 + 0.997373 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.2103 w_k1_ADM_s101
                                 + 1.24216 w_k1_AXP_s101
                                 + 1.26153 w_k1_DIS_s101
                                 + 1.18534 w_k1_FDX_s101
                                 + 1.20048 w_k1_IBM_s101 + 1.082 w_k1_INTC_s101
                                 + 1.36688 w_k1_KO_s101 + 1.24855 w_k1_MAT_s101
                                 + 1.11016 w_k1_MSFT_s101
                                 + 1.06767 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.22411 w_k1_ADM_s101
                                 + 1.39348 w_k1_AXP_s101
                                 + 1.97954 w_k1_DIS_s101 + 1.5014 w_k1_FDX_s101
                                 + 1.39893 w_k1_IBM_s101
                                 + 1.34181 w_k1_INTC_s101
                                 + 1.66387 w_k1_KO_s101 + 1.28091 w_k1_MAT_s101
                                 + 1.50249 w_k1_MSFT_s101
                                 + 1.12398 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.18407 w_k1_ADM_s101
                                 + 1.14908 w_k1_AXP_s101
                                 + 1.24327 w_k1_DIS_s101
                                 + 1.03699 w_k1_FDX_s101 + 1.8391 w_k1_IBM_s101
                                 + 1.20427 w_k1_INTC_s101
                                 + 1.59642 w_k1_KO_s101 + 1.87376 w_k1_MAT_s101
                                 + 1.10579 w_k1_MSFT_s101
                                 + 1.14675 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.816534 w_k1_ADM_s101
                                 + 0.70827 w_k1_AXP_s101
                                 + 0.750201 w_k1_DIS_s101
                                 + 0.758893 w_k1_FDX_s101
                                 + 0.806186 w_k1_IBM_s101
                                 + 0.673855 w_k1_INTC_s101
                                 + 0.967925 w_k1_KO_s101
                                 + 0.932834 w_k1_MAT_s101
                                 + 0.847208 w_k1_MSFT_s101
                                 + 0.765873 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.20648 w_k1_ADM_s101
                                 + 1.17648 w_k1_AXP_s101 + 1.076 w_k1_DIS_s101
                                 + 1.13583 w_k1_FDX_s101
                                 + 0.733097 w_k1_IBM_s101
                                 + 1.01335 w_k1_INTC_s101
                                 + 1.01751 w_k1_KO_s101
                                 + 0.847584 w_k1_MAT_s101
                                 + 1.13096 w_k1_MSFT_s101
                                 + 0.995915 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.994483 w_k1_ADM_s101
                                 + 0.906611 w_k1_AXP_s101
                                 + 1.02981 w_k1_DIS_s101
                                 + 0.947608 w_k1_FDX_s101
                                 + 1.06636 w_k1_IBM_s101
                                 + 1.06026 w_k1_INTC_s101
                                 + 1.30078 w_k1_KO_s101 + 1.33582 w_k1_MAT_s101
                                 + 1.14014 w_k1_MSFT_s101
                                 + 0.971698 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.99115 w_k1_ADM_s102
                                 + 0.959476 w_k1_AXP_s102
                                 + 1.21342 w_k1_DIS_s102
                                 + 1.05522 w_k1_FDX_s102
                                 + 0.808897 w_k1_IBM_s102
                                 + 1.09124 w_k1_INTC_s102
                                 + 1.11901 w_k1_KO_s102 + 0.84713 w_k1_MAT_s102
                                 + 1.25249 w_k1_MSFT_s102
                                 + 1.14882 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.865826 w_k1_ADM_s102
                                 + 0.8153 w_k1_AXP_s102
                                 + 0.764821 w_k1_DIS_s102
                                 + 0.833251 w_k1_FDX_s102
                                 + 0.739481 w_k1_IBM_s102
                                 + 0.874143 w_k1_INTC_s102
                                 + 0.861031 w_k1_KO_s102
                                 + 0.761877 w_k1_MAT_s102
                                 + 0.872938 w_k1_MSFT_s102
                                 + 0.991824 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.922272 w_k1_ADM_s102
                                 + 1.16195 w_k1_AXP_s102
                                 + 1.19227 w_k1_DIS_s102
                                 + 1.02906 w_k1_FDX_s102
                                 + 1.52876 w_k1_IBM_s102
                                 + 1.1538 w_k1_INTC_s102 + 1.37763 w_k1_KO_s102
                                 + 1.75831 w_k1_MAT_s102
                                 + 1.06948 w_k1_MSFT_s102 + 1.2538 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.887001 w_k1_ADM_s102
                                 + 0.861769 w_k1_AXP_s102
                                 + 0.88968 w_k1_DIS_s102
                                 + 0.78108 w_k1_FDX_s102
                                 + 1.09409 w_k1_IBM_s102
                                 + 0.806026 w_k1_INTC_s102
                                 + 1.12043 w_k1_KO_s102 + 1.22536 w_k1_MAT_s102
                                 + 0.949913 w_k1_MSFT_s102
                                 + 1.02157 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.01673 w_k1_ADM_s102
                                 + 1.34413 w_k1_AXP_s102
                                 + 1.02386 w_k1_DIS_s102
                                 + 1.28178 w_k1_FDX_s102
                                 + 0.725333 w_k1_IBM_s102
                                 + 1.17997 w_k1_INTC_s102
                                 + 0.999468 w_k1_KO_s102
                                 + 0.848904 w_k1_MAT_s102
                                 + 1.34819 w_k1_MSFT_s102
                                 + 1.15434 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.749866 w_k1_ADM_s102
                                 + 0.63028 w_k1_AXP_s102
                                 + 0.665778 w_k1_DIS_s102
                                 + 0.762072 w_k1_FDX_s102
                                 + 0.511461 w_k1_IBM_s102
                                 + 0.72843 w_k1_INTC_s102
                                 + 0.654369 w_k1_KO_s102
                                 + 0.412252 w_k1_MAT_s102
                                 + 0.794209 w_k1_MSFT_s102
                                 + 0.824876 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.24701 w_k1_ADM_s102
                                 + 1.0986 w_k1_AXP_s102 + 1.13299 w_k1_DIS_s102
                                 + 1.05806 w_k1_FDX_s102 + 1.0588 w_k1_IBM_s102
                                 + 1.11255 w_k1_INTC_s102
                                 + 1.28635 w_k1_KO_s102 + 1.23196 w_k1_MAT_s102
                                 + 1.17036 w_k1_MSFT_s102
                                 + 1.09888 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.3597 w_k1_ADM_s102
                                 + 1.44792 w_k1_AXP_s102
                                 + 1.85512 w_k1_DIS_s102
                                 + 1.45721 w_k1_FDX_s102
                                 + 1.21881 w_k1_IBM_s102
                                 + 1.38877 w_k1_INTC_s102
                                 + 1.72493 w_k1_KO_s102 + 1.31523 w_k1_MAT_s102
                                 + 1.66101 w_k1_MSFT_s102
                                 + 1.31421 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.11749 w_k1_ADM_s103
                                 + 1.24949 w_k1_AXP_s103
                                 + 0.933832 w_k1_DIS_s103
                                 + 1.18703 w_k1_FDX_s103
                                 + 0.731692 w_k1_IBM_s103
                                 + 1.22148 w_k1_INTC_s103
                                 + 0.859289 w_k1_KO_s103
                                 + 0.915732 w_k1_MAT_s103
                                 + 1.01701 w_k1_MSFT_s103
                                 + 0.959351 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.931473 w_k1_ADM_s103
                                 + 1.31828 w_k1_AXP_s103
                                 + 1.46608 w_k1_DIS_s103
                                 + 1.18951 w_k1_FDX_s103
                                 + 1.14431 w_k1_IBM_s103
                                 + 1.34754 w_k1_INTC_s103 + 1.3727 w_k1_KO_s103
                                 + 1.38093 w_k1_MAT_s103
                                 + 1.21412 w_k1_MSFT_s103
                                 + 1.13234 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.694769 w_k1_ADM_s103
                                 + 0.753932 w_k1_AXP_s103
                                 + 0.6784 w_k1_DIS_s103
                                 + 0.743987 w_k1_FDX_s103
                                 + 0.653184 w_k1_IBM_s103
                                 + 0.886001 w_k1_INTC_s103
                                 + 0.688729 w_k1_KO_s103
                                 + 0.657277 w_k1_MAT_s103
                                 + 0.779517 w_k1_MSFT_s103
                                 + 0.822855 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.822175 w_k1_ADM_s103
                                 + 1.00105 w_k1_AXP_s103
                                 + 0.81225 w_k1_DIS_s103
                                 + 0.853336 w_k1_FDX_s103
                                 + 1.17504 w_k1_IBM_s103
                                 + 1.02522 w_k1_INTC_s103
                                 + 0.982596 w_k1_KO_s103
                                 + 1.41079 w_k1_MAT_s103
                                 + 0.855349 w_k1_MSFT_s103
                                 + 0.946879 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.04239 w_k1_ADM_s103
                                 + 0.843339 w_k1_AXP_s103
                                 + 1.08862 w_k1_DIS_s103
                                 + 1.04741 w_k1_FDX_s103
                                 + 0.787058 w_k1_IBM_s103
                                 + 1.04775 w_k1_INTC_s103
                                 + 0.966167 w_k1_KO_s103
                                 + 0.95923 w_k1_MAT_s103
                                 + 1.11889 w_k1_MSFT_s103
                                 + 0.998887 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.900643 w_k1_ADM_s103
                                 + 1.09757 w_k1_AXP_s103
                                 + 1.04028 w_k1_DIS_s103
                                 + 1.02712 w_k1_FDX_s103
                                 + 0.925766 w_k1_IBM_s103
                                 + 1.08669 w_k1_INTC_s103 + 1.128 w_k1_KO_s103
                                 + 1.1365 w_k1_MAT_s103 + 1.0464 w_k1_MSFT_s103
                                 + 1.00505 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.00672 w_k1_ADM_s103
                                 + 1.53511 w_k1_AXP_s103
                                 + 1.06493 w_k1_DIS_s103
                                 + 1.03848 w_k1_FDX_s103
                                 + 1.33189 w_k1_IBM_s103
                                 + 1.10384 w_k1_INTC_s103
                                 + 1.19284 w_k1_KO_s103 + 2.24861 w_k1_MAT_s103
                                 + 1.09725 w_k1_MSFT_s103
                                 + 1.16387 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.35655 w_k1_ADM_s103
                                 + 1.77311 w_k1_AXP_s103
                                 + 1.41932 w_k1_DIS_s103
                                 + 1.30573 w_k1_FDX_s103
                                 + 1.55616 w_k1_IBM_s103
                                 + 1.08137 w_k1_INTC_s103
                                 + 1.21912 w_k1_KO_s103 + 2.62818 w_k1_MAT_s103
                                 + 1.18873 w_k1_MSFT_s103
                                 + 1.11248 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.09724 w_k1_ADM_s104
                                 + 1.11415 w_k1_AXP_s104
                                 + 1.03768 w_k1_DIS_s104
                                 + 1.05704 w_k1_FDX_s104
                                 + 1.30763 w_k1_IBM_s104
                                 + 1.12766 w_k1_INTC_s104
                                 + 1.00085 w_k1_KO_s104 + 1.56944 w_k1_MAT_s104
                                 + 1.1123 w_k1_MSFT_s104
                                 + 0.952303 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.14126 w_k1_ADM_s104
                                 + 0.791981 w_k1_AXP_s104
                                 + 0.942114 w_k1_DIS_s104
                                 + 1.1155 w_k1_FDX_s104
                                 + 0.777898 w_k1_IBM_s104
                                 + 0.931308 w_k1_INTC_s104
                                 + 0.865877 w_k1_KO_s104
                                 + 0.557124 w_k1_MAT_s104
                                 + 1.0894 w_k1_MSFT_s104
                                 + 0.948717 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.858829 w_k1_ADM_s104
                                 + 0.637452 w_k1_AXP_s104
                                 + 0.82469 w_k1_DIS_s104
                                 + 0.810613 w_k1_FDX_s104
                                 + 0.99627 w_k1_IBM_s104
                                 + 0.926173 w_k1_INTC_s104
                                 + 0.993612 w_k1_KO_s104
                                 + 1.00425 w_k1_MAT_s104
                                 + 0.934338 w_k1_MSFT_s104
                                 + 0.918969 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.03626 w_k1_ADM_s104
                                 + 1.28945 w_k1_AXP_s104
                                 + 1.41542 w_k1_DIS_s104 + 1.0707 w_k1_FDX_s104
                                 + 1.74365 w_k1_IBM_s104
                                 + 1.23842 w_k1_INTC_s104
                                 + 1.25262 w_k1_KO_s104 + 1.76862 w_k1_MAT_s104
                                 + 1.04927 w_k1_MSFT_s104
                                 + 1.06815 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.982808 w_k1_ADM_s104
                                 + 0.714013 w_k1_AXP_s104
                                 + 1.21941 w_k1_DIS_s104
                                 + 1.02454 w_k1_FDX_s104
                                 + 0.951648 w_k1_IBM_s104
                                 + 1.12018 w_k1_INTC_s104
                                 + 1.00564 w_k1_KO_s104
                                 + 0.978075 w_k1_MAT_s104
                                 + 1.102 w_k1_MSFT_s104 + 1.04783 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.37244 w_k1_ADM_s104
                                 + 1.18828 w_k1_AXP_s104
                                 + 1.56619 w_k1_DIS_s104
                                 + 1.35632 w_k1_FDX_s104
                                 + 0.919419 w_k1_IBM_s104
                                 + 1.18847 w_k1_INTC_s104
                                 + 1.14598 w_k1_KO_s104
                                 + 0.812695 w_k1_MAT_s104
                                 + 1.31801 w_k1_MSFT_s104
                                 + 1.12483 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.61002 w_k1_ADM_s104
                                 + 1.60183 w_k1_AXP_s104
                                 + 1.91845 w_k1_DIS_s104
                                 + 1.50435 w_k1_FDX_s104
                                 + 1.66686 w_k1_IBM_s104
                                 + 1.46752 w_k1_INTC_s104
                                 + 1.60758 w_k1_KO_s104 + 1.7479 w_k1_MAT_s104
                                 + 1.76067 w_k1_MSFT_s104
                                 + 1.35209 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.00997 w_k1_ADM_s104
                                 + 0.909548 w_k1_AXP_s104
                                 + 0.967388 w_k1_DIS_s104
                                 + 0.952699 w_k1_FDX_s104
                                 + 0.752358 w_k1_IBM_s104
                                 + 0.826317 w_k1_INTC_s104
                                 + 0.880804 w_k1_KO_s104
                                 + 0.785715 w_k1_MAT_s104
                                 + 0.973158 w_k1_MSFT_s104
                                 + 0.980032 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.11176 w_k1_ADM_s105
                                 + 1.32064 w_k1_AXP_s105
                                 + 0.806767 w_k1_DIS_s105
                                 + 1.01636 w_k1_FDX_s105
                                 + 1.10054 w_k1_IBM_s105
                                 + 0.965397 w_k1_INTC_s105
                                 + 0.917018 w_k1_KO_s105
                                 + 1.57779 w_k1_MAT_s105
                                 + 0.842668 w_k1_MSFT_s105
                                 + 0.994303 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.36309 w_k1_ADM_s105
                                 + 1.47388 w_k1_AXP_s105
                                 + 1.44929 w_k1_DIS_s105
                                 + 1.38772 w_k1_FDX_s105
                                 + 1.26955 w_k1_IBM_s105
                                 + 1.45667 w_k1_INTC_s105
                                 + 1.25484 w_k1_KO_s105 + 1.6305 w_k1_MAT_s105
                                 + 1.29737 w_k1_MSFT_s105
                                 + 1.17414 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.15945 w_k1_ADM_s105
                                 + 1.23138 w_k1_AXP_s105
                                 + 1.46114 w_k1_DIS_s105
                                 + 1.25245 w_k1_FDX_s105
                                 + 1.63986 w_k1_IBM_s105
                                 + 1.11429 w_k1_INTC_s105
                                 + 1.36628 w_k1_KO_s105 + 1.7133 w_k1_MAT_s105
                                 + 1.11293 w_k1_MSFT_s105 + 1.2451 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.781919 w_k1_ADM_s105
                                 + 0.74369 w_k1_AXP_s105
                                 + 0.779398 w_k1_DIS_s105
                                 + 0.867186 w_k1_FDX_s105
                                 + 0.939225 w_k1_IBM_s105
                                 + 0.956091 w_k1_INTC_s105
                                 + 0.880295 w_k1_KO_s105
                                 + 0.975515 w_k1_MAT_s105
                                 + 0.786034 w_k1_MSFT_s105
                                 + 0.88354 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.81437 w_k1_ADM_s105
                                 + 0.665838 w_k1_AXP_s105
                                 + 0.622389 w_k1_DIS_s105
                                 + 0.759847 w_k1_FDX_s105
                                 + 0.656293 w_k1_IBM_s105
                                 + 0.948799 w_k1_INTC_s105
                                 + 0.756527 w_k1_KO_s105 + 0.6791 w_k1_MAT_s105
                                 + 0.778343 w_k1_MSFT_s105
                                 + 0.761268 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.956037 w_k1_ADM_s105
                                 + 0.766371 w_k1_AXP_s105
                                 + 0.83818 w_k1_DIS_s105
                                 + 0.974051 w_k1_FDX_s105
                                 + 0.57356 w_k1_IBM_s105
                                 + 0.914646 w_k1_INTC_s105
                                 + 0.834873 w_k1_KO_s105
                                 + 0.532373 w_k1_MAT_s105
                                 + 0.892099 w_k1_MSFT_s105
                                 + 0.908174 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.21653 w_k1_ADM_s105
                                 + 0.961574 w_k1_AXP_s105
                                 + 1.23801 w_k1_DIS_s105
                                 + 1.06188 w_k1_FDX_s105
                                 + 1.26754 w_k1_IBM_s105
                                 + 1.05792 w_k1_INTC_s105
                                 + 1.18265 w_k1_KO_s105 + 1.0488 w_k1_MAT_s105
                                 + 1.03645 w_k1_MSFT_s105
                                 + 1.07321 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.09334 w_k1_ADM_s105
                                 + 1.05682 w_k1_AXP_s105
                                 + 1.24874 w_k1_DIS_s105
                                 + 1.19737 w_k1_FDX_s105
                                 + 0.864381 w_k1_IBM_s105
                                 + 1.09418 w_k1_INTC_s105
                                 + 0.89281 w_k1_KO_s105
                                 + 0.656263 w_k1_MAT_s105
                                 + 1.13142 w_k1_MSFT_s105
                                 + 0.986268 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.831845 w_k1_ADM_s106
                                 + 0.699261 w_k1_AXP_s106
                                 + 0.670059 w_k1_DIS_s106
                                 + 0.713814 w_k1_FDX_s106
                                 + 1.01075 w_k1_IBM_s106
                                 + 0.897692 w_k1_INTC_s106
                                 + 0.827453 w_k1_KO_s106
                                 + 0.584608 w_k1_MAT_s106
                                 + 0.738367 w_k1_MSFT_s106
                                 + 0.837867 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.06885 w_k1_ADM_s106
                                 + 1.02373 w_k1_AXP_s106
                                 + 0.832022 w_k1_DIS_s106
                                 + 0.970717 w_k1_FDX_s106
                                 + 0.893428 w_k1_IBM_s106
                                 + 0.935223 w_k1_INTC_s106
                                 + 0.962841 w_k1_KO_s106
                                 + 1.01068 w_k1_MAT_s106
                                 + 0.919309 w_k1_MSFT_s106
                                 + 0.868891 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.30005 w_k1_ADM_s106
                                 + 1.75674 w_k1_AXP_s106
                                 + 1.75912 w_k1_DIS_s106
                                 + 1.28339 w_k1_FDX_s106
                                 + 1.62416 w_k1_IBM_s106
                                 + 1.28649 w_k1_INTC_s106
                                 + 1.53029 w_k1_KO_s106 + 1.61861 w_k1_MAT_s106
                                 + 1.24743 w_k1_MSFT_s106
                                 + 1.28917 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.27739 w_k1_ADM_s106
                                 + 0.967937 w_k1_AXP_s106
                                 + 1.18697 w_k1_DIS_s106
                                 + 0.978145 w_k1_FDX_s106
                                 + 1.09601 w_k1_IBM_s106
                                 + 1.03262 w_k1_INTC_s106
                                 + 0.999117 w_k1_KO_s106
                                 + 0.796032 w_k1_MAT_s106
                                 + 0.979912 w_k1_MSFT_s106
                                 + 0.934037 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.34916 w_k1_ADM_s106
                                 + 1.49295 w_k1_AXP_s106
                                 + 1.44752 w_k1_DIS_s106
                                 + 1.34453 w_k1_FDX_s106
                                 + 1.02527 w_k1_IBM_s106
                                 + 1.42856 w_k1_INTC_s106
                                 + 1.11602 w_k1_KO_s106 + 1.18718 w_k1_MAT_s106
                                 + 1.32025 w_k1_MSFT_s106
                                 + 1.20306 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.891014 w_k1_ADM_s106
                                 + 1.00579 w_k1_AXP_s106
                                 + 1.15768 w_k1_DIS_s106
                                 + 0.890313 w_k1_FDX_s106
                                 + 1.44506 w_k1_IBM_s106
                                 + 1.06395 w_k1_INTC_s106
                                 + 1.06021 w_k1_KO_s106 + 1.04266 w_k1_MAT_s106
                                 + 1.01701 w_k1_MSFT_s106
                                 + 0.941379 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.802271 w_k1_ADM_s106
                                 + 0.623127 w_k1_AXP_s106
                                 + 0.652963 w_k1_DIS_s106
                                 + 0.687082 w_k1_FDX_s106
                                 + 0.763437 w_k1_IBM_s106
                                 + 0.769469 w_k1_INTC_s106
                                 + 0.806565 w_k1_KO_s106
                                 + 0.612807 w_k1_MAT_s106
                                 + 0.7863 w_k1_MSFT_s106
                                 + 0.704272 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.24647 w_k1_ADM_s106
                                 + 0.940181 w_k1_AXP_s106
                                 + 1.06112 w_k1_DIS_s106
                                 + 1.02489 w_k1_FDX_s106
                                 + 0.699954 w_k1_IBM_s106
                                 + 1.116 w_k1_INTC_s106 + 0.855848 w_k1_KO_s106
                                 + 0.534845 w_k1_MAT_s106
                                 + 1.07031 w_k1_MSFT_s106
                                 + 0.938632 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.03596 w_k1_ADM_s107
                                 + 1.0745 w_k1_AXP_s107
                                 + 0.856066 w_k1_DIS_s107
                                 + 0.984784 w_k1_FDX_s107
                                 + 0.791411 w_k1_IBM_s107
                                 + 0.945038 w_k1_INTC_s107
                                 + 0.837605 w_k1_KO_s107
                                 + 1.21754 w_k1_MAT_s107
                                 + 0.97034 w_k1_MSFT_s107
                                 + 1.03317 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.909906 w_k1_ADM_s107
                                 + 1.36464 w_k1_AXP_s107
                                 + 1.43569 w_k1_DIS_s107
                                 + 1.23816 w_k1_FDX_s107 + 1.5564 w_k1_IBM_s107
                                 + 1.06932 w_k1_INTC_s107 + 1.4862 w_k1_KO_s107
                                 + 2.44583 w_k1_MAT_s107
                                 + 1.18835 w_k1_MSFT_s107
                                 + 1.17017 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.733635 w_k1_ADM_s107
                                 + 0.689328 w_k1_AXP_s107
                                 + 0.740928 w_k1_DIS_s107
                                 + 0.793747 w_k1_FDX_s107
                                 + 0.944448 w_k1_IBM_s107
                                 + 0.832287 w_k1_INTC_s107
                                 + 0.860773 w_k1_KO_s107
                                 + 1.15311 w_k1_MAT_s107
                                 + 0.90754 w_k1_MSFT_s107
                                 + 0.883756 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.964608 w_k1_ADM_s107
                                 + 1.00794 w_k1_AXP_s107
                                 + 1.09888 w_k1_DIS_s107
                                 + 1.04831 w_k1_FDX_s107
                                 + 0.962126 w_k1_IBM_s107
                                 + 1.2341 w_k1_INTC_s107
                                 + 0.918601 w_k1_KO_s107
                                 + 1.19524 w_k1_MAT_s107
                                 + 1.14854 w_k1_MSFT_s107
                                 + 1.06898 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.12986 w_k1_ADM_s107
                                 + 1.19154 w_k1_AXP_s107
                                 + 1.18332 w_k1_DIS_s107
                                 + 1.31704 w_k1_FDX_s107
                                 + 0.915494 w_k1_IBM_s107
                                 + 1.12864 w_k1_INTC_s107
                                 + 0.956498 w_k1_KO_s107
                                 + 1.55448 w_k1_MAT_s107
                                 + 1.19353 w_k1_MSFT_s107
                                 + 1.07693 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.669592 w_k1_ADM_s107
                                 + 0.809284 w_k1_AXP_s107
                                 + 0.711306 w_k1_DIS_s107
                                 + 0.870077 w_k1_FDX_s107
                                 + 0.692769 w_k1_IBM_s107
                                 + 0.91295 w_k1_INTC_s107
                                 + 0.743368 w_k1_KO_s107
                                 + 0.958572 w_k1_MAT_s107
                                 + 0.849473 w_k1_MSFT_s107
                                 + 0.865011 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.00108 w_k1_ADM_s107
                                 + 1.09063 w_k1_AXP_s107
                                 + 1.37969 w_k1_DIS_s107
                                 + 1.18752 w_k1_FDX_s107
                                 + 1.22461 w_k1_IBM_s107
                                 + 1.11687 w_k1_INTC_s107
                                 + 1.26289 w_k1_KO_s107 + 1.58851 w_k1_MAT_s107
                                 + 1.31233 w_k1_MSFT_s107 + 1.0594 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.776626 w_k1_ADM_s107
                                 + 0.776096 w_k1_AXP_s107
                                 + 0.717601 w_k1_DIS_s107
                                 + 0.812965 w_k1_FDX_s107
                                 + 1.20469 w_k1_IBM_s107
                                 + 0.911978 w_k1_INTC_s107
                                 + 0.916351 w_k1_KO_s107 + 1.3868 w_k1_MAT_s107
                                 + 0.847153 w_k1_MSFT_s107
                                 + 0.946553 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.11687 w_k1_ADM_s110
                                 + 1.05475 w_k1_AXP_s110
                                 + 1.18449 w_k1_DIS_s110
                                 + 0.993679 w_k1_FDX_s110
                                 + 1.1199 w_k1_IBM_s110
                                 + 0.958598 w_k1_INTC_s110
                                 + 0.99778 w_k1_KO_s110 + 1.02515 w_k1_MAT_s110
                                 + 1.01271 w_k1_MSFT_s110
                                 + 1.01907 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.41906 w_k1_ADM_s110
                                 + 1.39722 w_k1_AXP_s110
                                 + 1.18046 w_k1_DIS_s110
                                 + 1.27684 w_k1_FDX_s110
                                 + 0.983289 w_k1_IBM_s110
                                 + 0.893695 w_k1_INTC_s110
                                 + 1.03138 w_k1_KO_s110
                                 + 0.902086 w_k1_MAT_s110
                                 + 1.08408 w_k1_MSFT_s110
                                 + 1.10295 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.22184 w_k1_ADM_s110
                                 + 1.30255 w_k1_AXP_s110
                                 + 0.891221 w_k1_DIS_s110
                                 + 1.02897 w_k1_FDX_s110
                                 + 1.27384 w_k1_IBM_s110
                                 + 0.868384 w_k1_INTC_s110
                                 + 0.926694 w_k1_KO_s110
                                 + 1.04559 w_k1_MAT_s110
                                 + 0.886689 w_k1_MSFT_s110
                                 + 1.03935 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.04603 w_k1_ADM_s110
                                 + 0.99353 w_k1_AXP_s110
                                 + 1.25691 w_k1_DIS_s110
                                 + 1.17015 w_k1_FDX_s110
                                 + 0.839859 w_k1_IBM_s110
                                 + 0.831095 w_k1_INTC_s110
                                 + 0.892924 w_k1_KO_s110
                                 + 0.844988 w_k1_MAT_s110
                                 + 1.08396 w_k1_MSFT_s110
                                 + 1.08781 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.03691 w_k1_ADM_s110
                                 + 0.914234 w_k1_AXP_s110
                                 + 0.70317 w_k1_DIS_s110
                                 + 0.778603 w_k1_FDX_s110
                                 + 1.39677 w_k1_IBM_s110
                                 + 0.654931 w_k1_INTC_s110
                                 + 0.912489 w_k1_KO_s110 + 1.1013 w_k1_MAT_s110
                                 + 0.62762 w_k1_MSFT_s110
                                 + 0.844579 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.30292 w_k1_ADM_s110
                                 + 1.33089 w_k1_AXP_s110
                                 + 1.28844 w_k1_DIS_s110
                                 + 1.07355 w_k1_FDX_s110
                                 + 1.69818 w_k1_IBM_s110
                                 + 0.832013 w_k1_INTC_s110
                                 + 1.1969 w_k1_KO_s110 + 1.95604 w_k1_MAT_s110
                                 + 0.997641 w_k1_MSFT_s110
                                 + 1.07682 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.03687 w_k1_ADM_s110
                                 + 0.78057 w_k1_AXP_s110
                                 + 0.760001 w_k1_DIS_s110
                                 + 0.75771 w_k1_FDX_s110
                                 + 1.04576 w_k1_IBM_s110
                                 + 0.708721 w_k1_INTC_s110
                                 + 0.931532 w_k1_KO_s110
                                 + 0.947652 w_k1_MAT_s110
                                 + 0.799701 w_k1_MSFT_s110
                                 + 0.835001 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.11272 w_k1_ADM_s110
                                 + 0.738581 w_k1_AXP_s110
                                 + 0.740623 w_k1_DIS_s110
                                 + 0.824781 w_k1_FDX_s110
                                 + 0.712857 w_k1_IBM_s110
                                 + 0.80376 w_k1_INTC_s110
                                 + 0.756296 w_k1_KO_s110
                                 + 0.489119 w_k1_MAT_s110
                                 + 0.814598 w_k1_MSFT_s110
                                 + 0.888876 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.15808 w_k1_ADM_s111
                                 + 0.946053 w_k1_AXP_s111
                                 + 1.02472 w_k1_DIS_s111
                                 + 0.934086 w_k1_FDX_s111
                                 + 1.32526 w_k1_IBM_s111
                                 + 0.980135 w_k1_INTC_s111
                                 + 0.984821 w_k1_KO_s111
                                 + 1.47835 w_k1_MAT_s111
                                 + 1.10516 w_k1_MSFT_s111
                                 + 0.989902 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.92595 w_k1_ADM_s111
                                 + 1.40001 w_k1_AXP_s111
                                 + 1.71582 w_k1_DIS_s111
                                 + 1.23516 w_k1_FDX_s111
                                 + 1.55873 w_k1_IBM_s111
                                 + 1.36692 w_k1_INTC_s111
                                 + 1.28316 w_k1_KO_s111 + 2.25956 w_k1_MAT_s111
                                 + 1.35237 w_k1_MSFT_s111
                                 + 1.22023 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.9723 w_k1_ADM_s111
                                 + 1.19928 w_k1_AXP_s111
                                 + 1.13194 w_k1_DIS_s111
                                 + 1.01639 w_k1_FDX_s111
                                 + 1.16362 w_k1_IBM_s111
                                 + 0.850715 w_k1_INTC_s111
                                 + 1.00111 w_k1_KO_s111 + 1.93118 w_k1_MAT_s111
                                 + 1.09556 w_k1_MSFT_s111
                                 + 1.12405 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.17788 w_k1_ADM_s111
                                 + 0.985147 w_k1_AXP_s111
                                 + 1.13361 w_k1_DIS_s111
                                 + 1.04021 w_k1_FDX_s111
                                 + 0.891038 w_k1_IBM_s111
                                 + 0.930579 w_k1_INTC_s111
                                 + 1.01878 w_k1_KO_s111 + 1.17463 w_k1_MAT_s111
                                 + 1.11664 w_k1_MSFT_s111
                                 + 1.02843 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.2229 w_k1_ADM_s111
                                 + 0.905845 w_k1_AXP_s111
                                 + 0.6026 w_k1_DIS_s111
                                 + 0.873983 w_k1_FDX_s111
                                 + 0.894779 w_k1_IBM_s111
                                 + 0.722852 w_k1_INTC_s111
                                 + 0.791721 w_k1_KO_s111
                                 + 1.08463 w_k1_MAT_s111
                                 + 0.908482 w_k1_MSFT_s111
                                 + 0.986457 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.977651 w_k1_ADM_s111
                                 + 0.853702 w_k1_AXP_s111
                                 + 0.812787 w_k1_DIS_s111
                                 + 0.867168 w_k1_FDX_s111
                                 + 0.977179 w_k1_IBM_s111
                                 + 0.937632 w_k1_INTC_s111
                                 + 0.860897 w_k1_KO_s111
                                 + 0.765134 w_k1_MAT_s111
                                 + 0.957632 w_k1_MSFT_s111
                                 + 0.954698 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.925277 w_k1_ADM_s111
                                 + 0.635397 w_k1_AXP_s111
                                 + 0.68669 w_k1_DIS_s111
                                 + 0.747304 w_k1_FDX_s111
                                 + 0.711986 w_k1_IBM_s111
                                 + 0.730096 w_k1_INTC_s111
                                 + 0.735254 w_k1_KO_s111
                                 + 0.744625 w_k1_MAT_s111
                                 + 0.830746 w_k1_MSFT_s111
                                 + 0.921103 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.118 w_k1_ADM_s111
                                 + 1.23872 w_k1_AXP_s111
                                 + 1.30976 w_k1_DIS_s111
                                 + 1.23585 w_k1_FDX_s111
                                 + 0.953381 w_k1_IBM_s111
                                 + 1.00651 w_k1_INTC_s111
                                 + 0.964997 w_k1_KO_s111
                                 + 1.01968 w_k1_MAT_s111
                                 + 1.17661 w_k1_MSFT_s111
                                 + 1.01587 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.842707 w_k1_ADM_s112
                                 + 0.72333 w_k1_AXP_s112
                                 + 0.816026 w_k1_DIS_s112
                                 + 0.811987 w_k1_FDX_s112
                                 + 1.19221 w_k1_IBM_s112
                                 + 0.763855 w_k1_INTC_s112
                                 + 0.999145 w_k1_KO_s112
                                 + 1.19716 w_k1_MAT_s112
                                 + 0.866321 w_k1_MSFT_s112
                                 + 0.927403 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.732275 w_k1_ADM_s112
                                 + 0.603625 w_k1_AXP_s112
                                 + 0.666853 w_k1_DIS_s112
                                 + 0.815822 w_k1_FDX_s112
                                 + 0.706264 w_k1_IBM_s112
                                 + 0.864701 w_k1_INTC_s112
                                 + 0.829533 w_k1_KO_s112
                                 + 0.836067 w_k1_MAT_s112
                                 + 0.99127 w_k1_MSFT_s112
                                 + 0.925307 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.06306 w_k1_ADM_s112
                                 + 1.03636 w_k1_AXP_s112
                                 + 1.44609 w_k1_DIS_s112
                                 + 1.33864 w_k1_FDX_s112
                                 + 1.21668 w_k1_IBM_s112
                                 + 0.916425 w_k1_INTC_s112
                                 + 1.19466 w_k1_KO_s112 + 1.3745 w_k1_MAT_s112
                                 + 1.0877 w_k1_MSFT_s112 + 1.16683 w_k1_PG_s112
                                 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.966838 w_k1_ADM_s112
                                 + 0.657463 w_k1_AXP_s112
                                 + 0.684596 w_k1_DIS_s112
                                 + 0.831396 w_k1_FDX_s112
                                 + 0.957766 w_k1_IBM_s112
                                 + 0.640756 w_k1_INTC_s112
                                 + 0.923757 w_k1_KO_s112
                                 + 1.02562 w_k1_MAT_s112
                                 + 0.810993 w_k1_MSFT_s112
                                 + 0.879704 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.08824 w_k1_ADM_s112
                                 + 0.867693 w_k1_AXP_s112
                                 + 1.02482 w_k1_DIS_s112 + 1.1143 w_k1_FDX_s112
                                 + 0.670034 w_k1_IBM_s112
                                 + 0.919774 w_k1_INTC_s112
                                 + 0.950013 w_k1_KO_s112
                                 + 0.904632 w_k1_MAT_s112
                                 + 1.01121 w_k1_MSFT_s112
                                 + 1.00817 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.10882 w_k1_ADM_s112
                                 + 0.981321 w_k1_AXP_s112
                                 + 0.944814 w_k1_DIS_s112
                                 + 1.17122 w_k1_FDX_s112
                                 + 0.98507 w_k1_IBM_s112
                                 + 0.977231 w_k1_INTC_s112
                                 + 1.09999 w_k1_KO_s112 + 1.34131 w_k1_MAT_s112
                                 + 1.0773 w_k1_MSFT_s112 + 1.06644 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.1953 w_k1_ADM_s112
                                 + 1.33884 w_k1_AXP_s112
                                 + 1.12709 w_k1_DIS_s112
                                 + 1.02972 w_k1_FDX_s112
                                 + 1.55182 w_k1_IBM_s112
                                 + 0.868246 w_k1_INTC_s112
                                 + 1.33924 w_k1_KO_s112 + 2.43084 w_k1_MAT_s112
                                 + 0.880966 w_k1_MSFT_s112
                                 + 1.18792 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.861437 w_k1_ADM_s112
                                 + 0.86131 w_k1_AXP_s112
                                 + 0.887498 w_k1_DIS_s112
                                 + 0.951022 w_k1_FDX_s112
                                 + 0.829637 w_k1_IBM_s112
                                 + 1.02122 w_k1_INTC_s112
                                 + 0.877331 w_k1_KO_s112
                                 + 0.904825 w_k1_MAT_s112
                                 + 1.07035 w_k1_MSFT_s112
                                 + 0.888797 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.698985 w_k1_ADM_s113
                                 + 0.830798 w_k1_AXP_s113
                                 + 0.5897 w_k1_DIS_s113
                                 + 0.786586 w_k1_FDX_s113
                                 + 0.759401 w_k1_IBM_s113
                                 + 1.0421 w_k1_INTC_s113
                                 + 0.769681 w_k1_KO_s113
                                 + 0.63303 w_k1_MAT_s113
                                 + 0.803528 w_k1_MSFT_s113
                                 + 0.837066 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.947173 w_k1_ADM_s113
                                 + 0.946191 w_k1_AXP_s113
                                 + 0.810174 w_k1_DIS_s113
                                 + 1.01343 w_k1_FDX_s113
                                 + 0.914568 w_k1_IBM_s113
                                 + 0.936425 w_k1_INTC_s113
                                 + 0.890044 w_k1_KO_s113
                                 + 1.05545 w_k1_MAT_s113
                                 + 0.880252 w_k1_MSFT_s113
                                 + 0.971448 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.784265 w_k1_ADM_s113
                                 + 1.06364 w_k1_AXP_s113
                                 + 0.724872 w_k1_DIS_s113
                                 + 0.819712 w_k1_FDX_s113
                                 + 1.13781 w_k1_IBM_s113
                                 + 1.13678 w_k1_INTC_s113
                                 + 0.853784 w_k1_KO_s113
                                 + 0.817321 w_k1_MAT_s113
                                 + 0.8755 w_k1_MSFT_s113
                                 + 0.872669 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.04203 w_k1_ADM_s113
                                 + 0.93378 w_k1_AXP_s113
                                 + 0.873445 w_k1_DIS_s113
                                 + 0.936331 w_k1_FDX_s113
                                 + 1.43014 w_k1_IBM_s113
                                 + 1.14614 w_k1_INTC_s113
                                 + 1.07912 w_k1_KO_s113 + 1.23366 w_k1_MAT_s113
                                 + 0.983462 w_k1_MSFT_s113
                                 + 0.974705 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.00181 w_k1_ADM_s113
                                 + 1.02161 w_k1_AXP_s113
                                 + 1.18379 w_k1_DIS_s113
                                 + 1.18655 w_k1_FDX_s113
                                 + 0.900192 w_k1_IBM_s113
                                 + 1.00996 w_k1_INTC_s113
                                 + 0.975249 w_k1_KO_s113
                                 + 0.782445 w_k1_MAT_s113
                                 + 1.09722 w_k1_MSFT_s113
                                 + 1.03492 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.27499 w_k1_ADM_s113
                                 + 1.27784 w_k1_AXP_s113
                                 + 1.37127 w_k1_DIS_s113
                                 + 1.16943 w_k1_FDX_s113
                                 + 1.44527 w_k1_IBM_s113
                                 + 1.14491 w_k1_INTC_s113 + 1.2207 w_k1_KO_s113
                                 + 1.35733 w_k1_MAT_s113
                                 + 0.970385 w_k1_MSFT_s113
                                 + 1.13229 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.32415 w_k1_ADM_s113
                                 + 1.59598 w_k1_AXP_s113
                                 + 1.71295 w_k1_DIS_s113
                                 + 1.59625 w_k1_FDX_s113
                                 + 1.33567 w_k1_IBM_s113
                                 + 1.29688 w_k1_INTC_s113
                                 + 1.42629 w_k1_KO_s113 + 1.17804 w_k1_MAT_s113
                                 + 1.46983 w_k1_MSFT_s113
                                 + 1.42416 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.15789 w_k1_ADM_s113
                                 + 1.49257 w_k1_AXP_s113
                                 + 1.15139 w_k1_DIS_s113 + 1.2788 w_k1_FDX_s113
                                 + 1.23736 w_k1_IBM_s113
                                 + 1.02419 w_k1_INTC_s113
                                 + 1.25881 w_k1_KO_s113 + 1.34311 w_k1_MAT_s113
                                 + 1.10119 w_k1_MSFT_s113
                                 + 0.96812 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.55409 w_k1_ADM_s114
                                 + 1.80583 w_k1_AXP_s114
                                 + 1.20042 w_k1_DIS_s114
                                 + 1.33205 w_k1_FDX_s114
                                 + 1.03928 w_k1_IBM_s114
                                 + 1.33962 w_k1_INTC_s114
                                 + 1.15891 w_k1_KO_s114 + 1.05308 w_k1_MAT_s114
                                 + 1.15269 w_k1_MSFT_s114
                                 + 1.23073 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.11822 w_k1_ADM_s114
                                 + 1.16079 w_k1_AXP_s114
                                 + 1.04746 w_k1_DIS_s114
                                 + 0.995254 w_k1_FDX_s114
                                 + 0.969406 w_k1_IBM_s114
                                 + 0.82392 w_k1_INTC_s114
                                 + 1.00719 w_k1_KO_s114 + 1.23641 w_k1_MAT_s114
                                 + 1.03072 w_k1_MSFT_s114
                                 + 0.897502 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.953878 w_k1_ADM_s114
                                 + 0.711353 w_k1_AXP_s114
                                 + 0.715748 w_k1_DIS_s114
                                 + 0.759759 w_k1_FDX_s114
                                 + 0.925877 w_k1_IBM_s114
                                 + 0.618883 w_k1_INTC_s114
                                 + 0.833575 w_k1_KO_s114
                                 + 0.752626 w_k1_MAT_s114
                                 + 0.796121 w_k1_MSFT_s114
                                 + 0.738569 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.44353 w_k1_ADM_s114
                                 + 1.17236 w_k1_AXP_s114
                                 + 1.64341 w_k1_DIS_s114
                                 + 1.15052 w_k1_FDX_s114
                                 + 1.28067 w_k1_IBM_s114
                                 + 1.07293 w_k1_INTC_s114 + 1.4983 w_k1_KO_s114
                                 + 1.15388 w_k1_MAT_s114
                                 + 1.21459 w_k1_MSFT_s114
                                 + 1.07899 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.37884 w_k1_ADM_s114
                                 + 1.04432 w_k1_AXP_s114
                                 + 1.18176 w_k1_DIS_s114
                                 + 1.12023 w_k1_FDX_s114
                                 + 0.930028 w_k1_IBM_s114
                                 + 1.0186 w_k1_INTC_s114 + 1.1432 w_k1_KO_s114
                                 + 0.979477 w_k1_MAT_s114
                                 + 1.27684 w_k1_MSFT_s114
                                 + 0.953522 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.09851 w_k1_ADM_s114
                                 + 0.707138 w_k1_AXP_s114
                                 + 0.640058 w_k1_DIS_s114
                                 + 0.785753 w_k1_FDX_s114
                                 + 0.692586 w_k1_IBM_s114
                                 + 0.786561 w_k1_INTC_s114
                                 + 0.677513 w_k1_KO_s114
                                 + 0.464268 w_k1_MAT_s114
                                 + 0.742994 w_k1_MSFT_s114
                                 + 0.675572 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.39193 w_k1_ADM_s114
                                 + 0.915471 w_k1_AXP_s114 + 0.922 w_k1_DIS_s114
                                 + 0.883446 w_k1_FDX_s114
                                 + 1.13536 w_k1_IBM_s114
                                 + 0.732357 w_k1_INTC_s114
                                 + 1.10143 w_k1_KO_s114 + 1.00116 w_k1_MAT_s114
                                 + 1.00062 w_k1_MSFT_s114
                                 + 1.03475 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.27038 w_k1_ADM_s114
                                 + 0.906203 w_k1_AXP_s114
                                 + 0.777035 w_k1_DIS_s114
                                 + 0.950476 w_k1_FDX_s114
                                 + 0.768575 w_k1_IBM_s114
                                 + 1.07272 w_k1_INTC_s114
                                 + 0.799673 w_k1_KO_s114
                                 + 0.470372 w_k1_MAT_s114
                                 + 0.924722 w_k1_MSFT_s114
                                 + 0.87014 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.982523 w_k1_ADM_s115
                                 + 1.16847 w_k1_AXP_s115
                                 + 1.53364 w_k1_DIS_s115
                                 + 0.988652 w_k1_FDX_s115
                                 + 1.44737 w_k1_IBM_s115
                                 + 1.10107 w_k1_INTC_s115
                                 + 1.39814 w_k1_KO_s115 + 1.26031 w_k1_MAT_s115
                                 + 1.02176 w_k1_MSFT_s115
                                 + 1.03925 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.57333 w_k1_ADM_s115
                                 + 0.627854 w_k1_AXP_s115
                                 + 0.551756 w_k1_DIS_s115
                                 + 0.52455 w_k1_FDX_s115
                                 + 1.04244 w_k1_IBM_s115
                                 + 0.564804 w_k1_INTC_s115
                                 + 0.772267 w_k1_KO_s115
                                 + 0.908675 w_k1_MAT_s115
                                 + 0.578893 w_k1_MSFT_s115
                                 + 0.706755 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.96854 w_k1_ADM_s115
                                 + 0.872896 w_k1_AXP_s115
                                 + 0.754341 w_k1_DIS_s115
                                 + 0.973738 w_k1_FDX_s115
                                 + 0.686659 w_k1_IBM_s115
                                 + 1.21809 w_k1_INTC_s115
                                 + 0.784001 w_k1_KO_s115
                                 + 0.464025 w_k1_MAT_s115
                                 + 0.998494 w_k1_MSFT_s115
                                 + 0.977672 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.616476 w_k1_ADM_s115
                                 + 0.868413 w_k1_AXP_s115
                                 + 0.820453 w_k1_DIS_s115
                                 + 0.839063 w_k1_FDX_s115
                                 + 0.980859 w_k1_IBM_s115
                                 + 0.808114 w_k1_INTC_s115
                                 + 0.825891 w_k1_KO_s115
                                 + 0.827941 w_k1_MAT_s115
                                 + 0.704567 w_k1_MSFT_s115
                                 + 0.945414 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.966134 w_k1_ADM_s115
                                 + 1.15405 w_k1_AXP_s115
                                 + 1.49316 w_k1_DIS_s115
                                 + 1.20337 w_k1_FDX_s115
                                 + 1.14801 w_k1_IBM_s115
                                 + 1.20415 w_k1_INTC_s115
                                 + 1.18628 w_k1_KO_s115 + 1.16959 w_k1_MAT_s115
                                 + 1.27791 w_k1_MSFT_s115
                                 + 1.01317 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.55867 w_k1_ADM_s115
                                 + 1.77383 w_k1_AXP_s115
                                 + 1.94463 w_k1_DIS_s115
                                 + 1.44238 w_k1_FDX_s115
                                 + 1.80415 w_k1_IBM_s115
                                 + 1.06624 w_k1_INTC_s115
                                 + 1.41006 w_k1_KO_s115 + 1.568 w_k1_MAT_s115
                                 + 1.11604 w_k1_MSFT_s115
                                 + 1.26048 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.950356 w_k1_ADM_s115
                                 + 0.988936 w_k1_AXP_s115
                                 + 1.03674 w_k1_DIS_s115
                                 + 0.996776 w_k1_FDX_s115
                                 + 0.959247 w_k1_IBM_s115
                                 + 1.23837 w_k1_INTC_s115
                                 + 0.986013 w_k1_KO_s115
                                 + 0.702424 w_k1_MAT_s115
                                 + 1.01547 w_k1_MSFT_s115
                                 + 0.90774 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.03312 w_k1_ADM_s115
                                 + 0.681271 w_k1_AXP_s115
                                 + 0.637461 w_k1_DIS_s115
                                 + 0.753683 w_k1_FDX_s115
                                 + 0.770726 w_k1_IBM_s115
                                 + 0.625413 w_k1_INTC_s115
                                 + 0.695686 w_k1_KO_s115
                                 + 0.559205 w_k1_MAT_s115
                                 + 0.650524 w_k1_MSFT_s115
                                 + 0.738034 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.42035 w_k1_ADM_s116
                                 + 1.42182 w_k1_AXP_s116
                                 + 1.36413 w_k1_DIS_s116
                                 + 1.37056 w_k1_FDX_s116
                                 + 0.800148 w_k1_IBM_s116
                                 + 1.161 w_k1_INTC_s116 + 1.08796 w_k1_KO_s116
                                 + 0.948463 w_k1_MAT_s116
                                 + 1.11043 w_k1_MSFT_s116
                                 + 1.12468 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.08563 w_k1_ADM_s116
                                 + 1.14316 w_k1_AXP_s116
                                 + 1.13973 w_k1_DIS_s116
                                 + 1.13722 w_k1_FDX_s116
                                 + 1.21758 w_k1_IBM_s116
                                 + 0.978542 w_k1_INTC_s116
                                 + 1.22768 w_k1_KO_s116 + 1.11943 w_k1_MAT_s116
                                 + 1.03831 w_k1_MSFT_s116
                                 + 1.10633 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.50341 w_k1_ADM_s116
                                 + 1.68016 w_k1_AXP_s116
                                 + 1.52615 w_k1_DIS_s116
                                 + 1.43075 w_k1_FDX_s116
                                 + 1.19579 w_k1_IBM_s116
                                 + 1.01921 w_k1_INTC_s116
                                 + 1.38991 w_k1_KO_s116 + 1.29574 w_k1_MAT_s116
                                 + 1.02606 w_k1_MSFT_s116
                                 + 1.22401 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.837174 w_k1_ADM_s116
                                 + 0.805226 w_k1_AXP_s116
                                 + 0.710743 w_k1_DIS_s116
                                 + 0.786378 w_k1_FDX_s116
                                 + 0.799566 w_k1_IBM_s116
                                 + 1.04034 w_k1_INTC_s116
                                 + 1.00544 w_k1_KO_s116
                                 + 0.706001 w_k1_MAT_s116
                                 + 0.892323 w_k1_MSFT_s116
                                 + 0.902857 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.16132 w_k1_ADM_s116
                                 + 1.11315 w_k1_AXP_s116
                                 + 1.05549 w_k1_DIS_s116
                                 + 1.04853 w_k1_FDX_s116
                                 + 0.902181 w_k1_IBM_s116
                                 + 1.12764 w_k1_INTC_s116
                                 + 1.00985 w_k1_KO_s116
                                 + 0.704917 w_k1_MAT_s116
                                 + 0.903251 w_k1_MSFT_s116
                                 + 0.97489 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.01724 w_k1_ADM_s116
                                 + 1.4567 w_k1_AXP_s116 + 1.28008 w_k1_DIS_s116
                                 + 1.21336 w_k1_FDX_s116
                                 + 1.59611 w_k1_IBM_s116
                                 + 1.17882 w_k1_INTC_s116
                                 + 1.34603 w_k1_KO_s116 + 2.09229 w_k1_MAT_s116
                                 + 0.929879 w_k1_MSFT_s116
                                 + 1.03807 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.03986 w_k1_ADM_s116
                                 + 0.97614 w_k1_AXP_s116
                                 + 0.925783 w_k1_DIS_s116
                                 + 0.909294 w_k1_FDX_s116
                                 + 1.35145 w_k1_IBM_s116
                                 + 0.993387 w_k1_INTC_s116
                                 + 1.12436 w_k1_KO_s116 + 1.07663 w_k1_MAT_s116
                                 + 0.781443 w_k1_MSFT_s116
                                 + 0.905734 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.950927 w_k1_ADM_s116
                                 + 0.907374 w_k1_AXP_s116
                                 + 0.898522 w_k1_DIS_s116
                                 + 0.911552 w_k1_FDX_s116
                                 + 1.10205 w_k1_IBM_s116
                                 + 0.790605 w_k1_INTC_s116
                                 + 0.994767 w_k1_KO_s116
                                 + 1.07067 w_k1_MAT_s116
                                 + 0.701708 w_k1_MSFT_s116
                                 + 0.986386 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.873801 w_k1_ADM_s117
                                 + 0.734929 w_k1_AXP_s117
                                 + 0.99095 w_k1_DIS_s117
                                 + 0.86857 w_k1_FDX_s117
                                 + 1.38901 w_k1_IBM_s117
                                 + 0.922131 w_k1_INTC_s117
                                 + 1.08973 w_k1_KO_s117 + 1.42671 w_k1_MAT_s117
                                 + 1.01172 w_k1_MSFT_s117
                                 + 0.982671 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.10347 w_k1_ADM_s117
                                 + 1.22153 w_k1_AXP_s117
                                 + 1.24833 w_k1_DIS_s117
                                 + 1.12996 w_k1_FDX_s117
                                 + 1.26627 w_k1_IBM_s117
                                 + 1.08972 w_k1_INTC_s117
                                 + 1.10618 w_k1_KO_s117 + 1.61641 w_k1_MAT_s117
                                 + 1.23006 w_k1_MSFT_s117 + 1.1129 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.83212 w_k1_ADM_s117
                                 + 0.987451 w_k1_AXP_s117
                                 + 0.938367 w_k1_DIS_s117
                                 + 0.867907 w_k1_FDX_s117
                                 + 1.28716 w_k1_IBM_s117
                                 + 0.999169 w_k1_INTC_s117
                                 + 1.04129 w_k1_KO_s117 + 1.45968 w_k1_MAT_s117
                                 + 0.909496 w_k1_MSFT_s117
                                 + 0.98843 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.93309 w_k1_ADM_s117
                                 + 1.05762 w_k1_AXP_s117
                                 + 1.03821 w_k1_DIS_s117 + 1.1367 w_k1_FDX_s117
                                 + 0.688796 w_k1_IBM_s117
                                 + 1.07592 w_k1_INTC_s117
                                 + 0.818725 w_k1_KO_s117
                                 + 0.754028 w_k1_MAT_s117
                                 + 1.04577 w_k1_MSFT_s117
                                 + 0.970124 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.527 w_k1_ADM_s117
                                 + 1.79584 w_k1_AXP_s117
                                 + 1.54356 w_k1_DIS_s117
                                 + 1.47469 w_k1_FDX_s117
                                 + 1.14318 w_k1_IBM_s117
                                 + 1.59241 w_k1_INTC_s117
                                 + 1.30977 w_k1_KO_s117 + 1.5488 w_k1_MAT_s117
                                 + 1.32219 w_k1_MSFT_s117
                                 + 1.43921 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.787817 w_k1_ADM_s117
                                 + 1.0046 w_k1_AXP_s117
                                 + 0.828523 w_k1_DIS_s117
                                 + 0.953915 w_k1_FDX_s117
                                 + 0.925383 w_k1_IBM_s117
                                 + 1.09551 w_k1_INTC_s117
                                 + 0.805895 w_k1_KO_s117
                                 + 1.05162 w_k1_MAT_s117
                                 + 0.961398 w_k1_MSFT_s117
                                 + 0.906498 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.0039 w_k1_ADM_s117
                                 + 0.972636 w_k1_AXP_s117
                                 + 1.25481 w_k1_DIS_s117
                                 + 1.06945 w_k1_FDX_s117
                                 + 0.923377 w_k1_IBM_s117
                                 + 1.01848 w_k1_INTC_s117
                                 + 0.946258 w_k1_KO_s117
                                 + 1.04706 w_k1_MAT_s117
                                 + 1.13002 w_k1_MSFT_s117
                                 + 1.08098 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.667248 w_k1_ADM_s117
                                 + 0.707332 w_k1_AXP_s117
                                 + 0.706027 w_k1_DIS_s117
                                 + 0.735097 w_k1_FDX_s117
                                 + 0.750726 w_k1_IBM_s117
                                 + 0.898437 w_k1_INTC_s117
                                 + 0.835582 w_k1_KO_s117
                                 + 0.888048 w_k1_MAT_s117
                                 + 0.830855 w_k1_MSFT_s117
                                 + 0.856337 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0149968385926531 w_k1_ADM_s000
                                 + 0.010742153794242 w_k1_ADM_s001
                                 + 0.0148836402516779 w_k1_ADM_s002
                                 + 0.0159908749013004 w_k1_ADM_s003
                                 + 0.00758079789311272 w_k1_ADM_s004
                                 + 0.0170986043012559 w_k1_ADM_s005
                                 + 0.00965993783004335 w_k1_ADM_s006
                                 + 0.00996461406585987 w_k1_ADM_s007
                                 + 0.0315466833770109 w_k1_ADM_s010
                                 + 0.021779536234563 w_k1_ADM_s011
                                 + 0.023797716825855 w_k1_ADM_s012
                                 + 0.0850607104266255 w_k1_ADM_s013
                                 + 0.0810715574994684 w_k1_ADM_s014
                                 + 0.0731540013437948 w_k1_ADM_s015
                                 + 0.0228271814398816 w_k1_ADM_s016
                                 + 0.0247182814037378 w_k1_ADM_s017
                                 + 0.0714745884705041 w_k1_ADM_s100
                                 + 0.0555701211323171 w_k1_ADM_s101
                                 + 0.0384408037830797 w_k1_ADM_s102
                                 + 0.0838361132208007 w_k1_ADM_s103
                                 + 0.0261409820733288 w_k1_ADM_s104
                                 + 0.0678563572348299 w_k1_ADM_s105
                                 + 0.0647506699225953 w_k1_ADM_s106
                                 + 0.0551357955352738 w_k1_ADM_s107
                                 + 0.0311311536343727 w_k1_ADM_s110
                                 + 0.00942715137609783 w_k1_ADM_s111
                                 + 0.0070131505251599 w_k1_ADM_s112
                                 + 0.0074648320039864 w_k1_ADM_s113
                                 + 0.0105711065658891 w_k1_ADM_s114
                                 + 0.00722552096518496 w_k1_ADM_s115
                                 + 0.0134998021537744 w_k1_ADM_s116
                                 + 0.0181301416492048 w_k1_ADM_s117
                                 + 0.0149613527856984 w_k1_AXP_s000
                                 + 0.0102602131465636 w_k1_AXP_s001
                                 + 0.0158487100148053 w_k1_AXP_s002
                                 + 0.0147173879440232 w_k1_AXP_s003
                                 + 0.00747649814696353 w_k1_AXP_s004
                                 + 0.0176831785416339 w_k1_AXP_s005
                                 + 0.00873252191268528 w_k1_AXP_s006
                                 + 0.0111709984597392 w_k1_AXP_s007
                                 + 0.0305456933033239 w_k1_AXP_s010
                                 + 0.0235639939010802 w_k1_AXP_s011
                                 + 0.0207435553248753 w_k1_AXP_s012
                                 + 0.0818671846958392 w_k1_AXP_s013
                                 + 0.0775182158999737 w_k1_AXP_s014
                                 + 0.0710864802007108 w_k1_AXP_s015
                                 + 0.0219008025793681 w_k1_AXP_s016
                                 + 0.0239993595204164 w_k1_AXP_s017
                                 + 0.0723843570855774 w_k1_AXP_s100
                                 + 0.0512818700585048 w_k1_AXP_s101
                                 + 0.0401313648191031 w_k1_AXP_s102
                                 + 0.0997498326379206 w_k1_AXP_s103
                                 + 0.0221548496118129 w_k1_AXP_s104
                                 + 0.063935310479598 w_k1_AXP_s105
                                 + 0.0607246444365152 w_k1_AXP_s106
                                 + 0.0602044065201044 w_k1_AXP_s107
                                 + 0.0277060959448155 w_k1_AXP_s110
                                 + 0.00935395615152903 w_k1_AXP_s111
                                 + 0.0062858486851689 w_k1_AXP_s112
                                 + 0.00825422606588873 w_k1_AXP_s113
                                 + 0.00851075589535698 w_k1_AXP_s114
                                 + 0.00788060699036891 w_k1_AXP_s115
                                 + 0.0137807766270422 w_k1_AXP_s116
                                 + 0.0199361506602183 w_k1_AXP_s117
                                 + 0.0162591619859388 w_k1_DIS_s000
                                 + 0.0109385679602545 w_k1_DIS_s001
                                 + 0.0170546568293861 w_k1_DIS_s002
                                 + 0.0148503074222804 w_k1_DIS_s003
                                 + 0.00726551311337058 w_k1_DIS_s004
                                 + 0.0162718110378823 w_k1_DIS_s005
                                 + 0.00924574802253493 w_k1_DIS_s006
                                 + 0.0109661552405328 w_k1_DIS_s007
                                 + 0.0300979780807804 w_k1_DIS_s010
                                 + 0.0228922413293449 w_k1_DIS_s011
                                 + 0.0219704116779513 w_k1_DIS_s012
                                 + 0.0783787342353308 w_k1_DIS_s013
                                 + 0.0795995443421112 w_k1_DIS_s014
                                 + 0.0705676561750402 w_k1_DIS_s015
                                 + 0.0245975596080972 w_k1_DIS_s016
                                 + 0.026438976946811 w_k1_DIS_s017
                                 + 0.0733219977803392 w_k1_DIS_s100
                                 + 0.0536074735235974 w_k1_DIS_s101
                                 + 0.0424619745741288 w_k1_DIS_s102
                                 + 0.0876862994262105 w_k1_DIS_s103
                                 + 0.0265976851404821 w_k1_DIS_s104
                                 + 0.0684469149961594 w_k1_DIS_s105
                                 + 0.0624028961915816 w_k1_DIS_s106
                                 + 0.0611821857807695 w_k1_DIS_s107
                                 + 0.0268267205967043 w_k1_DIS_s110
                                 + 0.00981528772267591 w_k1_DIS_s111
                                 + 0.00685701057942716 w_k1_DIS_s112
                                 + 0.00729519022064634 w_k1_DIS_s113
                                 + 0.00889700269113914 w_k1_DIS_s114
                                 + 0.00867942860572134 w_k1_DIS_s115
                                 + 0.0128927457207359 w_k1_DIS_s116
                                 + 0.0203801279276175 w_k1_DIS_s117
                                 + 0.0167197876050637 w_k1_FDX_s000
                                 + 0.0100913140949255 w_k1_FDX_s001
                                 + 0.016113486078104 w_k1_FDX_s002
                                 + 0.0138718236283869 w_k1_FDX_s003
                                 + 0.00699583980709664 w_k1_FDX_s004
                                 + 0.0168230346613427 w_k1_FDX_s005
                                 + 0.00952867763961562 w_k1_FDX_s006
                                 + 0.011421328316599 w_k1_FDX_s007
                                 + 0.0319539684171155 w_k1_FDX_s010
                                 + 0.0225052668824181 w_k1_FDX_s011
                                 + 0.0213973981924909 w_k1_FDX_s012
                                 + 0.0812282094626012 w_k1_FDX_s013
                                 + 0.0725536017874453 w_k1_FDX_s014
                                 + 0.0719352333367214 w_k1_FDX_s015
                                 + 0.0249490380104741 w_k1_FDX_s016
                                 + 0.0246486326472651 w_k1_FDX_s017
                                 + 0.0696977122147764 w_k1_FDX_s100
                                 + 0.0522372914777436 w_k1_FDX_s101
                                 + 0.0393213842073774 w_k1_FDX_s102
                                 + 0.0889531493183162 w_k1_FDX_s103
                                 + 0.0254081939909597 w_k1_FDX_s104
                                 + 0.0689369047320427 w_k1_FDX_s105
                                 + 0.0578011794818585 w_k1_FDX_s106
                                 + 0.0626473428429736 w_k1_FDX_s107
                                 + 0.0265012749935701 w_k1_FDX_s110
                                 + 0.00912308723501022 w_k1_FDX_s111
                                 + 0.00735811486052492 w_k1_FDX_s112
                                 + 0.00802886938252043 w_k1_FDX_s113
                                 + 0.00823559950693556 w_k1_FDX_s114
                                 + 0.00768383894745858 w_k1_FDX_s115
                                 + 0.0128686839413885 w_k1_FDX_s116
                                 + 0.0193847374536746 w_k1_FDX_s117
                                 + 0.015453478702617 w_k1_IBM_s000
                                 + 0.00981165712355728 w_k1_IBM_s001
                                 + 0.0153934154744058 w_k1_IBM_s002
                                 + 0.0144846678844463 w_k1_IBM_s003
                                 + 0.00720996339367918 w_k1_IBM_s004
                                 + 0.0167684651457485 w_k1_IBM_s005
                                 + 0.0100330213227482 w_k1_IBM_s006
                                 + 0.0121624912241919 w_k1_IBM_s007
                                 + 0.0304236998492191 w_k1_IBM_s010
                                 + 0.0228748688627673 w_k1_IBM_s011
                                 + 0.0214436780614958 w_k1_IBM_s012
                                 + 0.0840314816209835 w_k1_IBM_s013
                                 + 0.077331434403954 w_k1_IBM_s014
                                 + 0.0745308803606048 w_k1_IBM_s015
                                 + 0.0248578571275439 w_k1_IBM_s016
                                 + 0.0247247971899206 w_k1_IBM_s017
                                 + 0.0700124227699324 w_k1_IBM_s100
                                 + 0.0541866896803504 w_k1_IBM_s101
                                 + 0.0405877311045666 w_k1_IBM_s102
                                 + 0.0899033149034758 w_k1_IBM_s103
                                 + 0.0250714167781988 w_k1_IBM_s104
                                 + 0.0659388601059059 w_k1_IBM_s105
                                 + 0.0589842451242192 w_k1_IBM_s106
                                 + 0.0609826674093788 w_k1_IBM_s107
                                 + 0.0283803965304025 w_k1_IBM_s110
                                 + 0.00971207007064175 w_k1_IBM_s111
                                 + 0.00710373582128553 w_k1_IBM_s112
                                 + 0.00808290042589613 w_k1_IBM_s113
                                 + 0.00830737484641217 w_k1_IBM_s114
                                 + 0.00836837740859924 w_k1_IBM_s115
                                 + 0.0128732083389203 w_k1_IBM_s116
                                 + 0.020207462203836 w_k1_IBM_s117
                                 + 0.0161553510673655 w_k1_INTC_s000
                                 + 0.0119554366224951 w_k1_INTC_s001
                                 + 0.0181880591867563 w_k1_INTC_s002
                                 + 0.0136648878293016 w_k1_INTC_s003
                                 + 0.00615365257918012 w_k1_INTC_s004
                                 + 0.0164830495453388 w_k1_INTC_s005
                                 + 0.00939152455878155 w_k1_INTC_s006
                                 + 0.0110797437779235 w_k1_INTC_s007
                                 + 0.0325915211987496 w_k1_INTC_s010
                                 + 0.0251527888311214 w_k1_INTC_s011
                                 + 0.0189085353224438 w_k1_INTC_s012
                                 + 0.0748780847784052 w_k1_INTC_s013
                                 + 0.0678797816892605 w_k1_INTC_s014
                                 + 0.0713874757583362 w_k1_INTC_s015
                                 + 0.0275573621142983 w_k1_INTC_s016
                                 + 0.0213035937004528 w_k1_INTC_s017
                                 + 0.0661908129370582 w_k1_INTC_s100
                                 + 0.0509160705867874 w_k1_INTC_s101
                                 + 0.0406582929139522 w_k1_INTC_s102
                                 + 0.0957198398771855 w_k1_INTC_s103
                                 + 0.0250523428429909 w_k1_INTC_s104
                                 + 0.0690776619256792 w_k1_INTC_s105
                                 + 0.0618248332215728 w_k1_INTC_s106
                                 + 0.0623637197400168 w_k1_INTC_s107
                                 + 0.0223952754687146 w_k1_INTC_s110
                                 + 0.00871296119741677 w_k1_INTC_s111
                                 + 0.00606309444250948 w_k1_INTC_s112
                                 + 0.00793324482487081 w_k1_INTC_s113
                                 + 0.00762430426877986 w_k1_INTC_s114
                                 + 0.0079592313647136 w_k1_INTC_s115
                                 + 0.0127283178335191 w_k1_INTC_s116
                                 + 0.0202073627811295 w_k1_INTC_s117
                                 + 0.0153042182742433 w_k1_KO_s000
                                 + 0.00998308329245435 w_k1_KO_s001
                                 + 0.0165924083414557 w_k1_KO_s002
                                 + 0.0132519185654977 w_k1_KO_s003
                                 + 0.00708289012168275 w_k1_KO_s004
                                 + 0.0148216644283339 w_k1_KO_s005
                                 + 0.00853118029064234 w_k1_KO_s006
                                 + 0.0107533656927168 w_k1_KO_s007
                                 + 0.0299058765392325 w_k1_KO_s010
                                 + 0.0234580981946868 w_k1_KO_s011
                                 + 0.0223811428687574 w_k1_KO_s012
                                 + 0.0797956037296861 w_k1_KO_s013
                                 + 0.0860578121568308 w_k1_KO_s014
                                 + 0.0703299866432599 w_k1_KO_s015
                                 + 0.0254199955501157 w_k1_KO_s016
                                 + 0.0225433185868112 w_k1_KO_s017
                                 + 0.0682259402029158 w_k1_KO_s100
                                 + 0.0625629935531355 w_k1_KO_s101
                                 + 0.0455804248935346 w_k1_KO_s102
                                 + 0.0891454459043667 w_k1_KO_s103
                                 + 0.0248417336323415 w_k1_KO_s104
                                 + 0.0649243656747969 w_k1_KO_s105
                                 + 0.0573860428126483 w_k1_KO_s106
                                 + 0.0591145215136452 w_k1_KO_s107
                                 + 0.0250998035981457 w_k1_KO_s110
                                 + 0.00868622200077836 w_k1_KO_s111
                                 + 0.0072901968988724 w_k1_KO_s112
                                 + 0.00762167565758923 w_k1_KO_s113
                                 + 0.00889512256218069 w_k1_KO_s114
                                 + 0.0078713414268163 w_k1_KO_s115
                                 + 0.0135536787769929 w_k1_KO_s116
                                 + 0.0188525194996721 w_k1_KO_s117
                                 + 0.0118627373778794 w_k1_MAT_s000
                                 + 0.00921259963399425 w_k1_MAT_s001
                                 + 0.0197478053350685 w_k1_MAT_s002
                                 + 0.0133651211811612 w_k1_MAT_s003
                                 + 0.00792223389574342 w_k1_MAT_s004
                                 + 0.0190070021627574 w_k1_MAT_s005
                                 + 0.0102209980120944 w_k1_MAT_s006
                                 + 0.0129096828683036 w_k1_MAT_s007
                                 + 0.0295354050978955 w_k1_MAT_s010
                                 + 0.0289752201914911 w_k1_MAT_s011
                                 + 0.0179554650341952 w_k1_MAT_s012
                                 + 0.0628819352987485 w_k1_MAT_s013
                                 + 0.0748684545868461 w_k1_MAT_s014
                                 + 0.0708930036502269 w_k1_MAT_s015
                                 + 0.0258139691616825 w_k1_MAT_s016
                                 + 0.0252983000595309 w_k1_MAT_s017
                                 + 0.0840691609071659 w_k1_MAT_s100
                                 + 0.0586035587708026 w_k1_MAT_s101
                                 + 0.0446662521055208 w_k1_MAT_s102
                                 + 0.117256279752213 w_k1_MAT_s103
                                 + 0.0248286010043111 w_k1_MAT_s104
                                 + 0.0663521428706871 w_k1_MAT_s105
                                 + 0.052152848733902 w_k1_MAT_s106
                                 + 0.0839117378942495 w_k1_MAT_s107
                                 + 0.0257110642764877 w_k1_MAT_s110
                                 + 0.011588377094956 w_k1_MAT_s111
                                 + 0.00887637507394068 w_k1_MAT_s112
                                 + 0.00775611417110099 w_k1_MAT_s113
                                 + 0.0076527274080839 w_k1_MAT_s114
                                 + 0.0072203919156816 w_k1_MAT_s115
                                 + 0.01222765697851 w_k1_MAT_s116
                                 + 0.0239248314306112 w_k1_MAT_s117
                                 + 0.0152891054010284 w_k1_MSFT_s000
                                 + 0.0105387999600919 w_k1_MSFT_s001
                                 + 0.0177730167692367 w_k1_MSFT_s002
                                 + 0.0148934088707787 w_k1_MSFT_s003
                                 + 0.0065172932465468 w_k1_MSFT_s004
                                 + 0.016159648160986 w_k1_MSFT_s005
                                 + 0.00876014357378861 w_k1_MSFT_s006
                                 + 0.0117279004342018 w_k1_MSFT_s007
                                 + 0.0287259804070726 w_k1_MSFT_s010
                                 + 0.0227338800923873 w_k1_MSFT_s011
                                 + 0.0199963236215399 w_k1_MSFT_s012
                                 + 0.0840445874100997 w_k1_MSFT_s013
                                 + 0.071910778627033 w_k1_MSFT_s014
                                 + 0.0687569945858969 w_k1_MSFT_s015
                                 + 0.0250662489440852 w_k1_MSFT_s016
                                 + 0.0251496629672687 w_k1_MSFT_s017
                                 + 0.0775067905032798 w_k1_MSFT_s100
                                 + 0.0548445119719301 w_k1_MSFT_s101
                                 + 0.0435856697753112 w_k1_MSFT_s102
                                 + 0.0879955308965736 w_k1_MSFT_s103
                                 + 0.0266348411233009 w_k1_MSFT_s104
                                 + 0.0640375803569139 w_k1_MSFT_s105
                                 + 0.0588961855658875 w_k1_MSFT_s106
                                 + 0.0639778179960785 w_k1_MSFT_s107
                                 + 0.0249312396294919 w_k1_MSFT_s110
                                 + 0.00976662121505403 w_k1_MSFT_s111
                                 + 0.00684406513874418 w_k1_MSFT_s112
                                 + 0.00741560210847018 w_k1_MSFT_s113
                                 + 0.00856287526576486 w_k1_MSFT_s114
                                 + 0.00751709095179932 w_k1_MSFT_s115
                                 + 0.0110721562972444 w_k1_MSFT_s116
                                 + 0.0202788132854435 w_k1_MSFT_s117
                                 + 0.0153989419683469 w_k1_PG_s000
                                 + 0.0106791350303534 w_k1_PG_s001
                                 + 0.016591704044479 w_k1_PG_s002
                                 + 0.0138761273256326 w_k1_PG_s003
                                 + 0.0068068998554491 w_k1_PG_s004
                                 + 0.0166039010797571 w_k1_PG_s005
                                 + 0.00984077251026952 w_k1_PG_s006
                                 + 0.0105986731346891 w_k1_PG_s007
                                 + 0.0278176485419655 w_k1_PG_s010
                                 + 0.0230676233489233 w_k1_PG_s011
                                 + 0.0218786587957719 w_k1_PG_s012
                                 + 0.0814378998837416 w_k1_PG_s013
                                 + 0.0746595285772397 w_k1_PG_s014
                                 + 0.0778959057505683 w_k1_PG_s015
                                 + 0.0231184583272721 w_k1_PG_s016
                                 + 0.022369112248901 w_k1_PG_s017
                                 + 0.0673024303575145 w_k1_PG_s100
                                 + 0.0500638817789115 w_k1_PG_s101
                                 + 0.0440958704038478 w_k1_PG_s102
                                 + 0.0877504765565099 w_k1_PG_s103
                                 + 0.0244982449420515 w_k1_PG_s104
                                 + 0.0649251018571097 w_k1_PG_s105
                                 + 0.0550858476930057 w_k1_PG_s106
                                 + 0.061150036755142 w_k1_PG_s107
                                 + 0.026551524187781 w_k1_PG_s110
                                 + 0.00930795124231826 w_k1_PG_s111
                                 + 0.00711798317734518 w_k1_PG_s112
                                 + 0.00748680073914547 w_k1_PG_s113
                                 + 0.00776853969566088 w_k1_PG_s114
                                 + 0.00741380264855233 w_k1_PG_s115
                                 + 0.0123656872962191 w_k1_PG_s116
                                 + 0.0196845083771507 w_k1_PG_s117 >= 1.1
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
