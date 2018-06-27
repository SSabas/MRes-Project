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
 w_k1_ADM_s0_return_constraint:  1.21072 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.982339 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.14292 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.930779 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.97002 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.889675 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.905002 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.02105 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.31607 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.05571 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.880702 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.18539 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.826402 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.05748 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.998234 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.02034 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.999833 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.12222 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.22516 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.03347 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.93341 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.714378 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.30719 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.832551 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.21111 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.09626 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.47504 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.48945 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.963666 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.739823 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.12911 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.07479 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.52155 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.863951 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.880739 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.10207 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.16191 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.52438 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.35464 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.917121 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.23962 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.91006 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.12783 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.953721 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.949827 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.904291 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.36237 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.08707 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.691253 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.13467 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.73736 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.32781 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.10917 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.855733 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.761768 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.362 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.45926 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.88493 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.8054 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.679031 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.12158 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.677318 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.14931 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.05544 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.62262 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.41827 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.850126 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.840951 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.12791 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.16056 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.33525 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.657218 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.761963 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.763256 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.982098 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.77127 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.39933 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.896533 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.23027 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.953048 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.10467 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.8803 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.18705 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.21835 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.84491 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.03179 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.678494 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.03236 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.797519 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.81023 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.13363 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.82536 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.896151 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.09315 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.60647 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.84938 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.663051 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.544576 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.30725 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.628698 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.34321 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.10531 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.82828 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.66806 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.00076 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.88522 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.04322 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.0349 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.68297 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.717958 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.667156 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.15693 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.03143 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.67821 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.24847 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.939009 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.23399 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.924345 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.00921 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.0283 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.00715 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.15225 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.4282 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.920279 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.806545 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.19941 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.920082 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.32501 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.20209 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.814291 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.883714 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.00198 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.25596 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.913631 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.912147 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.75002 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.29097 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.776066 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.19388 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.958251 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.38846 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.51055 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.02603 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.872819 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.06494 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.908888 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.39535 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.79141 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.703699 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.947132 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.09582 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.38161 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.17212 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.973016 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.906577 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.07887 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.21839 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.677034 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.17973 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.869227 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.14585 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.3344 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.850531 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.839992 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.613226 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.85223 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.783163 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.24322 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.79022 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.722 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.3707 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.01799 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.536812 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.697553 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.801849 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.693979 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.15735 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.81231 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.59584 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.996774 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.779848 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.08184 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.10411 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.4113 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.01717 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.65167 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.922792 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.850703 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.836953 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.64899 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18179 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.918536 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.24529 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.872778 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13925 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.02923 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.03153 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.938332 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.15501 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.921373 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.765936 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.914161 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.859714 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.06472 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.28634 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.05053 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.860016 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.22997 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.36646 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.945414 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.820022 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.770741 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.06611 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.80469 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.0877 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.920635 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.27582 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.20896 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.962803 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.84808 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.13883 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.890348 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.27464 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.705749 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.865548 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.998096 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.974522 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.17006 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21867 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.896886 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.02837 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.03246 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.19993 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.84049 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.11483 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.991949 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.36599 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.15046 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.728547 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.933527 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.783775 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.51842 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.05455 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.02807 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.84312 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.27819 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.35748 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.919579 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.674665 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.625496 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.969791 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.713706 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.10458 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.22883 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.44389 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.27563 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.920751 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.93296 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.01533 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.15293 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.32695 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.716741 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.830707 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.984622 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.941238 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.48377 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.22534 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.944063 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.0688 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.12061 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.37591 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.65531 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.18268 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.785217 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.23191 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.38302 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.577837 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.910881 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.520127 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.54419 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.877286 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.07489 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.68372 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.97668 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.69931 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.04524 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.527839 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.69888 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.851178 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.51034 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.00362 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.39136 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.45167 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.0811 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.743879 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.913426 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.09645 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.68044 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.26049 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.688575 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.7448 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.699601 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.7739 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.78869 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13283 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.945576 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.25611 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.961971 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.04354 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.01768 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.11687 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.08278 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.28496 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.01206 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.787841 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.00834 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.827237 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.28937 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.27039 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.876663 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.03342 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.16005 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.26239 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.905328 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.90381 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.798749 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.27047 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.708381 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.12152 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.874592 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.29701 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.25599 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.04163 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.87898 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.920417 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.959261 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.2334 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.812108 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.738545 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.0434 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.973033 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.18766 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15974 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.924122 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.06154 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.923206 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.10034 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.94511 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.11249 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.04062 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.22059 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.02553 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.90709 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.01258 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.872767 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.24871 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.07273 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.0391 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.953851 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.18767 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.33504 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.932092 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.922935 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.826579 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.03852 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.775281 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.08216 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.954373 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.347 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.23353 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.0225 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.916607 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.08597 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.05649 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.24204 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.835688 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.797877 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.998991 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.06077 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.42145 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.0789998797588102 z_k1_s0000
                                 - 0.061000022208806 z_k1_s0001
                                 - 0.15699994660945 z_k1_s0002
                                 - 0.114999741458952 z_k1_s0003
                                 - 0.0930001924758648 z_k1_s0004
                                 - 0.0640000473338511 z_k1_s0005
                                 - 0.0660000396505256 z_k1_s0006
                                 - 0.0979999900173846 z_k1_s0007
                                 - 0.058000059104988 z_k1_s0010
                                 - 0.0750000017391232 z_k1_s0011
                                 - 0.0690001026801089 z_k1_s0012
                                 - 0.0600000735913811 z_k1_s0013
                                 - 0.100999973461986 z_k1_s0014
                                 - 0.0740000305959679 z_k1_s0015
                                 - 0.202999918067128 z_k1_s0016
                                 - 0.0819997997412103 z_k1_s0017
                                 - 0.102000020095797 z_k1_s0020
                                 - 0.0380000418318094 z_k1_s0021
                                 - 0.0609998462139516 z_k1_s0022
                                 - 0.135000204244654 z_k1_s0023
                                 - 0.0680002147307766 z_k1_s0024
                                 - 0.0360000141985254 z_k1_s0025
                                 - 0.109000388612622 z_k1_s0026
                                 - 0.0550000653144669 z_k1_s0027
                                 - 0.117000328189054 z_k1_s0030
                                 - 0.161000698482812 z_k1_s0031
                                 - 0.133000100658867 z_k1_s0032
                                 - 0.0930001714834478 z_k1_s0033
                                 - 0.0520000573283108 z_k1_s0034
                                 - 0.0470001284314942 z_k1_s0035
                                 - 0.234000656378108 z_k1_s0036
                                 - 0.159000726924086 z_k1_s0037
                                 - 0.161000444251993 z_k1_s0040
                                 - 0.285000843176042 z_k1_s0041
                                 - 0.181000567559257 z_k1_s0042
                                 - 0.34200101181125 z_k1_s0043
                                 - 0.155000447879924 z_k1_s0044
                                 - 0.284001588482704 z_k1_s0045
                                 - 0.136000391668188 z_k1_s0046
                                 - 0.487000585635352 z_k1_s0047
                                 - 0.0470000521509634 z_k1_s0050
                                 - 0.110000252055552 z_k1_s0051
                                 - 0.100000006959412 z_k1_s0052
                                 - 0.0480000155605279 z_k1_s0053
                                 - 0.0310000265974376 z_k1_s0054
                                 - 0.142000180962504 z_k1_s0055
                                 - 0.0340000390262299 z_k1_s0056
                                 - 0.0989997991496501 z_k1_s0057
                                 - 0.0740000401386688 z_k1_s0060
                                 - 0.1490002795875 z_k1_s0061
                                 - 0.0870000951684786 z_k1_s0062
                                 - 0.178000733977378 z_k1_s0063
                                 - 0.244000194035667 z_k1_s0064
                                 - 0.253000466225748 z_k1_s0065
                                 - 0.186999738166305 z_k1_s0066
                                 - 0.0960001137583292 z_k1_s0067
                                 - 0.12900051517729 z_k1_s0070
                                 - 0.058000104746167 z_k1_s0071
                                 - 0.0760004467713558 z_k1_s0072
                                 - 0.0710002784309033 z_k1_s0073
                                 - 0.153000135209481 z_k1_s0074
                                 - 0.0819998567785778 z_k1_s0075
                                 - 0.0550000697420056 z_k1_s0076
                                 - 0.0360000900493868 z_k1_s0077
                                 - 0.525999687388461 z_k1_s0100
                                 - 0.473998668989298 z_k1_s0101
                                 - 0.331999740505292 z_k1_s0102
                                 - 0.200000011975404 z_k1_s0103
                                 - 0.658000438017905 z_k1_s0104
                                 - 0.405000134977644 z_k1_s0105
                                 - 0.516000146734491 z_k1_s0106
                                 - 0.296000031351592 z_k1_s0107
                                 - 0.217000070591251 z_k1_s0110
                                 - 0.193999750528583 z_k1_s0111
                                 - 0.0990000272697412 z_k1_s0112
                                 - 0.0580000051580302 z_k1_s0113
                                 - 0.241999590659366 z_k1_s0114
                                 - 0.130000360354206 z_k1_s0115
                                 - 0.10299997428064 z_k1_s0116
                                 - 0.146999510400523 z_k1_s0117
                                 - 0.098999915999413 z_k1_s0120
                                 - 0.149000347273461 z_k1_s0121
                                 - 0.0759998958611993 z_k1_s0122
                                 - 0.0959999410509269 z_k1_s0123
                                 - 0.1299999754335 z_k1_s0124
                                 - 0.254000000970027 z_k1_s0125
                                 - 0.1299999754335 z_k1_s0126
                                 - 0.126999469985465 z_k1_s0127
                                 - 0.411000217944653 z_k1_s0130
                                 - 0.283999112395977 z_k1_s0131
                                 - 0.0989999544743662 z_k1_s0132
                                 - 0.099999995584163 z_k1_s0133
                                 - 0.242998943092657 z_k1_s0134
                                 - 0.444999882879632 z_k1_s0135
                                 - 0.395998518298882 z_k1_s0136
                                 - 0.187999714450529 z_k1_s0137
                                 - 0.393999328005761 z_k1_s0140
                                 - 0.771001880489469 z_k1_s0141
                                 - 0.683000056898708 z_k1_s0142
                                 - 0.315000240939884 z_k1_s0143
                                 - 0.368000186258945 z_k1_s0144
                                 - 0.452999246174142 z_k1_s0145
                                 - 0.362000213409624 z_k1_s0146
                                 - 0.35700005088513 z_k1_s0147
                                 - 0.207999592753463 z_k1_s0150
                                 - 0.346999303454136 z_k1_s0151
                                 - 0.128999769150894 z_k1_s0152
                                 - 0.47399846601656 z_k1_s0153
                                 - 0.421000107479528 z_k1_s0154
                                 - 0.27999931459153 z_k1_s0155
                                 - 0.392999224822663 z_k1_s0156
                                 - 0.125999691566189 z_k1_s0157
                                 - 0.278999833443866 z_k1_s0160
                                 - 0.499999191929403 z_k1_s0161
                                 - 0.15999965653763 z_k1_s0162
                                 - 0.140999530216722 z_k1_s0163
                                 - 0.298999631361484 z_k1_s0164
                                 - 0.15999965653763 z_k1_s0165
                                 - 0.135999686837041 z_k1_s0166
                                 - 0.446998135951859 z_k1_s0167
                                 - 0.194999600860008 z_k1_s0170
                                 - 0.34899989005233 z_k1_s0171
                                 - 0.407999553709384 z_k1_s0172
                                 - 0.557999048294406 z_k1_s0173
                                 - 0.181999661855938 z_k1_s0174
                                 - 0.135999679303476 z_k1_s0175
                                 - 0.540998314683037 z_k1_s0176
                                 - 0.20999955031851 z_k1_s0177
                                 - 0.507999592969021 z_k1_s1000
                                 - 0.335999776442489 z_k1_s1001
                                 - 0.594998137158977 z_k1_s1002
                                 - 0.459997968106261 z_k1_s1003
                                 - 0.559000025011057 z_k1_s1004
                                 - 0.270999802431893 z_k1_s1005
                                 - 0.533000698126637 z_k1_s1006
                                 - 0.880000795158523 z_k1_s1007
                                 - 2.02499733770826 z_k1_s1010
                                 - 1.80200394100797 z_k1_s1011
                                 - 0.804000960140746 z_k1_s1012
                                 - 1.83600875820269 z_k1_s1013
                                 - 2.68999749787053 z_k1_s1014
                                 - 1.24600213759579 z_k1_s1015
                                 - 2.25000169690384 z_k1_s1016
                                 - 1.31200220599166 z_k1_s1017
                                 - 0.511998951828632 z_k1_s1020
                                 - 0.462999956019919 z_k1_s1021
                                 - 0.619002342604471 z_k1_s1022
                                 - 0.526998248423015 z_k1_s1023
                                 - 0.953997952215821 z_k1_s1024
                                 - 0.471000227670454 z_k1_s1025
                                 - 0.860999402729756 z_k1_s1026
                                 - 0.44400003850012 z_k1_s1027
                                 - 0.546000796335895 z_k1_s1030
                                 - 0.666998732908861 z_k1_s1031
                                 - 0.413000061690059 z_k1_s1032
                                 - 0.484001478290303 z_k1_s1033
                                 - 0.151000065755922 z_k1_s1034
                                 - 0.585000158074285 z_k1_s1035
                                 - 0.154999917113719 z_k1_s1036
                                 - 0.242999890984731 z_k1_s1037
                                 - 0.407000144957589 z_k1_s1040
                                 - 1.03599830191048 z_k1_s1041
                                 - 0.502001133386886 z_k1_s1042
                                 - 0.549997657591642 z_k1_s1043
                                 - 0.871000981832744 z_k1_s1044
                                 - 0.293999718501597 z_k1_s1045
                                 - 0.985001377067996 z_k1_s1046
                                 - 0.298000087918581 z_k1_s1047
                                 - 0.972000638850757 z_k1_s1050
                                 - 0.370000924200686 z_k1_s1051
                                 - 0.275000077867814 z_k1_s1052
                                 - 0.389000248426894 z_k1_s1053
                                 - 0.434998612342978 z_k1_s1054
                                 - 0.487000283858337 z_k1_s1055
                                 - 0.370000924200686 z_k1_s1056
                                 - 0.223000166853217 z_k1_s1057
                                 - 0.70700015787105 z_k1_s1060
                                 - 1.1209950246929 z_k1_s1061
                                 - 0.979999298595635 z_k1_s1062
                                 - 0.638001013400741 z_k1_s1063
                                 - 0.755998961003484 z_k1_s1064
                                 - 0.430998932800425 z_k1_s1065
                                 - 0.946998444938805 z_k1_s1066
                                 - 0.228999551578609 z_k1_s1067
                                 - 0.807999034084797 z_k1_s1070
                                 - 1.79599397259127 z_k1_s1071
                                 - 1.04199839860662 z_k1_s1072
                                 - 1.73499914326768 z_k1_s1073
                                 - 0.868998959901473 z_k1_s1074
                                 - 0.763998970625386 z_k1_s1075
                                 - 0.620999528856265 z_k1_s1076
                                 - 2.55800123371635 z_k1_s1077
                                 - 0.206000045818659 z_k1_s1100
                                 - 0.240999778252038 z_k1_s1101
                                 - 0.256999829575798 z_k1_s1102
                                 - 0.532000910665343 z_k1_s1103
                                 - 0.648999495308564 z_k1_s1104
                                 - 0.370999073833043 z_k1_s1105
                                 - 0.355001048308463 z_k1_s1106
                                 - 0.282999804451952 z_k1_s1107
                                 - 0.448000208864524 z_k1_s1110
                                 - 0.26899928850652 z_k1_s1111
                                 - 0.164999715765318 z_k1_s1112
                                 - 0.32199911450529 z_k1_s1113
                                 - 0.216999502135919 z_k1_s1114
                                 - 0.448997649299376 z_k1_s1115
                                 - 0.712997714362548 z_k1_s1116
                                 - 0.665997801233624 z_k1_s1117
                                 - 0.49800137395348 z_k1_s1120
                                 - 0.179000200981822 z_k1_s1121
                                 - 0.293000186475708 z_k1_s1122
                                 - 0.65300015646168 z_k1_s1123
                                 - 0.459999678220719 z_k1_s1124
                                 - 0.484001696894858 z_k1_s1125
                                 - 0.244000116369498 z_k1_s1126
                                 - 0.190000075856564 z_k1_s1127
                                 - 0.2500017841587 z_k1_s1130
                                 - 0.193000831108486 z_k1_s1131
                                 - 0.247000819205288 z_k1_s1132
                                 - 0.279000848936864 z_k1_s1133
                                 - 0.610003360143536 z_k1_s1134
                                 - 0.348001192384111 z_k1_s1135
                                 - 0.433002047298992 z_k1_s1136
                                 - 0.123000579844973 z_k1_s1137
                                 - 0.557001051045444 z_k1_s1140
                                 - 0.607998551810911 z_k1_s1141
                                 - 1.00999644640791 z_k1_s1142
                                 - 0.999998565569236 z_k1_s1143
                                 - 0.663000743904003 z_k1_s1144
                                 - 0.472999465528045 z_k1_s1145
                                 - 0.520999263739556 z_k1_s1146
                                 - 0.706996958586321 z_k1_s1147
                                 - 0.135999345261805 z_k1_s1150
                                 - 0.112000013086068 z_k1_s1151
                                 - 0.158000428289181 z_k1_s1152
                                 - 0.144999551227604 z_k1_s1153
                                 - 0.0840000098145508 z_k1_s1154
                                 - 0.353999407990036 z_k1_s1155
                                 - 0.12699992169583 z_k1_s1156
                                 - 0.188000245508706 z_k1_s1157
                                 - 0.283999724088991 z_k1_s1160
                                 - 0.370000713946699 z_k1_s1161
                                 - 0.35400043132203 z_k1_s1162
                                 - 0.123000002752601 z_k1_s1163
                                 - 0.384999655782605 z_k1_s1164
                                 - 0.374998949892299 z_k1_s1165
                                 - 0.0840000018798249 z_k1_s1166
                                 - 0.142000073744451 z_k1_s1167
                                 - 0.583996816516539 z_k1_s1170
                                 - 0.141999324082649 z_k1_s1171
                                 - 0.465997778307865 z_k1_s1172
                                 - 0.326999401279985 z_k1_s1173
                                 - 0.233998953440091 z_k1_s1174
                                 - 0.423999100669454 z_k1_s1175
                                 - 0.182999194428726 z_k1_s1176
                                 - 0.153999341320254 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.32327 w_k1_ADM_s000
                                 + 1.25825 w_k1_AXP_s000
                                 + 1.15629 w_k1_DIS_s000
                                 + 1.10156 w_k1_FDX_s000
                                 + 0.926858 w_k1_IBM_s000
                                 + 0.91236 w_k1_INTC_s000
                                 + 1.02993 w_k1_KO_s000 + 1.01164 w_k1_MAT_s000
                                 + 1.02506 w_k1_MSFT_s000
                                 + 1.08014 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.64057 w_k1_ADM_s000
                                 + 1.2411 w_k1_AXP_s000 + 1.76116 w_k1_DIS_s000
                                 + 1.22058 w_k1_FDX_s000
                                 + 1.42722 w_k1_IBM_s000
                                 + 0.966044 w_k1_INTC_s000
                                 + 1.47324 w_k1_KO_s000
                                 + 0.848483 w_k1_MAT_s000
                                 + 1.29779 w_k1_MSFT_s000
                                 + 1.22823 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.23779 w_k1_ADM_s000
                                 + 0.893087 w_k1_AXP_s000
                                 + 1.0057 w_k1_DIS_s000
                                 + 0.961647 w_k1_FDX_s000
                                 + 1.10355 w_k1_IBM_s000
                                 + 0.823169 w_k1_INTC_s000
                                 + 1.18395 w_k1_KO_s000 + 1.29967 w_k1_MAT_s000
                                 + 1.00826 w_k1_MSFT_s000
                                 + 0.947737 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.14837 w_k1_ADM_s000
                                 + 1.14743 w_k1_AXP_s000
                                 + 1.48416 w_k1_DIS_s000
                                 + 1.33109 w_k1_FDX_s000
                                 + 1.15013 w_k1_IBM_s000
                                 + 1.03215 w_k1_INTC_s000
                                 + 1.18467 w_k1_KO_s000 + 1.07106 w_k1_MAT_s000
                                 + 1.23695 w_k1_MSFT_s000
                                 + 1.10776 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.859977 w_k1_ADM_s000
                                 + 0.879719 w_k1_AXP_s000
                                 + 1.00965 w_k1_DIS_s000 + 1.0123 w_k1_FDX_s000
                                 + 0.748196 w_k1_IBM_s000
                                 + 1.09502 w_k1_INTC_s000 + 1.0285 w_k1_KO_s000
                                 + 0.616648 w_k1_MAT_s000
                                 + 1.09424 w_k1_MSFT_s000
                                 + 0.963604 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.905783 w_k1_ADM_s000
                                 + 0.803025 w_k1_AXP_s000
                                 + 0.768418 w_k1_DIS_s000
                                 + 0.896569 w_k1_FDX_s000
                                 + 0.745084 w_k1_IBM_s000
                                 + 0.744248 w_k1_INTC_s000
                                 + 0.728316 w_k1_KO_s000
                                 + 0.542675 w_k1_MAT_s000
                                 + 0.76691 w_k1_MSFT_s000
                                 + 0.870069 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.893811 w_k1_ADM_s000
                                 + 0.999667 w_k1_AXP_s000
                                 + 1.11957 w_k1_DIS_s000
                                 + 0.918359 w_k1_FDX_s000
                                 + 1.41227 w_k1_IBM_s000
                                 + 0.820402 w_k1_INTC_s000
                                 + 1.2005 w_k1_KO_s000 + 1.40229 w_k1_MAT_s000
                                 + 0.902231 w_k1_MSFT_s000
                                 + 0.895851 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.872496 w_k1_ADM_s000
                                 + 0.652985 w_k1_AXP_s000
                                 + 0.743665 w_k1_DIS_s000
                                 + 0.729916 w_k1_FDX_s000
                                 + 0.781946 w_k1_IBM_s000
                                 + 0.588988 w_k1_INTC_s000
                                 + 0.891015 w_k1_KO_s000
                                 + 0.85873 w_k1_MAT_s000
                                 + 0.831508 w_k1_MSFT_s000
                                 + 0.827185 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.981052 w_k1_ADM_s001
                                 + 0.808438 w_k1_AXP_s001
                                 + 1.09944 w_k1_DIS_s001
                                 + 0.863892 w_k1_FDX_s001
                                 + 1.17189 w_k1_IBM_s001
                                 + 0.882239 w_k1_INTC_s001
                                 + 1.07519 w_k1_KO_s001
                                 + 0.892405 w_k1_MAT_s001
                                 + 1.0256 w_k1_MSFT_s001
                                 + 0.872158 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.52305 w_k1_ADM_s001
                                 + 1.30376 w_k1_AXP_s001
                                 + 1.64254 w_k1_DIS_s001
                                 + 1.37037 w_k1_FDX_s001
                                 + 1.13752 w_k1_IBM_s001
                                 + 1.40987 w_k1_INTC_s001
                                 + 1.60233 w_k1_KO_s001 + 1.15629 w_k1_MAT_s001
                                 + 1.50069 w_k1_MSFT_s001 + 1.274 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.919123 w_k1_ADM_s001
                                 + 0.809704 w_k1_AXP_s001
                                 + 0.729432 w_k1_DIS_s001
                                 + 0.769829 w_k1_FDX_s001
                                 + 0.770189 w_k1_IBM_s001
                                 + 0.944491 w_k1_INTC_s001
                                 + 0.865784 w_k1_KO_s001
                                 + 0.697851 w_k1_MAT_s001
                                 + 0.835904 w_k1_MSFT_s001
                                 + 0.800496 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.12261 w_k1_ADM_s001
                                 + 1.1925 w_k1_AXP_s001 + 1.21159 w_k1_DIS_s001
                                 + 1.15301 w_k1_FDX_s001
                                 + 0.901323 w_k1_IBM_s001
                                 + 1.24949 w_k1_INTC_s001 + 1.1127 w_k1_KO_s001
                                 + 0.91714 w_k1_MAT_s001
                                 + 1.28271 w_k1_MSFT_s001
                                 + 1.12123 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.13939 w_k1_ADM_s001
                                 + 1.36008 w_k1_AXP_s001
                                 + 1.50167 w_k1_DIS_s001 + 1.1205 w_k1_FDX_s001
                                 + 1.47467 w_k1_IBM_s001
                                 + 1.02629 w_k1_INTC_s001
                                 + 1.37899 w_k1_KO_s001 + 1.30604 w_k1_MAT_s001
                                 + 1.17464 w_k1_MSFT_s001
                                 + 1.08746 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.968676 w_k1_ADM_s001
                                 + 1.01352 w_k1_AXP_s001
                                 + 0.969461 w_k1_DIS_s001
                                 + 1.09661 w_k1_FDX_s001
                                 + 0.690662 w_k1_IBM_s001
                                 + 1.02116 w_k1_INTC_s001
                                 + 0.905418 w_k1_KO_s001
                                 + 0.903511 w_k1_MAT_s001
                                 + 1.09992 w_k1_MSFT_s001 + 0.9331 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05297 w_k1_ADM_s001
                                 + 0.981753 w_k1_AXP_s001
                                 + 0.849064 w_k1_DIS_s001
                                 + 0.93286 w_k1_FDX_s001
                                 + 1.07312 w_k1_IBM_s001
                                 + 1.08853 w_k1_INTC_s001
                                 + 1.16067 w_k1_KO_s001 + 1.3287 w_k1_MAT_s001
                                 + 1.00631 w_k1_MSFT_s001
                                 + 1.03861 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.912447 w_k1_ADM_s001
                                 + 0.694268 w_k1_AXP_s001
                                 + 0.891653 w_k1_DIS_s001
                                 + 0.906181 w_k1_FDX_s001
                                 + 0.707324 w_k1_IBM_s001
                                 + 0.867977 w_k1_INTC_s001
                                 + 0.949063 w_k1_KO_s001
                                 + 0.953657 w_k1_MAT_s001
                                 + 1.05934 w_k1_MSFT_s001
                                 + 0.900391 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.09688 w_k1_ADM_s002
                                 + 1.0353 w_k1_AXP_s002 + 0.94703 w_k1_DIS_s002
                                 + 1.03994 w_k1_FDX_s002
                                 + 0.722514 w_k1_IBM_s002
                                 + 1.23605 w_k1_INTC_s002
                                 + 0.852316 w_k1_KO_s002
                                 + 0.728426 w_k1_MAT_s002
                                 + 1.2332 w_k1_MSFT_s002
                                 + 0.944401 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.22868 w_k1_ADM_s002
                                 + 0.939109 w_k1_AXP_s002
                                 + 1.3351 w_k1_DIS_s002 + 1.1374 w_k1_FDX_s002
                                 + 0.900553 w_k1_IBM_s002
                                 + 1.23454 w_k1_INTC_s002
                                 + 1.02982 w_k1_KO_s002
                                 + 0.818827 w_k1_MAT_s002
                                 + 1.3097 w_k1_MSFT_s002 + 1.04897 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.15713 w_k1_ADM_s002
                                 + 0.848055 w_k1_AXP_s002
                                 + 0.809857 w_k1_DIS_s002
                                 + 0.891734 w_k1_FDX_s002
                                 + 0.96377 w_k1_IBM_s002
                                 + 1.08397 w_k1_INTC_s002 + 1.0411 w_k1_KO_s002
                                 + 0.866847 w_k1_MAT_s002
                                 + 0.903285 w_k1_MSFT_s002
                                 + 1.02845 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.838087 w_k1_ADM_s002
                                 + 0.865938 w_k1_AXP_s002
                                 + 0.634168 w_k1_DIS_s002
                                 + 0.865358 w_k1_FDX_s002
                                 + 0.666793 w_k1_IBM_s002
                                 + 1.01365 w_k1_INTC_s002
                                 + 0.684552 w_k1_KO_s002
                                 + 0.63874 w_k1_MAT_s002
                                 + 0.859941 w_k1_MSFT_s002
                                 + 0.892066 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.18224 w_k1_ADM_s002
                                 + 1.42503 w_k1_AXP_s002
                                 + 1.82183 w_k1_DIS_s002
                                 + 1.31849 w_k1_FDX_s002
                                 + 1.23237 w_k1_IBM_s002
                                 + 1.53186 w_k1_INTC_s002
                                 + 1.50073 w_k1_KO_s002 + 1.91613 w_k1_MAT_s002
                                 + 1.56581 w_k1_MSFT_s002
                                 + 1.22083 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.790956 w_k1_ADM_s002
                                 + 0.808881 w_k1_AXP_s002
                                 + 0.661306 w_k1_DIS_s002
                                 + 0.668536 w_k1_FDX_s002
                                 + 1.11822 w_k1_IBM_s002
                                 + 0.782207 w_k1_INTC_s002
                                 + 0.824207 w_k1_KO_s002
                                 + 0.961287 w_k1_MAT_s002
                                 + 0.779891 w_k1_MSFT_s002
                                 + 0.957413 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.12082 w_k1_ADM_s002
                                 + 1.59233 w_k1_AXP_s002
                                 + 1.10765 w_k1_DIS_s002
                                 + 1.24756 w_k1_FDX_s002
                                 + 1.29634 w_k1_IBM_s002
                                 + 1.15614 w_k1_INTC_s002
                                 + 1.36246 w_k1_KO_s002 + 1.53564 w_k1_MAT_s002
                                 + 1.0859 w_k1_MSFT_s002 + 1.10561 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.12562 w_k1_ADM_s002
                                 + 0.933474 w_k1_AXP_s002
                                 + 1.05971 w_k1_DIS_s002
                                 + 0.868029 w_k1_FDX_s002
                                 + 1.35535 w_k1_IBM_s002
                                 + 0.919803 w_k1_INTC_s002
                                 + 1.2207 w_k1_KO_s002 + 1.2478 w_k1_MAT_s002
                                 + 0.958707 w_k1_MSFT_s002
                                 + 1.00171 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.697437 w_k1_ADM_s003
                                 + 0.759736 w_k1_AXP_s003
                                 + 0.742631 w_k1_DIS_s003
                                 + 0.84051 w_k1_FDX_s003
                                 + 1.05806 w_k1_IBM_s003
                                 + 0.914213 w_k1_INTC_s003
                                 + 0.907563 w_k1_KO_s003 + 0.9249 w_k1_MAT_s003
                                 + 0.878883 w_k1_MSFT_s003
                                 + 0.943464 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.22381 w_k1_ADM_s003
                                 + 1.50007 w_k1_AXP_s003
                                 + 1.23205 w_k1_DIS_s003
                                 + 1.35958 w_k1_FDX_s003
                                 + 1.15327 w_k1_IBM_s003
                                 + 1.29908 w_k1_INTC_s003
                                 + 1.16263 w_k1_KO_s003 + 1.16151 w_k1_MAT_s003
                                 + 1.12821 w_k1_MSFT_s003
                                 + 1.29044 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.898446 w_k1_ADM_s003
                                 + 0.813308 w_k1_AXP_s003
                                 + 0.866145 w_k1_DIS_s003
                                 + 1.01404 w_k1_FDX_s003
                                 + 0.712342 w_k1_IBM_s003
                                 + 1.07965 w_k1_INTC_s003
                                 + 0.726834 w_k1_KO_s003
                                 + 0.550115 w_k1_MAT_s003
                                 + 0.961607 w_k1_MSFT_s003
                                 + 0.908773 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.592777 w_k1_ADM_s003
                                 + 0.593799 w_k1_AXP_s003
                                 + 0.663941 w_k1_DIS_s003
                                 + 0.820346 w_k1_FDX_s003
                                 + 0.669198 w_k1_IBM_s003
                                 + 0.82408 w_k1_INTC_s003
                                 + 0.751592 w_k1_KO_s003
                                 + 0.588172 w_k1_MAT_s003
                                 + 0.814171 w_k1_MSFT_s003
                                 + 0.761276 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.03382 w_k1_ADM_s003
                                 + 1.80924 w_k1_AXP_s003
                                 + 1.63003 w_k1_DIS_s003
                                 + 1.39252 w_k1_FDX_s003
                                 + 1.53946 w_k1_IBM_s003
                                 + 1.47703 w_k1_INTC_s003
                                 + 1.56771 w_k1_KO_s003 + 2.08795 w_k1_MAT_s003
                                 + 1.36138 w_k1_MSFT_s003
                                 + 1.43674 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.943709 w_k1_ADM_s003
                                 + 0.89934 w_k1_AXP_s003
                                 + 0.969245 w_k1_DIS_s003
                                 + 0.980832 w_k1_FDX_s003
                                 + 1.70661 w_k1_IBM_s003
                                 + 0.851581 w_k1_INTC_s003
                                 + 1.10752 w_k1_KO_s003 + 1.49991 w_k1_MAT_s003
                                 + 0.790726 w_k1_MSFT_s003
                                 + 1.07149 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.02569 w_k1_ADM_s003
                                 + 0.945782 w_k1_AXP_s003
                                 + 1.08054 w_k1_DIS_s003
                                 + 1.16643 w_k1_FDX_s003
                                 + 1.05105 w_k1_IBM_s003
                                 + 1.08733 w_k1_INTC_s003
                                 + 1.05712 w_k1_KO_s003 + 1.12692 w_k1_MAT_s003
                                 + 1.09678 w_k1_MSFT_s003
                                 + 1.08791 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.888485 w_k1_ADM_s003
                                 + 1.08981 w_k1_AXP_s003
                                 + 0.970981 w_k1_DIS_s003
                                 + 0.984122 w_k1_FDX_s003
                                 + 1.03449 w_k1_IBM_s003
                                 + 0.753918 w_k1_INTC_s003
                                 + 1.03132 w_k1_KO_s003
                                 + 0.912931 w_k1_MAT_s003
                                 + 0.778138 w_k1_MSFT_s003
                                 + 0.95837 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.0023 w_k1_ADM_s004
                                 + 1.14539 w_k1_AXP_s004
                                 + 1.06977 w_k1_DIS_s004
                                 + 1.15288 w_k1_FDX_s004
                                 + 0.830226 w_k1_IBM_s004
                                 + 1.02866 w_k1_INTC_s004
                                 + 0.987376 w_k1_KO_s004
                                 + 1.26946 w_k1_MAT_s004
                                 + 1.01557 w_k1_MSFT_s004
                                 + 0.974608 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.00917 w_k1_ADM_s004
                                 + 1.16235 w_k1_AXP_s004
                                 + 1.33095 w_k1_DIS_s004
                                 + 1.01035 w_k1_FDX_s004
                                 + 1.44602 w_k1_IBM_s004
                                 + 1.18117 w_k1_INTC_s004
                                 + 1.43359 w_k1_KO_s004 + 1.61691 w_k1_MAT_s004
                                 + 1.26166 w_k1_MSFT_s004
                                 + 1.03497 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.07901 w_k1_ADM_s004
                                 + 1.06852 w_k1_AXP_s004
                                 + 0.898167 w_k1_DIS_s004
                                 + 0.947803 w_k1_FDX_s004
                                 + 0.729218 w_k1_IBM_s004
                                 + 1.04107 w_k1_INTC_s004
                                 + 0.967834 w_k1_KO_s004
                                 + 0.816881 w_k1_MAT_s004
                                 + 1.01189 w_k1_MSFT_s004
                                 + 0.821327 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.19201 w_k1_ADM_s004
                                 + 1.46352 w_k1_AXP_s004
                                 + 1.63101 w_k1_DIS_s004
                                 + 1.29417 w_k1_FDX_s004
                                 + 1.04752 w_k1_IBM_s004
                                 + 1.47257 w_k1_INTC_s004
                                 + 1.44934 w_k1_KO_s004 + 1.33181 w_k1_MAT_s004
                                 + 1.41352 w_k1_MSFT_s004 + 1.0986 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.865403 w_k1_ADM_s004
                                 + 0.961224 w_k1_AXP_s004
                                 + 0.794407 w_k1_DIS_s004
                                 + 0.903105 w_k1_FDX_s004
                                 + 0.904711 w_k1_IBM_s004
                                 + 0.960065 w_k1_INTC_s004
                                 + 1.05597 w_k1_KO_s004 + 1.12689 w_k1_MAT_s004
                                 + 0.930459 w_k1_MSFT_s004
                                 + 0.999589 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.12318 w_k1_ADM_s004
                                 + 1.34214 w_k1_AXP_s004
                                 + 0.924226 w_k1_DIS_s004
                                 + 0.966339 w_k1_FDX_s004
                                 + 1.26117 w_k1_IBM_s004
                                 + 0.994634 w_k1_INTC_s004
                                 + 1.18535 w_k1_KO_s004 + 2.12303 w_k1_MAT_s004
                                 + 1.03394 w_k1_MSFT_s004
                                 + 1.02656 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.79188 w_k1_ADM_s004
                                 + 0.843398 w_k1_AXP_s004
                                 + 1.21705 w_k1_DIS_s004
                                 + 0.93566 w_k1_FDX_s004
                                 + 0.82195 w_k1_IBM_s004
                                 + 0.916962 w_k1_INTC_s004
                                 + 1.05853 w_k1_KO_s004
                                 + 0.928473 w_k1_MAT_s004
                                 + 0.952897 w_k1_MSFT_s004
                                 + 0.881758 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.801441 w_k1_ADM_s004
                                 + 0.769953 w_k1_AXP_s004
                                 + 0.621761 w_k1_DIS_s004
                                 + 0.760567 w_k1_FDX_s004
                                 + 0.767847 w_k1_IBM_s004
                                 + 0.766246 w_k1_INTC_s004
                                 + 0.791471 w_k1_KO_s004
                                 + 1.13479 w_k1_MAT_s004
                                 + 0.79865 w_k1_MSFT_s004
                                 + 0.754723 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.18256 w_k1_ADM_s005
                                 + 0.756919 w_k1_AXP_s005
                                 + 1.03139 w_k1_DIS_s005
                                 + 0.948225 w_k1_FDX_s005
                                 + 1.05478 w_k1_IBM_s005
                                 + 0.926732 w_k1_INTC_s005
                                 + 0.994755 w_k1_KO_s005
                                 + 0.918783 w_k1_MAT_s005
                                 + 1.10745 w_k1_MSFT_s005
                                 + 1.00199 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.970665 w_k1_ADM_s005
                                 + 1.2616 w_k1_AXP_s005 + 1.10061 w_k1_DIS_s005
                                 + 1.11954 w_k1_FDX_s005
                                 + 0.868787 w_k1_IBM_s005
                                 + 1.23714 w_k1_INTC_s005
                                 + 1.08567 w_k1_KO_s005 + 1.10973 w_k1_MAT_s005
                                 + 1.21422 w_k1_MSFT_s005
                                 + 1.20847 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.42469 w_k1_ADM_s005
                                 + 1.51776 w_k1_AXP_s005
                                 + 1.40038 w_k1_DIS_s005
                                 + 1.06078 w_k1_FDX_s005
                                 + 1.46973 w_k1_IBM_s005
                                 + 1.17309 w_k1_INTC_s005
                                 + 1.50809 w_k1_KO_s005 + 1.36389 w_k1_MAT_s005
                                 + 1.26829 w_k1_MSFT_s005
                                 + 1.23446 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.798617 w_k1_ADM_s005
                                 + 0.697946 w_k1_AXP_s005
                                 + 0.617368 w_k1_DIS_s005
                                 + 0.674433 w_k1_FDX_s005
                                 + 0.82011 w_k1_IBM_s005
                                 + 0.741172 w_k1_INTC_s005
                                 + 0.884355 w_k1_KO_s005
                                 + 0.900568 w_k1_MAT_s005
                                 + 0.724855 w_k1_MSFT_s005
                                 + 0.859843 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.16035 w_k1_ADM_s005
                                 + 0.981813 w_k1_AXP_s005
                                 + 1.03437 w_k1_DIS_s005
                                 + 0.967224 w_k1_FDX_s005
                                 + 1.13329 w_k1_IBM_s005
                                 + 1.20591 w_k1_INTC_s005
                                 + 1.09355 w_k1_KO_s005 + 1.06992 w_k1_MAT_s005
                                 + 1.22717 w_k1_MSFT_s005
                                 + 1.06622 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.00626 w_k1_ADM_s005
                                 + 1.06126 w_k1_AXP_s005
                                 + 0.826794 w_k1_DIS_s005
                                 + 0.853201 w_k1_FDX_s005
                                 + 1.23829 w_k1_IBM_s005
                                 + 0.949564 w_k1_INTC_s005
                                 + 1.17392 w_k1_KO_s005 + 1.82528 w_k1_MAT_s005
                                 + 0.912748 w_k1_MSFT_s005
                                 + 1.07025 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.31963 w_k1_ADM_s005
                                 + 1.25421 w_k1_AXP_s005
                                 + 1.33029 w_k1_DIS_s005
                                 + 1.22162 w_k1_FDX_s005
                                 + 1.25014 w_k1_IBM_s005
                                 + 1.36887 w_k1_INTC_s005
                                 + 1.23583 w_k1_KO_s005 + 1.3504 w_k1_MAT_s005
                                 + 1.1733 w_k1_MSFT_s005 + 1.20944 w_k1_PG_s005
                                 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.924988 w_k1_ADM_s005
                                 + 1.04818 w_k1_AXP_s005
                                 + 0.796054 w_k1_DIS_s005
                                 + 0.893246 w_k1_FDX_s005
                                 + 0.768179 w_k1_IBM_s005
                                 + 0.862458 w_k1_INTC_s005
                                 + 0.860576 w_k1_KO_s005
                                 + 1.14137 w_k1_MAT_s005
                                 + 0.875535 w_k1_MSFT_s005
                                 + 1.00363 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.52432 w_k1_ADM_s006
                                 + 1.64164 w_k1_AXP_s006
                                 + 1.54121 w_k1_DIS_s006
                                 + 1.42734 w_k1_FDX_s006
                                 + 1.13208 w_k1_IBM_s006
                                 + 0.981104 w_k1_INTC_s006
                                 + 1.23885 w_k1_KO_s006 + 1.55701 w_k1_MAT_s006
                                 + 1.36529 w_k1_MSFT_s006
                                 + 1.26953 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.944879 w_k1_ADM_s006
                                 + 0.599199 w_k1_AXP_s006
                                 + 0.519686 w_k1_DIS_s006
                                 + 0.739333 w_k1_FDX_s006
                                 + 0.728751 w_k1_IBM_s006
                                 + 0.715544 w_k1_INTC_s006
                                 + 0.688902 w_k1_KO_s006
                                 + 0.543125 w_k1_MAT_s006
                                 + 0.791517 w_k1_MSFT_s006
                                 + 0.801098 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.22772 w_k1_ADM_s006
                                 + 1.36579 w_k1_AXP_s006
                                 + 1.41868 w_k1_DIS_s006
                                 + 1.32042 w_k1_FDX_s006
                                 + 1.32781 w_k1_IBM_s006
                                 + 1.10749 w_k1_INTC_s006
                                 + 1.10565 w_k1_KO_s006 + 1.48114 w_k1_MAT_s006
                                 + 1.23503 w_k1_MSFT_s006
                                 + 1.02851 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.03002 w_k1_ADM_s006
                                 + 0.968268 w_k1_AXP_s006
                                 + 1.04367 w_k1_DIS_s006
                                 + 1.01027 w_k1_FDX_s006
                                 + 0.796871 w_k1_IBM_s006
                                 + 0.905431 w_k1_INTC_s006
                                 + 0.84483 w_k1_KO_s006 + 1.17231 w_k1_MAT_s006
                                 + 1.2211 w_k1_MSFT_s006 + 1.00454 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.820791 w_k1_ADM_s006
                                 + 0.875773 w_k1_AXP_s006
                                 + 0.760301 w_k1_DIS_s006
                                 + 0.867829 w_k1_FDX_s006
                                 + 0.846997 w_k1_IBM_s006
                                 + 0.969255 w_k1_INTC_s006
                                 + 0.774789 w_k1_KO_s006
                                 + 0.898023 w_k1_MAT_s006
                                 + 0.899659 w_k1_MSFT_s006
                                 + 0.882964 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.35848 w_k1_ADM_s006
                                 + 1.1772 w_k1_AXP_s006 + 1.10325 w_k1_DIS_s006
                                 + 1.14411 w_k1_FDX_s006
                                 + 0.982519 w_k1_IBM_s006
                                 + 1.02298 w_k1_INTC_s006
                                 + 1.04679 w_k1_KO_s006 + 1.5992 w_k1_MAT_s006
                                 + 1.20066 w_k1_MSFT_s006
                                 + 1.07021 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.0648 w_k1_ADM_s006
                                 + 0.77008 w_k1_AXP_s006
                                 + 1.05145 w_k1_DIS_s006
                                 + 0.889363 w_k1_FDX_s006
                                 + 1.25183 w_k1_IBM_s006
                                 + 0.91806 w_k1_INTC_s006 + 1.0888 w_k1_KO_s006
                                 + 1.09628 w_k1_MAT_s006
                                 + 1.00821 w_k1_MSFT_s006
                                 + 0.929418 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.17973 w_k1_ADM_s006
                                 + 0.98662 w_k1_AXP_s006 + 1.3765 w_k1_DIS_s006
                                 + 1.01581 w_k1_FDX_s006
                                 + 1.50295 w_k1_IBM_s006
                                 + 1.22821 w_k1_INTC_s006
                                 + 1.20789 w_k1_KO_s006 + 1.27869 w_k1_MAT_s006
                                 + 1.39564 w_k1_MSFT_s006
                                 + 1.11545 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.13088 w_k1_ADM_s007
                                 + 0.993024 w_k1_AXP_s007
                                 + 0.797924 w_k1_DIS_s007
                                 + 0.905446 w_k1_FDX_s007
                                 + 1.00653 w_k1_IBM_s007
                                 + 1.06249 w_k1_INTC_s007
                                 + 1.06435 w_k1_KO_s007 + 1.54329 w_k1_MAT_s007
                                 + 1.01914 w_k1_MSFT_s007
                                 + 1.01466 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.99942 w_k1_ADM_s007
                                 + 0.754729 w_k1_AXP_s007
                                 + 0.741962 w_k1_DIS_s007
                                 + 0.745651 w_k1_FDX_s007
                                 + 0.894627 w_k1_IBM_s007
                                 + 0.951233 w_k1_INTC_s007
                                 + 0.942709 w_k1_KO_s007
                                 + 1.16493 w_k1_MAT_s007
                                 + 0.968316 w_k1_MSFT_s007
                                 + 0.948961 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.833934 w_k1_ADM_s007
                                 + 0.78745 w_k1_AXP_s007
                                 + 0.708886 w_k1_DIS_s007
                                 + 0.818039 w_k1_FDX_s007
                                 + 0.654182 w_k1_IBM_s007
                                 + 1.11177 w_k1_INTC_s007
                                 + 0.788852 w_k1_KO_s007
                                 + 0.490901 w_k1_MAT_s007
                                 + 0.894154 w_k1_MSFT_s007
                                 + 0.872033 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.910037 w_k1_ADM_s007
                                 + 0.901639 w_k1_AXP_s007
                                 + 0.557835 w_k1_DIS_s007
                                 + 0.64787 w_k1_FDX_s007
                                 + 0.934627 w_k1_IBM_s007
                                 + 0.733337 w_k1_INTC_s007
                                 + 0.968905 w_k1_KO_s007
                                 + 1.17929 w_k1_MAT_s007
                                 + 0.715593 w_k1_MSFT_s007
                                 + 0.946417 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.33704 w_k1_ADM_s007
                                 + 1.30702 w_k1_AXP_s007
                                 + 1.16897 w_k1_DIS_s007
                                 + 1.11438 w_k1_FDX_s007
                                 + 0.940486 w_k1_IBM_s007
                                 + 1.29442 w_k1_INTC_s007
                                 + 1.27221 w_k1_KO_s007 + 1.17785 w_k1_MAT_s007
                                 + 1.26268 w_k1_MSFT_s007
                                 + 1.04005 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.28284 w_k1_ADM_s007
                                 + 0.920938 w_k1_AXP_s007
                                 + 1.0237 w_k1_DIS_s007
                                 + 0.983819 w_k1_FDX_s007
                                 + 1.4397 w_k1_IBM_s007
                                 + 1.14615 w_k1_INTC_s007
                                 + 1.25091 w_k1_KO_s007 + 1.66623 w_k1_MAT_s007
                                 + 1.10486 w_k1_MSFT_s007
                                 + 1.22385 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.66605 w_k1_ADM_s007
                                 + 1.63402 w_k1_AXP_s007
                                 + 1.48155 w_k1_DIS_s007
                                 + 1.28482 w_k1_FDX_s007
                                 + 1.58287 w_k1_IBM_s007
                                 + 1.37374 w_k1_INTC_s007
                                 + 1.45691 w_k1_KO_s007 + 1.35264 w_k1_MAT_s007
                                 + 1.20898 w_k1_MSFT_s007
                                 + 1.28584 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.847495 w_k1_ADM_s007
                                 + 0.869139 w_k1_AXP_s007
                                 + 1.02064 w_k1_DIS_s007
                                 + 0.990328 w_k1_FDX_s007
                                 + 0.79301 w_k1_IBM_s007
                                 + 1.17697 w_k1_INTC_s007
                                 + 0.783605 w_k1_KO_s007
                                 + 0.699036 w_k1_MAT_s007
                                 + 1.01188 w_k1_MSFT_s007
                                 + 0.924423 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.10743 w_k1_ADM_s010
                                 + 0.889295 w_k1_AXP_s010
                                 + 1.06571 w_k1_DIS_s010 + 1.099 w_k1_FDX_s010
                                 + 0.757369 w_k1_IBM_s010
                                 + 1.06255 w_k1_INTC_s010
                                 + 0.725951 w_k1_KO_s010
                                 + 0.443255 w_k1_MAT_s010
                                 + 0.975753 w_k1_MSFT_s010
                                 + 0.921925 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.878682 w_k1_ADM_s010
                                 + 0.838642 w_k1_AXP_s010
                                 + 0.640555 w_k1_DIS_s010
                                 + 0.764314 w_k1_FDX_s010
                                 + 0.968379 w_k1_IBM_s010
                                 + 0.804519 w_k1_INTC_s010
                                 + 0.69971 w_k1_KO_s010 + 1.15164 w_k1_MAT_s010
                                 + 0.825784 w_k1_MSFT_s010
                                 + 0.849967 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.30519 w_k1_ADM_s010
                                 + 1.02356 w_k1_AXP_s010
                                 + 1.28089 w_k1_DIS_s010
                                 + 1.07092 w_k1_FDX_s010
                                 + 1.03569 w_k1_IBM_s010
                                 + 0.945906 w_k1_INTC_s010
                                 + 0.894387 w_k1_KO_s010
                                 + 0.85818 w_k1_MAT_s010
                                 + 0.98436 w_k1_MSFT_s010
                                 + 1.14259 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.40006 w_k1_ADM_s010
                                 + 1.6829 w_k1_AXP_s010 + 1.72061 w_k1_DIS_s010
                                 + 1.41781 w_k1_FDX_s010
                                 + 1.76326 w_k1_IBM_s010
                                 + 1.46157 w_k1_INTC_s010
                                 + 1.27208 w_k1_KO_s010 + 1.42658 w_k1_MAT_s010
                                 + 1.24044 w_k1_MSFT_s010
                                 + 1.26983 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.20902 w_k1_ADM_s010
                                 + 0.881613 w_k1_AXP_s010
                                 + 0.93722 w_k1_DIS_s010
                                 + 0.873774 w_k1_FDX_s010
                                 + 1.1108 w_k1_IBM_s010
                                 + 0.841763 w_k1_INTC_s010
                                 + 0.946349 w_k1_KO_s010
                                 + 1.04091 w_k1_MAT_s010
                                 + 0.900244 w_k1_MSFT_s010
                                 + 0.995425 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.23463 w_k1_ADM_s010
                                 + 1.52827 w_k1_AXP_s010
                                 + 1.08451 w_k1_DIS_s010
                                 + 1.17476 w_k1_FDX_s010
                                 + 1.06753 w_k1_IBM_s010
                                 + 1.04539 w_k1_INTC_s010
                                 + 0.899146 w_k1_KO_s010
                                 + 0.844795 w_k1_MAT_s010
                                 + 0.866479 w_k1_MSFT_s010
                                 + 1.02046 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.25919 w_k1_ADM_s010
                                 + 1.20385 w_k1_AXP_s010
                                 + 1.33352 w_k1_DIS_s010
                                 + 1.12252 w_k1_FDX_s010
                                 + 1.27011 w_k1_IBM_s010
                                 + 0.953999 w_k1_INTC_s010
                                 + 1.17699 w_k1_KO_s010 + 1.39122 w_k1_MAT_s010
                                 + 1.09341 w_k1_MSFT_s010 + 1.1319 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.8799 w_k1_ADM_s010
                                 + 0.645013 w_k1_AXP_s010
                                 + 0.825467 w_k1_DIS_s010
                                 + 0.850078 w_k1_FDX_s010
                                 + 0.687444 w_k1_IBM_s010
                                 + 0.886733 w_k1_INTC_s010
                                 + 0.701673 w_k1_KO_s010
                                 + 0.458635 w_k1_MAT_s010
                                 + 0.785915 w_k1_MSFT_s010
                                 + 0.788191 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.703185 w_k1_ADM_s011
                                 + 1.00464 w_k1_AXP_s011
                                 + 0.777943 w_k1_DIS_s011
                                 + 0.891947 w_k1_FDX_s011
                                 + 0.931141 w_k1_IBM_s011
                                 + 0.915364 w_k1_INTC_s011
                                 + 0.773879 w_k1_KO_s011
                                 + 0.871962 w_k1_MAT_s011
                                 + 0.95406 w_k1_MSFT_s011
                                 + 0.897352 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.01108 w_k1_ADM_s011
                                 + 1.41429 w_k1_AXP_s011
                                 + 1.37299 w_k1_DIS_s011
                                 + 1.21441 w_k1_FDX_s011
                                 + 1.37714 w_k1_IBM_s011
                                 + 0.953663 w_k1_INTC_s011
                                 + 1.22234 w_k1_KO_s011 + 1.3378 w_k1_MAT_s011
                                 + 1.25565 w_k1_MSFT_s011
                                 + 1.18876 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.05192 w_k1_ADM_s011
                                 + 1.43868 w_k1_AXP_s011
                                 + 1.58123 w_k1_DIS_s011
                                 + 1.51324 w_k1_FDX_s011
                                 + 0.942191 w_k1_IBM_s011
                                 + 1.07194 w_k1_INTC_s011
                                 + 1.05051 w_k1_KO_s011 + 1.08842 w_k1_MAT_s011
                                 + 1.31626 w_k1_MSFT_s011
                                 + 1.12947 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.18404 w_k1_ADM_s011
                                 + 1.83995 w_k1_AXP_s011
                                 + 1.31922 w_k1_DIS_s011
                                 + 1.36694 w_k1_FDX_s011
                                 + 1.04948 w_k1_IBM_s011
                                 + 1.00669 w_k1_INTC_s011 + 1.0957 w_k1_KO_s011
                                 + 1.56588 w_k1_MAT_s011
                                 + 1.17282 w_k1_MSFT_s011 + 1.0571 w_k1_PG_s011
                                 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.922269 w_k1_ADM_s011
                                 + 1.08845 w_k1_AXP_s011
                                 + 1.07387 w_k1_DIS_s011
                                 + 1.13696 w_k1_FDX_s011
                                 + 0.987963 w_k1_IBM_s011
                                 + 1.02512 w_k1_INTC_s011
                                 + 1.02461 w_k1_KO_s011 + 1.17001 w_k1_MAT_s011
                                 + 1.08926 w_k1_MSFT_s011
                                 + 1.09328 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.81012 w_k1_ADM_s011
                                 + 0.700011 w_k1_AXP_s011
                                 + 0.753466 w_k1_DIS_s011
                                 + 0.898886 w_k1_FDX_s011
                                 + 0.65033 w_k1_IBM_s011
                                 + 0.733003 w_k1_INTC_s011
                                 + 0.737812 w_k1_KO_s011
                                 + 0.463534 w_k1_MAT_s011
                                 + 0.905007 w_k1_MSFT_s011
                                 + 0.868332 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.739214 w_k1_ADM_s011
                                 + 0.613799 w_k1_AXP_s011
                                 + 0.607318 w_k1_DIS_s011
                                 + 0.667001 w_k1_FDX_s011
                                 + 0.929294 w_k1_IBM_s011
                                 + 0.804478 w_k1_INTC_s011
                                 + 0.831898 w_k1_KO_s011
                                 + 0.752703 w_k1_MAT_s011
                                 + 0.832577 w_k1_MSFT_s011
                                 + 0.73729 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.704466 w_k1_ADM_s011
                                 + 1.12232 w_k1_AXP_s011
                                 + 1.00379 w_k1_DIS_s011
                                 + 0.976879 w_k1_FDX_s011
                                 + 1.23798 w_k1_IBM_s011
                                 + 1.2165 w_k1_INTC_s011 + 1.05125 w_k1_KO_s011
                                 + 1.43113 w_k1_MAT_s011
                                 + 1.02139 w_k1_MSFT_s011
                                 + 0.963115 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.26538 w_k1_ADM_s012
                                 + 1.79641 w_k1_AXP_s012
                                 + 1.60112 w_k1_DIS_s012
                                 + 1.50758 w_k1_FDX_s012
                                 + 1.26049 w_k1_IBM_s012
                                 + 1.53365 w_k1_INTC_s012
                                 + 1.44977 w_k1_KO_s012 + 1.39411 w_k1_MAT_s012
                                 + 1.14882 w_k1_MSFT_s012
                                 + 1.18886 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.984688 w_k1_ADM_s012
                                 + 1.10297 w_k1_AXP_s012
                                 + 1.10376 w_k1_DIS_s012
                                 + 1.03173 w_k1_FDX_s012
                                 + 1.43473 w_k1_IBM_s012
                                 + 1.19313 w_k1_INTC_s012
                                 + 1.28899 w_k1_KO_s012 + 1.04044 w_k1_MAT_s012
                                 + 1.00056 w_k1_MSFT_s012
                                 + 1.19644 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.09882 w_k1_ADM_s012
                                 + 1.07099 w_k1_AXP_s012
                                 + 1.41638 w_k1_DIS_s012
                                 + 1.23012 w_k1_FDX_s012
                                 + 1.28017 w_k1_IBM_s012
                                 + 1.38654 w_k1_INTC_s012
                                 + 1.26863 w_k1_KO_s012
                                 + 0.972607 w_k1_MAT_s012
                                 + 1.12978 w_k1_MSFT_s012
                                 + 1.02097 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.10644 w_k1_ADM_s012
                                 + 1.00767 w_k1_AXP_s012
                                 + 1.35742 w_k1_DIS_s012
                                 + 1.24367 w_k1_FDX_s012
                                 + 0.935789 w_k1_IBM_s012
                                 + 1.16256 w_k1_INTC_s012
                                 + 1.15339 w_k1_KO_s012
                                 + 0.935359 w_k1_MAT_s012
                                 + 1.09602 w_k1_MSFT_s012
                                 + 0.988283 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.990147 w_k1_ADM_s012
                                 + 0.888967 w_k1_AXP_s012
                                 + 0.917492 w_k1_DIS_s012
                                 + 1.0785 w_k1_FDX_s012
                                 + 0.753749 w_k1_IBM_s012
                                 + 1.04831 w_k1_INTC_s012
                                 + 0.867369 w_k1_KO_s012
                                 + 0.67765 w_k1_MAT_s012
                                 + 0.877451 w_k1_MSFT_s012
                                 + 0.942875 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.896474 w_k1_ADM_s012
                                 + 0.994587 w_k1_AXP_s012
                                 + 0.952786 w_k1_DIS_s012
                                 + 1.0454 w_k1_FDX_s012
                                 + 0.996526 w_k1_IBM_s012
                                 + 1.19926 w_k1_INTC_s012 + 1.1299 w_k1_KO_s012
                                 + 1.19533 w_k1_MAT_s012
                                 + 1.10684 w_k1_MSFT_s012
                                 + 1.03736 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.722126 w_k1_ADM_s012
                                 + 0.691348 w_k1_AXP_s012
                                 + 0.714754 w_k1_DIS_s012
                                 + 0.818517 w_k1_FDX_s012
                                 + 0.723753 w_k1_IBM_s012
                                 + 1.05676 w_k1_INTC_s012
                                 + 0.779468 w_k1_KO_s012
                                 + 0.52351 w_k1_MAT_s012
                                 + 0.791298 w_k1_MSFT_s012
                                 + 0.884608 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.817608 w_k1_ADM_s012
                                 + 0.783396 w_k1_AXP_s012
                                 + 0.738614 w_k1_DIS_s012
                                 + 0.885202 w_k1_FDX_s012
                                 + 1.04166 w_k1_IBM_s012
                                 + 1.33988 w_k1_INTC_s012
                                 + 0.943896 w_k1_KO_s012
                                 + 0.895684 w_k1_MAT_s012
                                 + 0.906659 w_k1_MSFT_s012
                                 + 0.940242 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.1352 w_k1_ADM_s013
                                 + 1.00684 w_k1_AXP_s013
                                 + 0.895296 w_k1_DIS_s013
                                 + 1.07349 w_k1_FDX_s013
                                 + 0.706683 w_k1_IBM_s013
                                 + 0.982276 w_k1_INTC_s013
                                 + 0.847496 w_k1_KO_s013
                                 + 0.843219 w_k1_MAT_s013
                                 + 0.941743 w_k1_MSFT_s013
                                 + 1.01794 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.737948 w_k1_ADM_s013
                                 + 0.571842 w_k1_AXP_s013
                                 + 0.777477 w_k1_DIS_s013
                                 + 0.741911 w_k1_FDX_s013
                                 + 0.99758 w_k1_IBM_s013
                                 + 0.881164 w_k1_INTC_s013
                                 + 0.808542 w_k1_KO_s013
                                 + 0.589166 w_k1_MAT_s013
                                 + 0.779818 w_k1_MSFT_s013
                                 + 0.890077 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.68967 w_k1_ADM_s013
                                 + 1.51386 w_k1_AXP_s013
                                 + 1.27913 w_k1_DIS_s013
                                 + 1.37749 w_k1_FDX_s013
                                 + 0.779756 w_k1_IBM_s013
                                 + 0.964005 w_k1_INTC_s013
                                 + 0.973191 w_k1_KO_s013
                                 + 0.93959 w_k1_MAT_s013
                                 + 0.999166 w_k1_MSFT_s013
                                 + 1.21333 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.17172 w_k1_ADM_s013
                                 + 1.61262 w_k1_AXP_s013
                                 + 2.13206 w_k1_DIS_s013
                                 + 1.36462 w_k1_FDX_s013
                                 + 2.17507 w_k1_IBM_s013
                                 + 1.13907 w_k1_INTC_s013
                                 + 1.70021 w_k1_KO_s013 + 1.73002 w_k1_MAT_s013
                                 + 1.26641 w_k1_MSFT_s013
                                 + 1.40494 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.719518 w_k1_ADM_s013
                                 + 0.660005 w_k1_AXP_s013
                                 + 0.60324 w_k1_DIS_s013
                                 + 0.74396 w_k1_FDX_s013
                                 + 0.750509 w_k1_IBM_s013
                                 + 0.701816 w_k1_INTC_s013
                                 + 0.644608 w_k1_KO_s013
                                 + 0.448175 w_k1_MAT_s013
                                 + 0.591695 w_k1_MSFT_s013
                                 + 0.72687 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.992879 w_k1_ADM_s013
                                 + 0.913573 w_k1_AXP_s013
                                 + 1.16142 w_k1_DIS_s013
                                 + 1.12538 w_k1_FDX_s013
                                 + 1.01104 w_k1_IBM_s013
                                 + 0.939944 w_k1_INTC_s013
                                 + 0.939797 w_k1_KO_s013
                                 + 0.980027 w_k1_MAT_s013
                                 + 0.962705 w_k1_MSFT_s013
                                 + 0.982151 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.831391 w_k1_ADM_s013
                                 + 1.10505 w_k1_AXP_s013
                                 + 1.23232 w_k1_DIS_s013
                                 + 1.05442 w_k1_FDX_s013
                                 + 1.43356 w_k1_IBM_s013
                                 + 0.971757 w_k1_INTC_s013
                                 + 1.19842 w_k1_KO_s013 + 1.1222 w_k1_MAT_s013
                                 + 1.00737 w_k1_MSFT_s013
                                 + 0.994617 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.56377 w_k1_ADM_s013
                                 + 1.62664 w_k1_AXP_s013
                                 + 1.64565 w_k1_DIS_s013
                                 + 1.58528 w_k1_FDX_s013
                                 + 1.19393 w_k1_IBM_s013
                                 + 1.47978 w_k1_INTC_s013
                                 + 1.47862 w_k1_KO_s013 + 1.37718 w_k1_MAT_s013
                                 + 1.41384 w_k1_MSFT_s013
                                 + 1.14667 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.800202 w_k1_ADM_s014
                                 + 0.616221 w_k1_AXP_s014
                                 + 0.685161 w_k1_DIS_s014
                                 + 0.749485 w_k1_FDX_s014
                                 + 0.883127 w_k1_IBM_s014
                                 + 0.773234 w_k1_INTC_s014
                                 + 0.918938 w_k1_KO_s014
                                 + 0.619215 w_k1_MAT_s014
                                 + 0.759608 w_k1_MSFT_s014
                                 + 0.798709 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.18706 w_k1_ADM_s014
                                 + 0.971769 w_k1_AXP_s014
                                 + 1.2155 w_k1_DIS_s014 + 1.1261 w_k1_FDX_s014
                                 + 1.09195 w_k1_IBM_s014
                                 + 1.22727 w_k1_INTC_s014
                                 + 1.11212 w_k1_KO_s014
                                 + 0.623823 w_k1_MAT_s014
                                 + 1.19506 w_k1_MSFT_s014 + 1.0642 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.02939 w_k1_ADM_s014
                                 + 0.829688 w_k1_AXP_s014
                                 + 0.845494 w_k1_DIS_s014
                                 + 0.976099 w_k1_FDX_s014
                                 + 0.71929 w_k1_IBM_s014
                                 + 0.981516 w_k1_INTC_s014
                                 + 0.905155 w_k1_KO_s014
                                 + 0.590642 w_k1_MAT_s014
                                 + 1.09503 w_k1_MSFT_s014
                                 + 0.858036 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.872114 w_k1_ADM_s014
                                 + 0.667987 w_k1_AXP_s014
                                 + 0.590726 w_k1_DIS_s014
                                 + 0.756123 w_k1_FDX_s014
                                 + 0.588003 w_k1_IBM_s014
                                 + 0.690632 w_k1_INTC_s014
                                 + 0.690075 w_k1_KO_s014
                                 + 0.579206 w_k1_MAT_s014
                                 + 0.897556 w_k1_MSFT_s014
                                 + 0.825184 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.19642 w_k1_ADM_s014
                                 + 1.04459 w_k1_AXP_s014
                                 + 0.974624 w_k1_DIS_s014
                                 + 0.981662 w_k1_FDX_s014
                                 + 1.37232 w_k1_IBM_s014
                                 + 0.634439 w_k1_INTC_s014
                                 + 1.01516 w_k1_KO_s014 + 1.32257 w_k1_MAT_s014
                                 + 0.909952 w_k1_MSFT_s014
                                 + 0.935605 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.02981 w_k1_ADM_s014
                                 + 1.50194 w_k1_AXP_s014 + 1.5554 w_k1_DIS_s014
                                 + 1.35296 w_k1_FDX_s014
                                 + 1.59088 w_k1_IBM_s014
                                 + 1.14881 w_k1_INTC_s014
                                 + 1.55818 w_k1_KO_s014 + 1.52501 w_k1_MAT_s014
                                 + 1.36281 w_k1_MSFT_s014
                                 + 1.20119 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.853268 w_k1_ADM_s014
                                 + 0.666376 w_k1_AXP_s014
                                 + 0.79056 w_k1_DIS_s014
                                 + 0.789906 w_k1_FDX_s014
                                 + 1.14585 w_k1_IBM_s014
                                 + 0.780254 w_k1_INTC_s014
                                 + 0.979911 w_k1_KO_s014
                                 + 0.75574 w_k1_MAT_s014
                                 + 0.868281 w_k1_MSFT_s014
                                 + 0.89803 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.940714 w_k1_ADM_s014
                                 + 1.19958 w_k1_AXP_s014
                                 + 1.34216 w_k1_DIS_s014
                                 + 1.28103 w_k1_FDX_s014
                                 + 0.858814 w_k1_IBM_s014
                                 + 1.44021 w_k1_INTC_s014
                                 + 1.16974 w_k1_KO_s014
                                 + 0.842799 w_k1_MAT_s014
                                 + 1.37491 w_k1_MSFT_s014
                                 + 1.11657 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.980819 w_k1_ADM_s015
                                 + 0.995863 w_k1_AXP_s015
                                 + 1.21061 w_k1_DIS_s015
                                 + 0.994181 w_k1_FDX_s015 + 1.313 w_k1_IBM_s015
                                 + 1.10786 w_k1_INTC_s015
                                 + 1.24033 w_k1_KO_s015 + 1.00855 w_k1_MAT_s015
                                 + 1.03385 w_k1_MSFT_s015
                                 + 1.08077 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.814619 w_k1_ADM_s015
                                 + 0.926467 w_k1_AXP_s015
                                 + 0.899205 w_k1_DIS_s015
                                 + 0.956156 w_k1_FDX_s015
                                 + 0.694548 w_k1_IBM_s015
                                 + 0.941495 w_k1_INTC_s015
                                 + 0.809138 w_k1_KO_s015
                                 + 0.77195 w_k1_MAT_s015
                                 + 0.938803 w_k1_MSFT_s015
                                 + 0.921841 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.927422 w_k1_ADM_s015
                                 + 1.25391 w_k1_AXP_s015
                                 + 1.24518 w_k1_DIS_s015
                                 + 1.08145 w_k1_FDX_s015
                                 + 1.55725 w_k1_IBM_s015
                                 + 1.49704 w_k1_INTC_s015
                                 + 1.40306 w_k1_KO_s015 + 1.42999 w_k1_MAT_s015
                                 + 1.24578 w_k1_MSFT_s015
                                 + 1.15762 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.856454 w_k1_ADM_s015
                                 + 1.07435 w_k1_AXP_s015
                                 + 0.869924 w_k1_DIS_s015
                                 + 0.911227 w_k1_FDX_s015
                                 + 1.05331 w_k1_IBM_s015
                                 + 1.06725 w_k1_INTC_s015
                                 + 1.01717 w_k1_KO_s015 + 1.0052 w_k1_MAT_s015
                                 + 0.963683 w_k1_MSFT_s015
                                 + 1.09342 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.754015 w_k1_ADM_s015
                                 + 0.701021 w_k1_AXP_s015
                                 + 0.645045 w_k1_DIS_s015
                                 + 0.788135 w_k1_FDX_s015
                                 + 0.822738 w_k1_IBM_s015
                                 + 0.845567 w_k1_INTC_s015
                                 + 0.814481 w_k1_KO_s015
                                 + 0.775805 w_k1_MAT_s015
                                 + 0.890454 w_k1_MSFT_s015
                                 + 0.863632 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.97559 w_k1_ADM_s015
                                 + 1.73585 w_k1_AXP_s015
                                 + 1.56554 w_k1_DIS_s015
                                 + 1.30504 w_k1_FDX_s015
                                 + 1.29628 w_k1_IBM_s015
                                 + 1.61695 w_k1_INTC_s015
                                 + 1.36538 w_k1_KO_s015 + 1.27964 w_k1_MAT_s015
                                 + 1.41852 w_k1_MSFT_s015
                                 + 1.29295 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.912925 w_k1_ADM_s015
                                 + 1.2245 w_k1_AXP_s015 + 1.44393 w_k1_DIS_s015
                                 + 1.26214 w_k1_FDX_s015
                                 + 0.94385 w_k1_IBM_s015
                                 + 1.26739 w_k1_INTC_s015
                                 + 1.18178 w_k1_KO_s015
                                 + 0.844543 w_k1_MAT_s015
                                 + 1.29514 w_k1_MSFT_s015
                                 + 1.20036 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.00633 w_k1_ADM_s015
                                 + 1.3482 w_k1_AXP_s015
                                 + 0.944127 w_k1_DIS_s015
                                 + 1.05027 w_k1_FDX_s015
                                 + 1.11148 w_k1_IBM_s015
                                 + 1.22401 w_k1_INTC_s015
                                 + 0.940075 w_k1_KO_s015
                                 + 0.846204 w_k1_MAT_s015
                                 + 1.25476 w_k1_MSFT_s015
                                 + 1.07998 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.76633 w_k1_ADM_s016
                                 + 0.668605 w_k1_AXP_s016
                                 + 0.766032 w_k1_DIS_s016
                                 + 0.816611 w_k1_FDX_s016
                                 + 0.929302 w_k1_IBM_s016
                                 + 0.961527 w_k1_INTC_s016
                                 + 0.88177 w_k1_KO_s016
                                 + 0.687748 w_k1_MAT_s016
                                 + 0.870456 w_k1_MSFT_s016
                                 + 0.867401 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.16794 w_k1_ADM_s016
                                 + 1.07977 w_k1_AXP_s016
                                 + 1.23161 w_k1_DIS_s016
                                 + 1.09422 w_k1_FDX_s016
                                 + 1.08256 w_k1_IBM_s016
                                 + 1.18597 w_k1_INTC_s016
                                 + 1.12536 w_k1_KO_s016 + 1.26447 w_k1_MAT_s016
                                 + 1.22043 w_k1_MSFT_s016
                                 + 1.12902 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.619512 w_k1_ADM_s016
                                 + 0.627059 w_k1_AXP_s016
                                 + 0.598302 w_k1_DIS_s016
                                 + 0.699542 w_k1_FDX_s016
                                 + 0.594051 w_k1_IBM_s016
                                 + 0.828903 w_k1_INTC_s016
                                 + 0.682416 w_k1_KO_s016
                                 + 0.714724 w_k1_MAT_s016
                                 + 0.679897 w_k1_MSFT_s016
                                 + 0.719195 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.09523 w_k1_ADM_s016
                                 + 0.923954 w_k1_AXP_s016
                                 + 1.04465 w_k1_DIS_s016
                                 + 1.06026 w_k1_FDX_s016
                                 + 0.811646 w_k1_IBM_s016
                                 + 1.14196 w_k1_INTC_s016
                                 + 0.812761 w_k1_KO_s016
                                 + 0.601742 w_k1_MAT_s016
                                 + 1.07694 w_k1_MSFT_s016
                                 + 0.879009 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.769374 w_k1_ADM_s016
                                 + 0.741115 w_k1_AXP_s016
                                 + 0.966374 w_k1_DIS_s016
                                 + 0.943391 w_k1_FDX_s016
                                 + 0.73697 w_k1_IBM_s016
                                 + 1.09846 w_k1_INTC_s016
                                 + 0.857234 w_k1_KO_s016
                                 + 0.638436 w_k1_MAT_s016
                                 + 0.937631 w_k1_MSFT_s016
                                 + 0.877201 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.37724 w_k1_ADM_s016
                                 + 1.50745 w_k1_AXP_s016
                                 + 1.81849 w_k1_DIS_s016
                                 + 1.43618 w_k1_FDX_s016
                                 + 0.93263 w_k1_IBM_s016
                                 + 1.39981 w_k1_INTC_s016
                                 + 1.16802 w_k1_KO_s016 + 1.29698 w_k1_MAT_s016
                                 + 1.36004 w_k1_MSFT_s016
                                 + 1.20291 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.14328 w_k1_ADM_s016
                                 + 1.28385 w_k1_AXP_s016
                                 + 1.48156 w_k1_DIS_s016
                                 + 1.17084 w_k1_FDX_s016
                                 + 1.79661 w_k1_IBM_s016
                                 + 1.25428 w_k1_INTC_s016
                                 + 1.38751 w_k1_KO_s016 + 1.58977 w_k1_MAT_s016
                                 + 1.22809 w_k1_MSFT_s016
                                 + 1.18685 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.89573 w_k1_ADM_s016
                                 + 1.04662 w_k1_AXP_s016
                                 + 0.853144 w_k1_DIS_s016
                                 + 0.940533 w_k1_FDX_s016
                                 + 1.25808 w_k1_IBM_s016
                                 + 1.30591 w_k1_INTC_s016
                                 + 0.892585 w_k1_KO_s016
                                 + 1.44063 w_k1_MAT_s016
                                 + 0.887849 w_k1_MSFT_s016
                                 + 0.925079 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.03706 w_k1_ADM_s017
                                 + 1.25077 w_k1_AXP_s017
                                 + 1.27106 w_k1_DIS_s017
                                 + 0.913078 w_k1_FDX_s017
                                 + 1.59357 w_k1_IBM_s017
                                 + 1.06899 w_k1_INTC_s017
                                 + 1.35252 w_k1_KO_s017 + 1.55911 w_k1_MAT_s017
                                 + 0.871113 w_k1_MSFT_s017
                                 + 1.16267 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.11187 w_k1_ADM_s017
                                 + 0.920481 w_k1_AXP_s017
                                 + 1.27557 w_k1_DIS_s017 + 1.062 w_k1_FDX_s017
                                 + 0.729324 w_k1_IBM_s017
                                 + 1.09686 w_k1_INTC_s017
                                 + 1.01594 w_k1_KO_s017
                                 + 0.599875 w_k1_MAT_s017
                                 + 1.02798 w_k1_MSFT_s017
                                 + 0.988821 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.94759 w_k1_ADM_s017
                                 + 0.622647 w_k1_AXP_s017
                                 + 0.677342 w_k1_DIS_s017
                                 + 0.709829 w_k1_FDX_s017
                                 + 0.702862 w_k1_IBM_s017
                                 + 0.678384 w_k1_INTC_s017
                                 + 0.736956 w_k1_KO_s017
                                 + 0.719682 w_k1_MAT_s017
                                 + 0.751845 w_k1_MSFT_s017
                                 + 0.754497 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.30588 w_k1_ADM_s017
                                 + 1.12922 w_k1_AXP_s017
                                 + 1.00208 w_k1_DIS_s017
                                 + 0.952331 w_k1_FDX_s017
                                 + 1.0177 w_k1_IBM_s017
                                 + 0.968088 w_k1_INTC_s017
                                 + 1.07685 w_k1_KO_s017 + 1.20723 w_k1_MAT_s017
                                 + 0.87723 w_k1_MSFT_s017
                                 + 0.937093 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.1429 w_k1_ADM_s017
                                 + 1.53824 w_k1_AXP_s017
                                 + 1.69666 w_k1_DIS_s017
                                 + 1.21313 w_k1_FDX_s017
                                 + 1.55451 w_k1_IBM_s017
                                 + 1.34629 w_k1_INTC_s017
                                 + 1.48361 w_k1_KO_s017 + 1.52089 w_k1_MAT_s017
                                 + 1.09816 w_k1_MSFT_s017
                                 + 1.05706 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.0162 w_k1_ADM_s017
                                 + 0.866243 w_k1_AXP_s017
                                 + 0.889732 w_k1_DIS_s017
                                 + 0.910816 w_k1_FDX_s017
                                 + 0.611344 w_k1_IBM_s017
                                 + 1.15083 w_k1_INTC_s017
                                 + 0.787895 w_k1_KO_s017
                                 + 0.428529 w_k1_MAT_s017
                                 + 0.866663 w_k1_MSFT_s017
                                 + 0.832777 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.10452 w_k1_ADM_s017
                                 + 1.36185 w_k1_AXP_s017
                                 + 1.71579 w_k1_DIS_s017
                                 + 1.16431 w_k1_FDX_s017
                                 + 1.08004 w_k1_IBM_s017
                                 + 1.43186 w_k1_INTC_s017
                                 + 1.39337 w_k1_KO_s017
                                 + 0.981109 w_k1_MAT_s017
                                 + 1.15776 w_k1_MSFT_s017
                                 + 1.17394 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.889904 w_k1_ADM_s017
                                 + 0.875478 w_k1_AXP_s017
                                 + 0.791822 w_k1_DIS_s017
                                 + 0.704728 w_k1_FDX_s017
                                 + 1.27086 w_k1_IBM_s017
                                 + 0.958294 w_k1_INTC_s017
                                 + 1.02248 w_k1_KO_s017 + 1.04421 w_k1_MAT_s017
                                 + 0.817422 w_k1_MSFT_s017
                                 + 0.978287 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.891325 w_k1_ADM_s100
                                 + 1.38808 w_k1_AXP_s100
                                 + 1.74457 w_k1_DIS_s100
                                 + 1.33463 w_k1_FDX_s100
                                 + 1.12914 w_k1_IBM_s100
                                 + 1.36311 w_k1_INTC_s100
                                 + 1.33998 w_k1_KO_s100 + 1.30693 w_k1_MAT_s100
                                 + 1.19955 w_k1_MSFT_s100
                                 + 1.22276 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.17336 w_k1_ADM_s100
                                 + 1.30276 w_k1_AXP_s100
                                 + 1.11385 w_k1_DIS_s100
                                 + 1.17528 w_k1_FDX_s100
                                 + 1.03687 w_k1_IBM_s100
                                 + 1.04823 w_k1_INTC_s100
                                 + 1.15755 w_k1_KO_s100 + 1.04479 w_k1_MAT_s100
                                 + 1.07422 w_k1_MSFT_s100
                                 + 1.07662 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.974867 w_k1_ADM_s100
                                 + 1.04873 w_k1_AXP_s100
                                 + 0.888993 w_k1_DIS_s100
                                 + 0.929676 w_k1_FDX_s100
                                 + 1.21844 w_k1_IBM_s100
                                 + 0.919006 w_k1_INTC_s100
                                 + 0.94487 w_k1_KO_s100
                                 + 0.913329 w_k1_MAT_s100
                                 + 0.851509 w_k1_MSFT_s100
                                 + 1.01067 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.876953 w_k1_ADM_s100
                                 + 0.683518 w_k1_AXP_s100
                                 + 1.06555 w_k1_DIS_s100
                                 + 0.944124 w_k1_FDX_s100
                                 + 0.807484 w_k1_IBM_s100
                                 + 0.936496 w_k1_INTC_s100
                                 + 1.03953 w_k1_KO_s100
                                 + 0.580486 w_k1_MAT_s100
                                 + 1.03556 w_k1_MSFT_s100
                                 + 0.963817 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.920218 w_k1_ADM_s100
                                 + 0.759519 w_k1_AXP_s100
                                 + 0.693577 w_k1_DIS_s100
                                 + 0.807268 w_k1_FDX_s100
                                 + 0.780553 w_k1_IBM_s100
                                 + 1.02658 w_k1_INTC_s100
                                 + 0.918332 w_k1_KO_s100
                                 + 0.669386 w_k1_MAT_s100
                                 + 0.747299 w_k1_MSFT_s100
                                 + 0.849162 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.979091 w_k1_ADM_s100
                                 + 0.913209 w_k1_AXP_s100
                                 + 0.803495 w_k1_DIS_s100
                                 + 0.979874 w_k1_FDX_s100
                                 + 0.649077 w_k1_IBM_s100
                                 + 1.03731 w_k1_INTC_s100
                                 + 0.926155 w_k1_KO_s100
                                 + 0.732682 w_k1_MAT_s100
                                 + 0.935661 w_k1_MSFT_s100
                                 + 1.0631 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.719621 w_k1_ADM_s100
                                 + 0.70256 w_k1_AXP_s100
                                 + 0.740637 w_k1_DIS_s100
                                 + 0.707815 w_k1_FDX_s100
                                 + 1.14864 w_k1_IBM_s100
                                 + 0.738156 w_k1_INTC_s100
                                 + 0.904349 w_k1_KO_s100
                                 + 0.765383 w_k1_MAT_s100
                                 + 0.74738 w_k1_MSFT_s100
                                 + 0.905335 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.905976 w_k1_ADM_s100
                                 + 0.910498 w_k1_AXP_s100
                                 + 1.25485 w_k1_DIS_s100
                                 + 1.08602 w_k1_FDX_s100
                                 + 1.18999 w_k1_IBM_s100
                                 + 1.15773 w_k1_INTC_s100
                                 + 1.17218 w_k1_KO_s100
                                 + 0.993618 w_k1_MAT_s100
                                 + 1.03446 w_k1_MSFT_s100 + 1.125 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.682275 w_k1_ADM_s101
                                 + 0.667765 w_k1_AXP_s101
                                 + 0.578435 w_k1_DIS_s101
                                 + 0.680442 w_k1_FDX_s101
                                 + 0.724552 w_k1_IBM_s101
                                 + 0.737095 w_k1_INTC_s101
                                 + 0.7146 w_k1_KO_s101 + 0.71476 w_k1_MAT_s101
                                 + 0.808118 w_k1_MSFT_s101
                                 + 0.860657 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.951903 w_k1_ADM_s101
                                 + 1.11843 w_k1_AXP_s101
                                 + 0.987258 w_k1_DIS_s101
                                 + 1.04884 w_k1_FDX_s101
                                 + 1.06482 w_k1_IBM_s101
                                 + 1.0798 w_k1_INTC_s101 + 0.8967 w_k1_KO_s101
                                 + 1.11834 w_k1_MAT_s101
                                 + 0.997981 w_k1_MSFT_s101
                                 + 1.00549 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.4237 w_k1_ADM_s101
                                 + 1.64249 w_k1_AXP_s101
                                 + 1.85097 w_k1_DIS_s101
                                 + 1.53848 w_k1_FDX_s101
                                 + 1.05965 w_k1_IBM_s101
                                 + 1.42381 w_k1_INTC_s101
                                 + 1.38852 w_k1_KO_s101 + 1.41814 w_k1_MAT_s101
                                 + 1.56459 w_k1_MSFT_s101
                                 + 1.33557 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.872022 w_k1_ADM_s101
                                 + 0.750684 w_k1_AXP_s101
                                 + 0.801521 w_k1_DIS_s101
                                 + 0.773027 w_k1_FDX_s101
                                 + 1.17419 w_k1_IBM_s101
                                 + 0.773095 w_k1_INTC_s101
                                 + 0.913052 w_k1_KO_s101
                                 + 1.17192 w_k1_MAT_s101
                                 + 0.910839 w_k1_MSFT_s101
                                 + 0.9642 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.11977 w_k1_ADM_s101
                                 + 1.18475 w_k1_AXP_s101
                                 + 1.33521 w_k1_DIS_s101
                                 + 1.25655 w_k1_FDX_s101
                                 + 0.963907 w_k1_IBM_s101
                                 + 1.1809 w_k1_INTC_s101 + 1.21504 w_k1_KO_s101
                                 + 1.13879 w_k1_MAT_s101
                                 + 1.44057 w_k1_MSFT_s101
                                 + 1.18377 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.954421 w_k1_ADM_s101
                                 + 0.812688 w_k1_AXP_s101
                                 + 0.808378 w_k1_DIS_s101
                                 + 0.883077 w_k1_FDX_s101
                                 + 0.865339 w_k1_IBM_s101
                                 + 0.865137 w_k1_INTC_s101
                                 + 0.882549 w_k1_KO_s101
                                 + 1.10084 w_k1_MAT_s101
                                 + 1.06413 w_k1_MSFT_s101
                                 + 0.965326 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.22948 w_k1_ADM_s101
                                 + 1.2384 w_k1_AXP_s101 + 1.11323 w_k1_DIS_s101
                                 + 1.00326 w_k1_FDX_s101
                                 + 1.59683 w_k1_IBM_s101
                                 + 0.928802 w_k1_INTC_s101
                                 + 1.22123 w_k1_KO_s101 + 1.74897 w_k1_MAT_s101
                                 + 1.1047 w_k1_MSFT_s101 + 1.07258 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.936323 w_k1_ADM_s101
                                 + 0.901883 w_k1_AXP_s101
                                 + 0.94521 w_k1_DIS_s101
                                 + 1.08196 w_k1_FDX_s101
                                 + 0.61201 w_k1_IBM_s101
                                 + 0.92091 w_k1_INTC_s101
                                 + 0.827562 w_k1_KO_s101
                                 + 0.619992 w_k1_MAT_s101
                                 + 1.13236 w_k1_MSFT_s101
                                 + 0.945214 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.07797 w_k1_ADM_s102
                                 + 1.29498 w_k1_AXP_s102
                                 + 1.12561 w_k1_DIS_s102
                                 + 1.06838 w_k1_FDX_s102
                                 + 1.46992 w_k1_IBM_s102
                                 + 1.09396 w_k1_INTC_s102
                                 + 1.32034 w_k1_KO_s102 + 2.03526 w_k1_MAT_s102
                                 + 1.23338 w_k1_MSFT_s102
                                 + 1.19988 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.33155 w_k1_ADM_s102
                                 + 1.11488 w_k1_AXP_s102 + 1.4021 w_k1_DIS_s102
                                 + 1.15708 w_k1_FDX_s102
                                 + 1.26677 w_k1_IBM_s102
                                 + 0.92985 w_k1_INTC_s102
                                 + 1.43234 w_k1_KO_s102 + 1.41303 w_k1_MAT_s102
                                 + 1.12708 w_k1_MSFT_s102
                                 + 1.17358 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.05792 w_k1_ADM_s102
                                 + 0.949448 w_k1_AXP_s102
                                 + 1.11476 w_k1_DIS_s102
                                 + 1.01677 w_k1_FDX_s102
                                 + 0.952357 w_k1_IBM_s102
                                 + 0.84841 w_k1_INTC_s102
                                 + 1.14552 w_k1_KO_s102 + 1.00843 w_k1_MAT_s102
                                 + 1.00144 w_k1_MSFT_s102
                                 + 1.15939 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.808923 w_k1_ADM_s102
                                 + 0.947451 w_k1_AXP_s102
                                 + 0.972083 w_k1_DIS_s102
                                 + 0.943644 w_k1_FDX_s102
                                 + 1.11891 w_k1_IBM_s102
                                 + 0.865735 w_k1_INTC_s102
                                 + 1.03171 w_k1_KO_s102 + 1.14826 w_k1_MAT_s102
                                 + 0.963307 w_k1_MSFT_s102
                                 + 1.0222 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.878349 w_k1_ADM_s102
                                 + 0.943215 w_k1_AXP_s102
                                 + 0.688003 w_k1_DIS_s102
                                 + 0.841302 w_k1_FDX_s102
                                 + 0.824121 w_k1_IBM_s102
                                 + 0.825054 w_k1_INTC_s102
                                 + 0.812872 w_k1_KO_s102
                                 + 0.935589 w_k1_MAT_s102
                                 + 0.950574 w_k1_MSFT_s102
                                 + 0.943703 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.35168 w_k1_ADM_s102
                                 + 1.55246 w_k1_AXP_s102
                                 + 1.40922 w_k1_DIS_s102
                                 + 1.22041 w_k1_FDX_s102
                                 + 1.08034 w_k1_IBM_s102
                                 + 1.04922 w_k1_INTC_s102
                                 + 1.19215 w_k1_KO_s102 + 1.4347 w_k1_MAT_s102
                                 + 0.95965 w_k1_MSFT_s102 + 1.1888 w_k1_PG_s102
                                 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.09882 w_k1_ADM_s102
                                 + 0.999165 w_k1_AXP_s102
                                 + 0.890305 w_k1_DIS_s102
                                 + 1.00537 w_k1_FDX_s102
                                 + 0.684136 w_k1_IBM_s102
                                 + 1.01013 w_k1_INTC_s102
                                 + 0.869226 w_k1_KO_s102
                                 + 0.579124 w_k1_MAT_s102
                                 + 0.912529 w_k1_MSFT_s102
                                 + 0.957654 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.842792 w_k1_ADM_s102
                                 + 0.74292 w_k1_AXP_s102
                                 + 0.746071 w_k1_DIS_s102
                                 + 0.781078 w_k1_FDX_s102
                                 + 1.10209 w_k1_IBM_s102
                                 + 0.878853 w_k1_INTC_s102
                                 + 1.09955 w_k1_KO_s102 + 1.14769 w_k1_MAT_s102
                                 + 0.889491 w_k1_MSFT_s102
                                 + 0.943455 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.02738 w_k1_ADM_s103
                                 + 1.2371 w_k1_AXP_s103 + 1.16344 w_k1_DIS_s103
                                 + 1.18628 w_k1_FDX_s103
                                 + 0.893255 w_k1_IBM_s103
                                 + 1.22599 w_k1_INTC_s103
                                 + 1.10394 w_k1_KO_s103 + 1.01637 w_k1_MAT_s103
                                 + 1.24486 w_k1_MSFT_s103
                                 + 1.19473 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.837036 w_k1_ADM_s103
                                 + 0.955004 w_k1_AXP_s103
                                 + 1.13274 w_k1_DIS_s103 + 1.0323 w_k1_FDX_s103
                                 + 1.03547 w_k1_IBM_s103
                                 + 1.05621 w_k1_INTC_s103
                                 + 1.17716 w_k1_KO_s103
                                 + 0.981225 w_k1_MAT_s103
                                 + 1.22679 w_k1_MSFT_s103
                                 + 1.12765 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.819535 w_k1_ADM_s103
                                 + 0.814829 w_k1_AXP_s103
                                 + 0.711851 w_k1_DIS_s103
                                 + 0.733941 w_k1_FDX_s103
                                 + 1.02409 w_k1_IBM_s103
                                 + 0.728811 w_k1_INTC_s103
                                 + 0.922287 w_k1_KO_s103
                                 + 0.999178 w_k1_MAT_s103
                                 + 0.750484 w_k1_MSFT_s103
                                 + 0.977593 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.24307 w_k1_ADM_s103
                                 + 1.59305 w_k1_AXP_s103
                                 + 1.38664 w_k1_DIS_s103
                                 + 1.24572 w_k1_FDX_s103
                                 + 1.57676 w_k1_IBM_s103
                                 + 1.24465 w_k1_INTC_s103
                                 + 1.50395 w_k1_KO_s103 + 2.1038 w_k1_MAT_s103
                                 + 1.22346 w_k1_MSFT_s103 + 1.4001 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.816573 w_k1_ADM_s103
                                 + 0.914584 w_k1_AXP_s103
                                 + 1.01214 w_k1_DIS_s103
                                 + 1.04379 w_k1_FDX_s103
                                 + 0.606212 w_k1_IBM_s103
                                 + 1.18833 w_k1_INTC_s103
                                 + 0.796535 w_k1_KO_s103
                                 + 0.65068 w_k1_MAT_s103
                                 + 1.2288 w_k1_MSFT_s103
                                 + 0.968902 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.658638 w_k1_ADM_s103
                                 + 0.815206 w_k1_AXP_s103
                                 + 0.737051 w_k1_DIS_s103
                                 + 0.769985 w_k1_FDX_s103
                                 + 0.756196 w_k1_IBM_s103
                                 + 0.765075 w_k1_INTC_s103
                                 + 0.773745 w_k1_KO_s103
                                 + 0.709253 w_k1_MAT_s103
                                 + 0.879512 w_k1_MSFT_s103
                                 + 0.957781 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.726699 w_k1_ADM_s103
                                 + 0.765401 w_k1_AXP_s103
                                 + 0.577559 w_k1_DIS_s103
                                 + 0.845768 w_k1_FDX_s103
                                 + 0.563789 w_k1_IBM_s103
                                 + 0.777703 w_k1_INTC_s103
                                 + 0.64155 w_k1_KO_s103
                                 + 0.644944 w_k1_MAT_s103
                                 + 0.916798 w_k1_MSFT_s103
                                 + 0.970137 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.01154 w_k1_ADM_s103
                                 + 1.07808 w_k1_AXP_s103
                                 + 1.34599 w_k1_DIS_s103
                                 + 1.08793 w_k1_FDX_s103 + 1.2807 w_k1_IBM_s103
                                 + 1.13476 w_k1_INTC_s103
                                 + 1.46071 w_k1_KO_s103 + 1.39825 w_k1_MAT_s103
                                 + 1.46217 w_k1_MSFT_s103
                                 + 1.40309 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.625594 w_k1_ADM_s104
                                 + 0.713758 w_k1_AXP_s104
                                 + 0.513942 w_k1_DIS_s104
                                 + 0.719593 w_k1_FDX_s104
                                 + 0.774446 w_k1_IBM_s104
                                 + 0.768161 w_k1_INTC_s104
                                 + 0.744425 w_k1_KO_s104
                                 + 0.930126 w_k1_MAT_s104
                                 + 0.69934 w_k1_MSFT_s104
                                 + 0.707747 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.08568 w_k1_ADM_s104
                                 + 1.04048 w_k1_AXP_s104
                                 + 0.907302 w_k1_DIS_s104
                                 + 0.999437 w_k1_FDX_s104
                                 + 0.931708 w_k1_IBM_s104
                                 + 0.846984 w_k1_INTC_s104
                                 + 0.974314 w_k1_KO_s104
                                 + 1.13243 w_k1_MAT_s104
                                 + 0.840017 w_k1_MSFT_s104
                                 + 0.988664 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.793035 w_k1_ADM_s104
                                 + 0.965958 w_k1_AXP_s104
                                 + 1.11146 w_k1_DIS_s104
                                 + 1.08425 w_k1_FDX_s104
                                 + 0.783015 w_k1_IBM_s104
                                 + 1.00037 w_k1_INTC_s104
                                 + 1.04805 w_k1_KO_s104 + 0.83282 w_k1_MAT_s104
                                 + 1.13001 w_k1_MSFT_s104
                                 + 0.972527 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.780643 w_k1_ADM_s104
                                 + 0.924739 w_k1_AXP_s104
                                 + 0.739593 w_k1_DIS_s104
                                 + 0.905864 w_k1_FDX_s104
                                 + 0.709009 w_k1_IBM_s104
                                 + 0.784233 w_k1_INTC_s104
                                 + 0.776724 w_k1_KO_s104
                                 + 0.940593 w_k1_MAT_s104
                                 + 0.860495 w_k1_MSFT_s104
                                 + 0.802846 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.742616 w_k1_ADM_s104
                                 + 0.856797 w_k1_AXP_s104
                                 + 0.922813 w_k1_DIS_s104
                                 + 0.854831 w_k1_FDX_s104
                                 + 1.3456 w_k1_IBM_s104
                                 + 1.03436 w_k1_INTC_s104
                                 + 1.19327 w_k1_KO_s104 + 1.1833 w_k1_MAT_s104
                                 + 0.965975 w_k1_MSFT_s104
                                 + 1.00077 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.21267 w_k1_ADM_s104
                                 + 1.7088 w_k1_AXP_s104 + 1.93769 w_k1_DIS_s104
                                 + 1.62223 w_k1_FDX_s104
                                 + 1.37468 w_k1_IBM_s104
                                 + 1.17318 w_k1_INTC_s104
                                 + 1.56068 w_k1_KO_s104 + 1.85944 w_k1_MAT_s104
                                 + 1.27287 w_k1_MSFT_s104
                                 + 1.17072 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.27611 w_k1_ADM_s104
                                 + 1.34331 w_k1_AXP_s104
                                 + 1.47253 w_k1_DIS_s104 + 1.2482 w_k1_FDX_s104
                                 + 1.13827 w_k1_IBM_s104
                                 + 1.08816 w_k1_INTC_s104
                                 + 1.27456 w_k1_KO_s104 + 1.41792 w_k1_MAT_s104
                                 + 1.25248 w_k1_MSFT_s104
                                 + 1.08147 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.876214 w_k1_ADM_s104
                                 + 1.22108 w_k1_AXP_s104
                                 + 0.70807 w_k1_DIS_s104
                                 + 0.872347 w_k1_FDX_s104
                                 + 1.12709 w_k1_IBM_s104
                                 + 0.822609 w_k1_INTC_s104
                                 + 0.981746 w_k1_KO_s104
                                 + 1.51991 w_k1_MAT_s104
                                 + 0.684404 w_k1_MSFT_s104
                                 + 1.04996 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.917981 w_k1_ADM_s105
                                 + 0.924047 w_k1_AXP_s105
                                 + 0.989693 w_k1_DIS_s105
                                 + 0.98307 w_k1_FDX_s105
                                 + 1.05292 w_k1_IBM_s105
                                 + 0.91041 w_k1_INTC_s105
                                 + 0.956199 w_k1_KO_s105
                                 + 1.28532 w_k1_MAT_s105
                                 + 1.0477 w_k1_MSFT_s105
                                 + 0.968759 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.970695 w_k1_ADM_s105
                                 + 1.0502 w_k1_AXP_s105 + 1.6403 w_k1_DIS_s105
                                 + 1.2114 w_k1_FDX_s105 + 1.13459 w_k1_IBM_s105
                                 + 1.28176 w_k1_INTC_s105
                                 + 1.14756 w_k1_KO_s105
                                 + 0.892637 w_k1_MAT_s105
                                 + 1.38822 w_k1_MSFT_s105
                                 + 1.14699 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.32999 w_k1_ADM_s105
                                 + 1.58889 w_k1_AXP_s105
                                 + 1.74387 w_k1_DIS_s105
                                 + 1.49897 w_k1_FDX_s105
                                 + 1.22505 w_k1_IBM_s105
                                 + 1.24127 w_k1_INTC_s105
                                 + 1.19805 w_k1_KO_s105 + 1.26115 w_k1_MAT_s105
                                 + 1.29787 w_k1_MSFT_s105
                                 + 1.14211 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.826827 w_k1_ADM_s105
                                 + 0.886354 w_k1_AXP_s105
                                 + 0.755507 w_k1_DIS_s105
                                 + 0.850271 w_k1_FDX_s105
                                 + 1.28187 w_k1_IBM_s105
                                 + 1.04427 w_k1_INTC_s105
                                 + 1.03613 w_k1_KO_s105 + 1.0677 w_k1_MAT_s105
                                 + 0.995597 w_k1_MSFT_s105
                                 + 0.914474 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.762038 w_k1_ADM_s105
                                 + 0.725789 w_k1_AXP_s105
                                 + 0.566901 w_k1_DIS_s105
                                 + 0.760719 w_k1_FDX_s105
                                 + 0.850163 w_k1_IBM_s105
                                 + 0.917094 w_k1_INTC_s105
                                 + 0.693633 w_k1_KO_s105
                                 + 0.701613 w_k1_MAT_s105
                                 + 0.821178 w_k1_MSFT_s105
                                 + 0.799959 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.91901 w_k1_ADM_s105
                                 + 1.32659 w_k1_AXP_s105
                                 + 1.11939 w_k1_DIS_s105
                                 + 1.16711 w_k1_FDX_s105
                                 + 1.21994 w_k1_IBM_s105
                                 + 1.09254 w_k1_INTC_s105
                                 + 1.16562 w_k1_KO_s105 + 1.33474 w_k1_MAT_s105
                                 + 1.19806 w_k1_MSFT_s105
                                 + 1.13103 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.831936 w_k1_ADM_s105
                                 + 0.799629 w_k1_AXP_s105
                                 + 0.80607 w_k1_DIS_s105
                                 + 0.90379 w_k1_FDX_s105
                                 + 0.67187 w_k1_IBM_s105
                                 + 0.95494 w_k1_INTC_s105
                                 + 0.833896 w_k1_KO_s105 + 0.5545 w_k1_MAT_s105
                                 + 0.925881 w_k1_MSFT_s105
                                 + 0.910105 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.05601 w_k1_ADM_s105
                                 + 1.21276 w_k1_AXP_s105
                                 + 0.972962 w_k1_DIS_s105
                                 + 1.10526 w_k1_FDX_s105
                                 + 0.771537 w_k1_IBM_s105
                                 + 1.06048 w_k1_INTC_s105
                                 + 0.910631 w_k1_KO_s105
                                 + 0.951652 w_k1_MAT_s105
                                 + 0.978335 w_k1_MSFT_s105
                                 + 1.04386 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.52124 w_k1_ADM_s106
                                 + 1.37256 w_k1_AXP_s106
                                 + 1.56908 w_k1_DIS_s106
                                 + 1.39864 w_k1_FDX_s106
                                 + 1.07411 w_k1_IBM_s106
                                 + 1.27674 w_k1_INTC_s106 + 1.2498 w_k1_KO_s106
                                 + 1.00939 w_k1_MAT_s106
                                 + 1.2757 w_k1_MSFT_s106 + 1.13601 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.19998 w_k1_ADM_s106
                                 + 1.10628 w_k1_AXP_s106 + 1.0441 w_k1_DIS_s106
                                 + 1.0731 w_k1_FDX_s106 + 1.04752 w_k1_IBM_s106
                                 + 1.00581 w_k1_INTC_s106
                                 + 0.939703 w_k1_KO_s106
                                 + 0.941326 w_k1_MAT_s106
                                 + 1.05838 w_k1_MSFT_s106
                                 + 0.92344 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.854745 w_k1_ADM_s106
                                 + 0.878985 w_k1_AXP_s106
                                 + 0.796464 w_k1_DIS_s106
                                 + 0.830952 w_k1_FDX_s106
                                 + 0.996688 w_k1_IBM_s106
                                 + 0.811832 w_k1_INTC_s106
                                 + 0.832742 w_k1_KO_s106
                                 + 0.744479 w_k1_MAT_s106
                                 + 0.848765 w_k1_MSFT_s106
                                 + 0.764657 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.97923 w_k1_ADM_s106
                                 + 0.714793 w_k1_AXP_s106
                                 + 0.70116 w_k1_DIS_s106
                                 + 0.865657 w_k1_FDX_s106
                                 + 0.679174 w_k1_IBM_s106
                                 + 0.853635 w_k1_INTC_s106
                                 + 0.785509 w_k1_KO_s106
                                 + 0.735656 w_k1_MAT_s106
                                 + 0.948838 w_k1_MSFT_s106
                                 + 0.859971 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.07567 w_k1_ADM_s106
                                 + 0.959058 w_k1_AXP_s106
                                 + 0.986521 w_k1_DIS_s106
                                 + 0.927005 w_k1_FDX_s106
                                 + 0.878758 w_k1_IBM_s106
                                 + 0.958425 w_k1_INTC_s106
                                 + 0.822856 w_k1_KO_s106
                                 + 0.776379 w_k1_MAT_s106
                                 + 0.919459 w_k1_MSFT_s106
                                 + 0.853517 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.13621 w_k1_ADM_s106
                                 + 1.03042 w_k1_AXP_s106
                                 + 1.10424 w_k1_DIS_s106
                                 + 1.13619 w_k1_FDX_s106
                                 + 0.731929 w_k1_IBM_s106
                                 + 1.01097 w_k1_INTC_s106
                                 + 0.895437 w_k1_KO_s106
                                 + 0.707369 w_k1_MAT_s106
                                 + 1.1361 w_k1_MSFT_s106
                                 + 0.994437 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.12006 w_k1_ADM_s106
                                 + 1.22739 w_k1_AXP_s106
                                 + 1.22459 w_k1_DIS_s106
                                 + 1.06706 w_k1_FDX_s106
                                 + 1.51847 w_k1_IBM_s106
                                 + 1.08095 w_k1_INTC_s106
                                 + 1.14216 w_k1_KO_s106 + 1.65236 w_k1_MAT_s106
                                 + 1.03353 w_k1_MSFT_s106
                                 + 1.08015 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.30035 w_k1_ADM_s106
                                 + 1.60604 w_k1_AXP_s106
                                 + 1.09785 w_k1_DIS_s106 + 1.1273 w_k1_FDX_s106
                                 + 0.954175 w_k1_IBM_s106
                                 + 1.13217 w_k1_INTC_s106 + 1.0394 w_k1_KO_s106
                                 + 1.28566 w_k1_MAT_s106
                                 + 1.02133 w_k1_MSFT_s106
                                 + 0.987326 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.00854 w_k1_ADM_s107
                                 + 0.697325 w_k1_AXP_s107
                                 + 0.608308 w_k1_DIS_s107
                                 + 0.660867 w_k1_FDX_s107
                                 + 1.04987 w_k1_IBM_s107
                                 + 0.707156 w_k1_INTC_s107
                                 + 0.783682 w_k1_KO_s107
                                 + 0.901678 w_k1_MAT_s107
                                 + 0.686728 w_k1_MSFT_s107
                                 + 0.835859 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.46146 w_k1_ADM_s107
                                 + 1.11521 w_k1_AXP_s107
                                 + 1.02785 w_k1_DIS_s107 + 1.0919 w_k1_FDX_s107
                                 + 1.11449 w_k1_IBM_s107
                                 + 1.13111 w_k1_INTC_s107
                                 + 1.08085 w_k1_KO_s107 + 1.27194 w_k1_MAT_s107
                                 + 1.10791 w_k1_MSFT_s107
                                 + 1.10334 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.36245 w_k1_ADM_s107
                                 + 1.16367 w_k1_AXP_s107
                                 + 1.02782 w_k1_DIS_s107
                                 + 1.12893 w_k1_FDX_s107
                                 + 0.805231 w_k1_IBM_s107
                                 + 0.980742 w_k1_INTC_s107
                                 + 0.929933 w_k1_KO_s107
                                 + 0.932292 w_k1_MAT_s107
                                 + 1.02216 w_k1_MSFT_s107
                                 + 1.01215 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.09839 w_k1_ADM_s107
                                 + 0.848186 w_k1_AXP_s107
                                 + 0.599046 w_k1_DIS_s107
                                 + 0.825613 w_k1_FDX_s107
                                 + 0.742073 w_k1_IBM_s107
                                 + 0.928035 w_k1_INTC_s107
                                 + 0.805684 w_k1_KO_s107
                                 + 0.664899 w_k1_MAT_s107
                                 + 0.85759 w_k1_MSFT_s107
                                 + 0.920511 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.50156 w_k1_ADM_s107
                                 + 1.10731 w_k1_AXP_s107
                                 + 1.07941 w_k1_DIS_s107
                                 + 1.13968 w_k1_FDX_s107
                                 + 0.958019 w_k1_IBM_s107
                                 + 1.11679 w_k1_INTC_s107
                                 + 1.19331 w_k1_KO_s107 + 1.13294 w_k1_MAT_s107
                                 + 1.06404 w_k1_MSFT_s107
                                 + 1.14282 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.14485 w_k1_ADM_s107
                                 + 0.839159 w_k1_AXP_s107
                                 + 0.910951 w_k1_DIS_s107
                                 + 0.880919 w_k1_FDX_s107
                                 + 0.912826 w_k1_IBM_s107
                                 + 0.88833 w_k1_INTC_s107
                                 + 0.954737 w_k1_KO_s107
                                 + 0.693472 w_k1_MAT_s107
                                 + 0.875994 w_k1_MSFT_s107
                                 + 0.933857 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.67578 w_k1_ADM_s107
                                 + 1.50731 w_k1_AXP_s107
                                 + 1.71476 w_k1_DIS_s107
                                 + 1.44388 w_k1_FDX_s107
                                 + 1.04659 w_k1_IBM_s107
                                 + 1.31536 w_k1_INTC_s107
                                 + 1.40276 w_k1_KO_s107 + 1.26998 w_k1_MAT_s107
                                 + 1.4271 w_k1_MSFT_s107 + 1.39115 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.41041 w_k1_ADM_s107
                                 + 1.26757 w_k1_AXP_s107
                                 + 1.18059 w_k1_DIS_s107
                                 + 1.01029 w_k1_FDX_s107
                                 + 1.41879 w_k1_IBM_s107
                                 + 0.953089 w_k1_INTC_s107
                                 + 1.37254 w_k1_KO_s107 + 1.61594 w_k1_MAT_s107
                                 + 1.05535 w_k1_MSFT_s107 + 1.0892 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.05555 w_k1_ADM_s110
                                 + 1.60814 w_k1_AXP_s110
                                 + 1.62645 w_k1_DIS_s110
                                 + 1.38089 w_k1_FDX_s110
                                 + 1.20986 w_k1_IBM_s110
                                 + 1.28012 w_k1_INTC_s110
                                 + 1.51241 w_k1_KO_s110 + 1.39284 w_k1_MAT_s110
                                 + 1.4319 w_k1_MSFT_s110 + 1.13985 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.03594 w_k1_ADM_s110
                                 + 0.973017 w_k1_AXP_s110
                                 + 1.41626 w_k1_DIS_s110
                                 + 1.12166 w_k1_FDX_s110
                                 + 1.24805 w_k1_IBM_s110
                                 + 0.821569 w_k1_INTC_s110
                                 + 1.42425 w_k1_KO_s110 + 1.69827 w_k1_MAT_s110
                                 + 1.18473 w_k1_MSFT_s110
                                 + 1.20353 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.4486 w_k1_ADM_s110
                                 + 1.26497 w_k1_AXP_s110
                                 + 1.29669 w_k1_DIS_s110
                                 + 1.13433 w_k1_FDX_s110
                                 + 0.976744 w_k1_IBM_s110
                                 + 0.921357 w_k1_INTC_s110
                                 + 1.10603 w_k1_KO_s110 + 1.28149 w_k1_MAT_s110
                                 + 1.23081 w_k1_MSFT_s110
                                 + 1.11634 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.954637 w_k1_ADM_s110
                                 + 0.994244 w_k1_AXP_s110
                                 + 1.03366 w_k1_DIS_s110
                                 + 1.07427 w_k1_FDX_s110
                                 + 0.871849 w_k1_IBM_s110
                                 + 0.929039 w_k1_INTC_s110
                                 + 0.916837 w_k1_KO_s110
                                 + 1.16184 w_k1_MAT_s110
                                 + 1.04297 w_k1_MSFT_s110
                                 + 0.976709 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.9053 w_k1_ADM_s110
                                 + 0.855595 w_k1_AXP_s110
                                 + 0.790175 w_k1_DIS_s110
                                 + 0.807704 w_k1_FDX_s110
                                 + 0.992394 w_k1_IBM_s110
                                 + 0.798324 w_k1_INTC_s110
                                 + 0.980673 w_k1_KO_s110
                                 + 0.983696 w_k1_MAT_s110
                                 + 0.863715 w_k1_MSFT_s110
                                 + 0.883226 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.976895 w_k1_ADM_s110
                                 + 1.25817 w_k1_AXP_s110
                                 + 1.09392 w_k1_DIS_s110
                                 + 1.01614 w_k1_FDX_s110
                                 + 1.33538 w_k1_IBM_s110
                                 + 1.00441 w_k1_INTC_s110
                                 + 1.20086 w_k1_KO_s110 + 1.53416 w_k1_MAT_s110
                                 + 1.01998 w_k1_MSFT_s110
                                 + 0.963806 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.861773 w_k1_ADM_s110
                                 + 0.758985 w_k1_AXP_s110
                                 + 0.65781 w_k1_DIS_s110
                                 + 0.782964 w_k1_FDX_s110
                                 + 0.661603 w_k1_IBM_s110
                                 + 0.829207 w_k1_INTC_s110
                                 + 0.828823 w_k1_KO_s110
                                 + 0.802083 w_k1_MAT_s110
                                 + 0.942353 w_k1_MSFT_s110
                                 + 0.802644 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.89565 w_k1_ADM_s110
                                 + 1.18901 w_k1_AXP_s110
                                 + 1.05304 w_k1_DIS_s110
                                 + 1.02171 w_k1_FDX_s110
                                 + 0.943385 w_k1_IBM_s110
                                 + 0.955366 w_k1_INTC_s110
                                 + 0.972595 w_k1_KO_s110
                                 + 1.28679 w_k1_MAT_s110
                                 + 1.11275 w_k1_MSFT_s110
                                 + 0.982294 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.17851 w_k1_ADM_s111
                                 + 1.02829 w_k1_AXP_s111
                                 + 0.761452 w_k1_DIS_s111
                                 + 1.01093 w_k1_FDX_s111
                                 + 0.805952 w_k1_IBM_s111
                                 + 0.955031 w_k1_INTC_s111
                                 + 0.836125 w_k1_KO_s111
                                 + 0.629538 w_k1_MAT_s111
                                 + 0.845354 w_k1_MSFT_s111
                                 + 0.887878 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.986396 w_k1_ADM_s111
                                 + 1.09808 w_k1_AXP_s111
                                 + 0.971784 w_k1_DIS_s111
                                 + 1.07946 w_k1_FDX_s111
                                 + 1.17999 w_k1_IBM_s111
                                 + 1.16377 w_k1_INTC_s111
                                 + 1.00828 w_k1_KO_s111 + 1.31203 w_k1_MAT_s111
                                 + 0.917168 w_k1_MSFT_s111
                                 + 1.09634 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.86773 w_k1_ADM_s111
                                 + 0.764729 w_k1_AXP_s111
                                 + 0.592089 w_k1_DIS_s111
                                 + 0.749353 w_k1_FDX_s111
                                 + 1.18515 w_k1_IBM_s111
                                 + 0.822857 w_k1_INTC_s111
                                 + 0.82426 w_k1_KO_s111
                                 + 0.831236 w_k1_MAT_s111
                                 + 0.64487 w_k1_MSFT_s111
                                 + 0.773825 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.923676 w_k1_ADM_s111
                                 + 0.981105 w_k1_AXP_s111
                                 + 0.936539 w_k1_DIS_s111
                                 + 0.948923 w_k1_FDX_s111
                                 + 1.42727 w_k1_IBM_s111
                                 + 0.940463 w_k1_INTC_s111
                                 + 1.1052 w_k1_KO_s111 + 1.41439 w_k1_MAT_s111
                                 + 0.82696 w_k1_MSFT_s111
                                 + 1.11467 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.53848 w_k1_ADM_s111
                                 + 1.26628 w_k1_AXP_s111
                                 + 1.37753 w_k1_DIS_s111
                                 + 1.40588 w_k1_FDX_s111
                                 + 1.01415 w_k1_IBM_s111
                                 + 1.16675 w_k1_INTC_s111 + 1.0335 w_k1_KO_s111
                                 + 1.00476 w_k1_MAT_s111
                                 + 1.18963 w_k1_MSFT_s111
                                 + 1.16219 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.03246 w_k1_ADM_s111
                                 + 1.08323 w_k1_AXP_s111 + 1.5718 w_k1_DIS_s111
                                 + 1.2855 w_k1_FDX_s111 + 1.39194 w_k1_IBM_s111
                                 + 1.49476 w_k1_INTC_s111
                                 + 1.21814 w_k1_KO_s111 + 1.11642 w_k1_MAT_s111
                                 + 1.20426 w_k1_MSFT_s111 + 1.2562 w_k1_PG_s111
                                 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.717536 w_k1_ADM_s111
                                 + 0.640645 w_k1_AXP_s111
                                 + 0.733801 w_k1_DIS_s111
                                 + 0.785218 w_k1_FDX_s111
                                 + 0.887906 w_k1_IBM_s111
                                 + 0.844514 w_k1_INTC_s111
                                 + 0.789188 w_k1_KO_s111
                                 + 0.625075 w_k1_MAT_s111
                                 + 0.73747 w_k1_MSFT_s111
                                 + 0.847382 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.866466 w_k1_ADM_s111
                                 + 0.730762 w_k1_AXP_s111
                                 + 1.03868 w_k1_DIS_s111
                                 + 1.02179 w_k1_FDX_s111
                                 + 0.861653 w_k1_IBM_s111
                                 + 1.14469 w_k1_INTC_s111
                                 + 0.94121 w_k1_KO_s111
                                 + 0.664954 w_k1_MAT_s111
                                 + 0.959303 w_k1_MSFT_s111
                                 + 0.917082 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.67401 w_k1_ADM_s112
                                 + 0.644356 w_k1_AXP_s112
                                 + 0.672507 w_k1_DIS_s112
                                 + 0.780235 w_k1_FDX_s112
                                 + 0.754637 w_k1_IBM_s112
                                 + 0.730171 w_k1_INTC_s112
                                 + 0.675275 w_k1_KO_s112
                                 + 0.773548 w_k1_MAT_s112
                                 + 0.821374 w_k1_MSFT_s112
                                 + 0.884627 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.820277 w_k1_ADM_s112
                                 + 0.999026 w_k1_AXP_s112
                                 + 0.74733 w_k1_DIS_s112
                                 + 0.923062 w_k1_FDX_s112
                                 + 0.896566 w_k1_IBM_s112
                                 + 0.914462 w_k1_INTC_s112
                                 + 0.732236 w_k1_KO_s112
                                 + 0.936502 w_k1_MAT_s112
                                 + 0.89852 w_k1_MSFT_s112
                                 + 0.777893 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.13846 w_k1_ADM_s112
                                 + 0.973241 w_k1_AXP_s112
                                 + 0.945311 w_k1_DIS_s112
                                 + 0.996566 w_k1_FDX_s112
                                 + 0.952699 w_k1_IBM_s112
                                 + 0.990156 w_k1_INTC_s112
                                 + 1.00752 w_k1_KO_s112 + 1.04252 w_k1_MAT_s112
                                 + 1.00132 w_k1_MSFT_s112
                                 + 0.960964 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.03905 w_k1_ADM_s112
                                 + 1.21679 w_k1_AXP_s112
                                 + 1.18439 w_k1_DIS_s112
                                 + 1.20494 w_k1_FDX_s112
                                 + 1.13305 w_k1_IBM_s112
                                 + 1.0809 w_k1_INTC_s112 + 1.03727 w_k1_KO_s112
                                 + 1.05436 w_k1_MAT_s112
                                 + 1.13498 w_k1_MSFT_s112 + 1.0642 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.16752 w_k1_ADM_s112
                                 + 1.37179 w_k1_AXP_s112
                                 + 1.77468 w_k1_DIS_s112
                                 + 1.31164 w_k1_FDX_s112
                                 + 1.47592 w_k1_IBM_s112
                                 + 1.37436 w_k1_INTC_s112
                                 + 1.50004 w_k1_KO_s112 + 1.71607 w_k1_MAT_s112
                                 + 1.49184 w_k1_MSFT_s112
                                 + 1.39036 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.898518 w_k1_ADM_s112
                                 + 0.803807 w_k1_AXP_s112
                                 + 0.841755 w_k1_DIS_s112
                                 + 0.893237 w_k1_FDX_s112
                                 + 0.903459 w_k1_IBM_s112
                                 + 0.991119 w_k1_INTC_s112
                                 + 0.852676 w_k1_KO_s112
                                 + 1.11499 w_k1_MAT_s112
                                 + 1.08334 w_k1_MSFT_s112
                                 + 0.933336 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.804195 w_k1_ADM_s112
                                 + 0.962845 w_k1_AXP_s112
                                 + 0.933264 w_k1_DIS_s112
                                 + 0.846864 w_k1_FDX_s112
                                 + 1.37529 w_k1_IBM_s112
                                 + 0.991034 w_k1_INTC_s112
                                 + 0.979186 w_k1_KO_s112
                                 + 1.03699 w_k1_MAT_s112
                                 + 0.989192 w_k1_MSFT_s112
                                 + 0.828873 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.848872 w_k1_ADM_s112
                                 + 0.770369 w_k1_AXP_s112
                                 + 1.24246 w_k1_DIS_s112
                                 + 0.965693 w_k1_FDX_s112
                                 + 1.14648 w_k1_IBM_s112
                                 + 0.997146 w_k1_INTC_s112
                                 + 1.04004 w_k1_KO_s112
                                 + 0.933346 w_k1_MAT_s112
                                 + 1.07968 w_k1_MSFT_s112
                                 + 0.987686 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.41316 w_k1_ADM_s113
                                 + 1.22333 w_k1_AXP_s113
                                 + 1.37351 w_k1_DIS_s113
                                 + 1.36846 w_k1_FDX_s113
                                 + 1.10569 w_k1_IBM_s113
                                 + 1.39889 w_k1_INTC_s113
                                 + 1.14035 w_k1_KO_s113 + 1.05457 w_k1_MAT_s113
                                 + 1.1934 w_k1_MSFT_s113 + 1.23372 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.39937 w_k1_ADM_s113
                                 + 1.39703 w_k1_AXP_s113
                                 + 1.33869 w_k1_DIS_s113
                                 + 1.13407 w_k1_FDX_s113
                                 + 1.57087 w_k1_IBM_s113
                                 + 1.24307 w_k1_INTC_s113
                                 + 1.48052 w_k1_KO_s113 + 1.68677 w_k1_MAT_s113
                                 + 1.17324 w_k1_MSFT_s113
                                 + 1.23584 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.0711 w_k1_ADM_s113
                                 + 1.1805 w_k1_AXP_s113 + 1.25006 w_k1_DIS_s113
                                 + 1.1872 w_k1_FDX_s113
                                 + 0.846231 w_k1_IBM_s113
                                 + 1.28772 w_k1_INTC_s113
                                 + 1.06637 w_k1_KO_s113
                                 + 0.736859 w_k1_MAT_s113
                                 + 1.11199 w_k1_MSFT_s113
                                 + 1.01741 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.16343 w_k1_ADM_s113
                                 + 1.09222 w_k1_AXP_s113
                                 + 0.976596 w_k1_DIS_s113
                                 + 1.06683 w_k1_FDX_s113
                                 + 0.794781 w_k1_IBM_s113
                                 + 0.953575 w_k1_INTC_s113
                                 + 0.949896 w_k1_KO_s113
                                 + 0.868157 w_k1_MAT_s113
                                 + 0.943403 w_k1_MSFT_s113
                                 + 0.907543 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.757303 w_k1_ADM_s113
                                 + 0.813916 w_k1_AXP_s113
                                 + 0.83941 w_k1_DIS_s113
                                 + 0.871867 w_k1_FDX_s113
                                 + 0.852674 w_k1_IBM_s113
                                 + 1.08386 w_k1_INTC_s113
                                 + 1.00117 w_k1_KO_s113
                                 + 0.867264 w_k1_MAT_s113
                                 + 0.870808 w_k1_MSFT_s113
                                 + 0.924466 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.833178 w_k1_ADM_s113
                                 + 0.643413 w_k1_AXP_s113
                                 + 0.602827 w_k1_DIS_s113
                                 + 0.706257 w_k1_FDX_s113
                                 + 0.91613 w_k1_IBM_s113
                                 + 0.819932 w_k1_INTC_s113
                                 + 0.864428 w_k1_KO_s113
                                 + 0.847747 w_k1_MAT_s113
                                 + 0.767711 w_k1_MSFT_s113
                                 + 0.826999 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.960024 w_k1_ADM_s113
                                 + 1.00566 w_k1_AXP_s113
                                 + 1.07529 w_k1_DIS_s113
                                 + 1.01993 w_k1_FDX_s113 + 1.2325 w_k1_IBM_s113
                                 + 1.45682 w_k1_INTC_s113 + 1.2484 w_k1_KO_s113
                                 + 0.968606 w_k1_MAT_s113
                                 + 1.1027 w_k1_MSFT_s113 + 1.11244 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.26167 w_k1_ADM_s113
                                 + 1.36289 w_k1_AXP_s113 + 1.7872 w_k1_DIS_s113
                                 + 1.41678 w_k1_FDX_s113
                                 + 1.48733 w_k1_IBM_s113
                                 + 1.66946 w_k1_INTC_s113
                                 + 1.33788 w_k1_KO_s113 + 1.41116 w_k1_MAT_s113
                                 + 1.54466 w_k1_MSFT_s113
                                 + 1.22345 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.98741 w_k1_ADM_s114
                                 + 0.94851 w_k1_AXP_s114
                                 + 1.10176 w_k1_DIS_s114
                                 + 1.11775 w_k1_FDX_s114
                                 + 1.04407 w_k1_IBM_s114
                                 + 1.19941 w_k1_INTC_s114
                                 + 1.15502 w_k1_KO_s114 + 1.13383 w_k1_MAT_s114
                                 + 1.13024 w_k1_MSFT_s114
                                 + 1.10407 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.776975 w_k1_ADM_s114
                                 + 0.585538 w_k1_AXP_s114
                                 + 0.613025 w_k1_DIS_s114
                                 + 0.767859 w_k1_FDX_s114
                                 + 0.84413 w_k1_IBM_s114
                                 + 0.781048 w_k1_INTC_s114
                                 + 0.799255 w_k1_KO_s114
                                 + 1.18407 w_k1_MAT_s114
                                 + 0.791584 w_k1_MSFT_s114
                                 + 0.908677 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.05253 w_k1_ADM_s114
                                 + 1.01642 w_k1_AXP_s114
                                 + 1.19062 w_k1_DIS_s114 + 1.2229 w_k1_FDX_s114
                                 + 0.863153 w_k1_IBM_s114
                                 + 1.0566 w_k1_INTC_s114 + 1.05309 w_k1_KO_s114
                                 + 0.780827 w_k1_MAT_s114
                                 + 1.14329 w_k1_MSFT_s114
                                 + 1.19729 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.885078 w_k1_ADM_s114
                                 + 0.843857 w_k1_AXP_s114
                                 + 0.778605 w_k1_DIS_s114
                                 + 0.909196 w_k1_FDX_s114
                                 + 0.694353 w_k1_IBM_s114
                                 + 0.862057 w_k1_INTC_s114
                                 + 0.729658 w_k1_KO_s114
                                 + 0.954492 w_k1_MAT_s114
                                 + 0.79389 w_k1_MSFT_s114
                                 + 0.903137 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.03893 w_k1_ADM_s114
                                 + 1.04784 w_k1_AXP_s114
                                 + 1.06881 w_k1_DIS_s114
                                 + 1.01383 w_k1_FDX_s114
                                 + 1.49189 w_k1_IBM_s114
                                 + 0.786746 w_k1_INTC_s114
                                 + 1.2885 w_k1_KO_s114 + 1.82877 w_k1_MAT_s114
                                 + 1.02964 w_k1_MSFT_s114
                                 + 1.28622 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.624474 w_k1_ADM_s114
                                 + 0.584671 w_k1_AXP_s114
                                 + 0.532745 w_k1_DIS_s114
                                 + 0.654101 w_k1_FDX_s114
                                 + 0.884485 w_k1_IBM_s114
                                 + 0.613907 w_k1_INTC_s114
                                 + 0.710576 w_k1_KO_s114
                                 + 1.02326 w_k1_MAT_s114
                                 + 0.636495 w_k1_MSFT_s114
                                 + 0.773302 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.3272 w_k1_ADM_s114
                                 + 1.30398 w_k1_AXP_s114 + 1.6118 w_k1_DIS_s114
                                 + 1.37976 w_k1_FDX_s114
                                 + 1.45156 w_k1_IBM_s114
                                 + 1.22418 w_k1_INTC_s114 + 1.4339 w_k1_KO_s114
                                 + 1.46733 w_k1_MAT_s114
                                 + 1.32732 w_k1_MSFT_s114
                                 + 1.44844 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.968607 w_k1_ADM_s114
                                 + 0.877285 w_k1_AXP_s114
                                 + 0.829117 w_k1_DIS_s114
                                 + 0.91775 w_k1_FDX_s114
                                 + 1.15095 w_k1_IBM_s114
                                 + 0.792829 w_k1_INTC_s114
                                 + 1.01978 w_k1_KO_s114 + 1.31831 w_k1_MAT_s114
                                 + 0.924285 w_k1_MSFT_s114
                                 + 1.04097 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.45989 w_k1_ADM_s115
                                 + 1.565 w_k1_AXP_s115 + 1.37994 w_k1_DIS_s115
                                 + 1.3847 w_k1_FDX_s115 + 1.05131 w_k1_IBM_s115
                                 + 1.22211 w_k1_INTC_s115
                                 + 1.28848 w_k1_KO_s115 + 1.28467 w_k1_MAT_s115
                                 + 1.21078 w_k1_MSFT_s115
                                 + 1.23341 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.03869 w_k1_ADM_s115
                                 + 0.998158 w_k1_AXP_s115
                                 + 1.17341 w_k1_DIS_s115
                                 + 1.05705 w_k1_FDX_s115
                                 + 0.932762 w_k1_IBM_s115
                                 + 0.929486 w_k1_INTC_s115
                                 + 1.19386 w_k1_KO_s115 + 1.34648 w_k1_MAT_s115
                                 + 1.14739 w_k1_MSFT_s115
                                 + 1.11549 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.836683 w_k1_ADM_s115
                                 + 0.854025 w_k1_AXP_s115
                                 + 1.04228 w_k1_DIS_s115
                                 + 0.872988 w_k1_FDX_s115
                                 + 1.47831 w_k1_IBM_s115
                                 + 1.21502 w_k1_INTC_s115
                                 + 1.31672 w_k1_KO_s115 + 1.32234 w_k1_MAT_s115
                                 + 0.94473 w_k1_MSFT_s115 + 1.0473 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.29924 w_k1_ADM_s115
                                 + 0.964025 w_k1_AXP_s115
                                 + 1.14986 w_k1_DIS_s115
                                 + 1.17645 w_k1_FDX_s115
                                 + 0.84246 w_k1_IBM_s115
                                 + 1.14738 w_k1_INTC_s115
                                 + 1.05429 w_k1_KO_s115
                                 + 0.800862 w_k1_MAT_s115
                                 + 1.15582 w_k1_MSFT_s115
                                 + 1.08604 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.37945 w_k1_ADM_s115
                                 + 1.41195 w_k1_AXP_s115
                                 + 1.54948 w_k1_DIS_s115
                                 + 1.24201 w_k1_FDX_s115
                                 + 1.51815 w_k1_IBM_s115
                                 + 0.976861 w_k1_INTC_s115
                                 + 1.46524 w_k1_KO_s115 + 1.30669 w_k1_MAT_s115
                                 + 1.06333 w_k1_MSFT_s115
                                 + 1.23908 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.847796 w_k1_ADM_s115
                                 + 0.779286 w_k1_AXP_s115
                                 + 0.717694 w_k1_DIS_s115
                                 + 0.803878 w_k1_FDX_s115
                                 + 0.844878 w_k1_IBM_s115
                                 + 0.89123 w_k1_INTC_s115
                                 + 0.928031 w_k1_KO_s115
                                 + 0.677498 w_k1_MAT_s115
                                 + 0.84956 w_k1_MSFT_s115
                                 + 0.868074 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.18917 w_k1_ADM_s115
                                 + 1.23105 w_k1_AXP_s115
                                 + 0.965932 w_k1_DIS_s115
                                 + 1.1612 w_k1_FDX_s115
                                 + 0.718969 w_k1_IBM_s115
                                 + 1.00499 w_k1_INTC_s115
                                 + 0.932057 w_k1_KO_s115
                                 + 1.05958 w_k1_MAT_s115
                                 + 1.06715 w_k1_MSFT_s115
                                 + 1.06593 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.01022 w_k1_ADM_s115
                                 + 1.1047 w_k1_AXP_s115
                                 + 0.929283 w_k1_DIS_s115
                                 + 1.02093 w_k1_FDX_s115
                                 + 1.22786 w_k1_IBM_s115
                                 + 0.951806 w_k1_INTC_s115
                                 + 1.02731 w_k1_KO_s115 + 1.12216 w_k1_MAT_s115
                                 + 0.985668 w_k1_MSFT_s115
                                 + 1.04069 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.761561 w_k1_ADM_s116
                                 + 0.815367 w_k1_AXP_s116
                                 + 0.881112 w_k1_DIS_s116
                                 + 0.920828 w_k1_FDX_s116
                                 + 0.908114 w_k1_IBM_s116
                                 + 1.02632 w_k1_INTC_s116
                                 + 0.950507 w_k1_KO_s116
                                 + 0.934928 w_k1_MAT_s116
                                 + 1.01598 w_k1_MSFT_s116
                                 + 0.874062 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.898187 w_k1_ADM_s116
                                 + 1.06541 w_k1_AXP_s116
                                 + 0.928854 w_k1_DIS_s116
                                 + 1.03967 w_k1_FDX_s116
                                 + 0.785739 w_k1_IBM_s116
                                 + 1.06695 w_k1_INTC_s116
                                 + 0.840475 w_k1_KO_s116
                                 + 0.889003 w_k1_MAT_s116
                                 + 1.06266 w_k1_MSFT_s116
                                 + 0.933541 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.998139 w_k1_ADM_s116
                                 + 1.10982 w_k1_AXP_s116
                                 + 1.14331 w_k1_DIS_s116
                                 + 1.01009 w_k1_FDX_s116
                                 + 1.16812 w_k1_IBM_s116
                                 + 1.01515 w_k1_INTC_s116
                                 + 1.10152 w_k1_KO_s116 + 1.65936 w_k1_MAT_s116
                                 + 1.16204 w_k1_MSFT_s116
                                 + 1.05394 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.17185 w_k1_ADM_s116
                                 + 1.07861 w_k1_AXP_s116
                                 + 1.32865 w_k1_DIS_s116
                                 + 1.11153 w_k1_FDX_s116
                                 + 1.47625 w_k1_IBM_s116
                                 + 1.19998 w_k1_INTC_s116 + 1.3216 w_k1_KO_s116
                                 + 1.59233 w_k1_MAT_s116
                                 + 1.22807 w_k1_MSFT_s116
                                 + 1.03439 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.44174 w_k1_ADM_s116
                                 + 1.35888 w_k1_AXP_s116
                                 + 1.67464 w_k1_DIS_s116
                                 + 1.36173 w_k1_FDX_s116
                                 + 1.14648 w_k1_IBM_s116
                                 + 1.1694 w_k1_INTC_s116 + 1.28984 w_k1_KO_s116
                                 + 1.69609 w_k1_MAT_s116
                                 + 1.25477 w_k1_MSFT_s116
                                 + 1.26751 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.719421 w_k1_ADM_s116
                                 + 0.670126 w_k1_AXP_s116
                                 + 0.669443 w_k1_DIS_s116
                                 + 0.754964 w_k1_FDX_s116
                                 + 0.832458 w_k1_IBM_s116
                                 + 0.681127 w_k1_INTC_s116
                                 + 0.744635 w_k1_KO_s116
                                 + 0.894441 w_k1_MAT_s116
                                 + 0.744772 w_k1_MSFT_s116
                                 + 0.740406 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.21169 w_k1_ADM_s116
                                 + 1.21547 w_k1_AXP_s116
                                 + 1.09954 w_k1_DIS_s116
                                 + 1.19013 w_k1_FDX_s116
                                 + 0.687817 w_k1_IBM_s116
                                 + 1.21955 w_k1_INTC_s116
                                 + 0.843592 w_k1_KO_s116
                                 + 0.965898 w_k1_MAT_s116
                                 + 1.21964 w_k1_MSFT_s116
                                 + 0.961438 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.863919 w_k1_ADM_s116
                                 + 1.01709 w_k1_AXP_s116
                                 + 1.33375 w_k1_DIS_s116
                                 + 1.01222 w_k1_FDX_s116
                                 + 1.31641 w_k1_IBM_s116
                                 + 1.09795 w_k1_INTC_s116
                                 + 1.32754 w_k1_KO_s116 + 1.65439 w_k1_MAT_s116
                                 + 1.13737 w_k1_MSFT_s116
                                 + 1.10295 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.10548 w_k1_ADM_s117
                                 + 1.07047 w_k1_AXP_s117
                                 + 1.29485 w_k1_DIS_s117 + 1.2044 w_k1_FDX_s117
                                 + 0.797215 w_k1_IBM_s117
                                 + 1.2789 w_k1_INTC_s117 + 1.13854 w_k1_KO_s117
                                 + 0.986464 w_k1_MAT_s117
                                 + 1.19186 w_k1_MSFT_s117
                                 + 0.976324 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.54453 w_k1_ADM_s117
                                 + 1.44618 w_k1_AXP_s117
                                 + 2.02497 w_k1_DIS_s117
                                 + 1.63947 w_k1_FDX_s117
                                 + 1.38909 w_k1_IBM_s117
                                 + 1.58342 w_k1_INTC_s117
                                 + 1.81858 w_k1_KO_s117 + 1.76308 w_k1_MAT_s117
                                 + 1.71648 w_k1_MSFT_s117
                                 + 1.28713 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.06885 w_k1_ADM_s117
                                 + 1.17567 w_k1_AXP_s117
                                 + 1.22386 w_k1_DIS_s117
                                 + 1.07982 w_k1_FDX_s117
                                 + 1.23507 w_k1_IBM_s117
                                 + 1.19246 w_k1_INTC_s117
                                 + 1.38983 w_k1_KO_s117 + 1.70836 w_k1_MAT_s117
                                 + 1.24579 w_k1_MSFT_s117 + 1.067 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.813645 w_k1_ADM_s117
                                 + 0.757152 w_k1_AXP_s117
                                 + 0.737877 w_k1_DIS_s117
                                 + 0.755851 w_k1_FDX_s117
                                 + 0.877813 w_k1_IBM_s117
                                 + 0.699745 w_k1_INTC_s117
                                 + 0.76956 w_k1_KO_s117 + 1.17071 w_k1_MAT_s117
                                 + 0.771307 w_k1_MSFT_s117
                                 + 0.818688 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.760549 w_k1_ADM_s117
                                 + 0.835148 w_k1_AXP_s117
                                 + 1.00687 w_k1_DIS_s117
                                 + 0.805678 w_k1_FDX_s117
                                 + 1.24529 w_k1_IBM_s117
                                 + 0.833266 w_k1_INTC_s117
                                 + 1.13593 w_k1_KO_s117 + 1.33846 w_k1_MAT_s117
                                 + 0.912086 w_k1_MSFT_s117
                                 + 0.859519 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.790706 w_k1_ADM_s117
                                 + 0.74426 w_k1_AXP_s117
                                 + 0.692074 w_k1_DIS_s117
                                 + 0.832295 w_k1_FDX_s117
                                 + 0.678566 w_k1_IBM_s117
                                 + 0.873223 w_k1_INTC_s117
                                 + 0.738515 w_k1_KO_s117
                                 + 0.741646 w_k1_MAT_s117
                                 + 0.771592 w_k1_MSFT_s117
                                 + 0.736091 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.885845 w_k1_ADM_s117
                                 + 0.697791 w_k1_AXP_s117
                                 + 0.946563 w_k1_DIS_s117
                                 + 0.92931 w_k1_FDX_s117
                                 + 0.855014 w_k1_IBM_s117
                                 + 1.21461 w_k1_INTC_s117
                                 + 0.95228 w_k1_KO_s117
                                 + 0.813457 w_k1_MAT_s117
                                 + 1.10016 w_k1_MSFT_s117
                                 + 0.908223 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.879338 w_k1_ADM_s117
                                 + 0.724957 w_k1_AXP_s117
                                 + 1.05092 w_k1_DIS_s117
                                 + 0.87678 w_k1_FDX_s117
                                 + 1.06936 w_k1_IBM_s117
                                 + 1.11555 w_k1_INTC_s117
                                 + 1.11965 w_k1_KO_s117 + 1.04584 w_k1_MAT_s117
                                 + 1.12752 w_k1_MSFT_s117
                                 + 0.989855 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00813452536465768 w_k1_ADM_s000
                                 + 0.00777239799025953 w_k1_ADM_s001
                                 + 0.00635244331842717 w_k1_ADM_s002
                                 + 0.0093265143136016 w_k1_ADM_s003
                                 + 0.02003076077928 w_k1_ADM_s004
                                 + 0.0065845749534874 w_k1_ADM_s005
                                 + 0.0140008382123684 w_k1_ADM_s006
                                 + 0.00763745563809746 w_k1_ADM_s007
                                 + 0.0391809034891857 w_k1_ADM_s010
                                 + 0.0102975468968573 w_k1_ADM_s011
                                 + 0.0101585537547086 w_k1_ADM_s012
                                 + 0.0220048434215085 w_k1_ADM_s013
                                 + 0.0375979817173118 w_k1_ADM_s014
                                 + 0.0208845829856313 w_k1_ADM_s015
                                 + 0.0205759945291747 w_k1_ADM_s016
                                 + 0.0283619807411023 w_k1_ADM_s017
                                 + 0.0379103553035172 w_k1_ADM_s100
                                 + 0.140388132445269 w_k1_ADM_s101
                                 + 0.0504445084316177 w_k1_ADM_s102
                                 + 0.0292641627801572 w_k1_ADM_s103
                                 + 0.0472826163541932 w_k1_ADM_s104
                                 + 0.0326117052603508 w_k1_ADM_s105
                                 + 0.0654447310767447 w_k1_ADM_s106
                                 + 0.135930565344373 w_k1_ADM_s107
                                 + 0.0285662784172093 w_k1_ADM_s110
                                 + 0.0311999809164368 w_k1_ADM_s111
                                 + 0.0282400893776891 w_k1_ADM_s112
                                 + 0.0253531544060355 w_k1_ADM_s113
                                 + 0.0533097602976158 w_k1_ADM_s114
                                 + 0.0139240206756169 w_k1_ADM_s115
                                 + 0.0209540218212503 w_k1_ADM_s116
                                 + 0.0243981486706733 w_k1_ADM_s117
                                 + 0.00710455750383476 w_k1_AXP_s000
                                 + 0.00740685371393026 w_k1_AXP_s001
                                 + 0.00660847626785598 w_k1_AXP_s002
                                 + 0.010247390040275 w_k1_AXP_s003
                                 + 0.0222943713888485 w_k1_AXP_s004
                                 + 0.00687177119472211 w_k1_AXP_s005
                                 + 0.0125217984046426 w_k1_AXP_s006
                                 + 0.00692389457541361 w_k1_AXP_s007
                                 + 0.0355284769961272 w_k1_AXP_s010
                                 + 0.0132413323524811 w_k1_AXP_s011
                                 + 0.0107787562813059 w_k1_AXP_s012
                                 + 0.0219767455742348 w_k1_AXP_s013
                                 + 0.0350338478107216 w_k1_AXP_s014
                                 + 0.0263188391615638 w_k1_AXP_s015
                                 + 0.0206225609671983 w_k1_AXP_s016
                                 + 0.0276766727321343 w_k1_AXP_s017
                                 + 0.0392903496435494 w_k1_AXP_s100
                                 + 0.142357236584132 w_k1_AXP_s101
                                 + 0.050874045160338 w_k1_AXP_s102
                                 + 0.0341561532236124 w_k1_AXP_s103
                                 + 0.0529765331171593 w_k1_AXP_s104
                                 + 0.0359656211062926 w_k1_AXP_s105
                                 + 0.0622725348748164 w_k1_AXP_s106
                                 + 0.110323448985966 w_k1_AXP_s107
                                 + 0.030478002788152 w_k1_AXP_s110
                                 + 0.0290276604152887 w_k1_AXP_s111
                                 + 0.0298081125401375 w_k1_AXP_s112
                                 + 0.0249526649321766 w_k1_AXP_s113
                                 + 0.0502564191730749 w_k1_AXP_s114
                                 + 0.0135784991386825 w_k1_AXP_s115
                                 + 0.0217230347117103 w_k1_AXP_s116
                                 + 0.0237628293337641 w_k1_AXP_s117
                                 + 0.00817197692341675 w_k1_DIS_s000
                                 + 0.00778868523582362 w_k1_DIS_s001
                                 + 0.00609055153115394 w_k1_DIS_s002
                                 + 0.00999743027329908 w_k1_DIS_s003
                                 + 0.0212586346523082 w_k1_DIS_s004
                                 + 0.00612723937714021 w_k1_DIS_s005
                                 + 0.0129408366937599 w_k1_DIS_s006
                                 + 0.00620472880492794 w_k1_DIS_s007
                                 + 0.0362191515293885 w_k1_DIS_s010
                                 + 0.0123616731217965 w_k1_DIS_s011
                                 + 0.0110893151264227 w_k1_DIS_s012
                                 + 0.023894065019461 w_k1_DIS_s013
                                 + 0.0379924795978847 w_k1_DIS_s014
                                 + 0.025331376925403 w_k1_DIS_s015
                                 + 0.0223529879783476 w_k1_DIS_s016
                                 + 0.030528614776795 w_k1_DIS_s017
                                 + 0.0438407979722463 w_k1_DIS_s100
                                 + 0.142539926441908 w_k1_DIS_s101
                                 + 0.0484571248819414 w_k1_DIS_s102
                                 + 0.0335651080643265 w_k1_DIS_s103
                                 + 0.0514876674072296 w_k1_DIS_s104
                                 + 0.0364931327010843 w_k1_DIS_s105
                                 + 0.0614047276085875 w_k1_DIS_s106
                                 + 0.101666433942 w_k1_DIS_s107
                                 + 0.0300972412752073 w_k1_DIS_s110
                                 + 0.0322141396635812 w_k1_DIS_s111
                                 + 0.0320661188094895 w_k1_DIS_s112
                                 + 0.0259024203335903 w_k1_DIS_s113
                                 + 0.0535406537835334 w_k1_DIS_s114
                                 + 0.0133210104906373 w_k1_DIS_s115
                                 + 0.0233959860083192 w_k1_DIS_s116
                                 + 0.0271944056143547 w_k1_DIS_s117
                                 + 0.00749200092578801 w_k1_FDX_s000
                                 + 0.00733179312254683 w_k1_FDX_s001
                                 + 0.00617965377211443 w_k1_FDX_s002
                                 + 0.0107632486503896 w_k1_FDX_s003
                                 + 0.0199979531584085 w_k1_FDX_s004
                                 + 0.0058727207690358 w_k1_FDX_s005
                                 + 0.0127552883967245 w_k1_FDX_s006
                                 + 0.00625711510987862 w_k1_FDX_s007
                                 + 0.0346102988686928 w_k1_FDX_s010
                                 + 0.0126254230639859 w_k1_FDX_s011
                                 + 0.011404236094795 w_k1_FDX_s012
                                 + 0.0232189653758831 w_k1_FDX_s013
                                 + 0.0378579197839457 w_k1_FDX_s014
                                 + 0.0243557067255964 w_k1_FDX_s015
                                 + 0.021278783295161 w_k1_FDX_s016
                                 + 0.0249223968021811 w_k1_FDX_s017
                                 + 0.0411010876912468 w_k1_FDX_s100
                                 + 0.140814335862776 w_k1_FDX_s101
                                 + 0.0479925458093143 w_k1_FDX_s102
                                 + 0.0324581516311848 w_k1_FDX_s103
                                 + 0.0508173930234599 w_k1_FDX_s104
                                 + 0.0362690719493979 w_k1_FDX_s105
                                 + 0.0601756773168132 w_k1_FDX_s106
                                 + 0.102481806261017 w_k1_FDX_s107
                                 + 0.0288610170776569 w_k1_FDX_s110
                                 + 0.0329509649169629 w_k1_FDX_s111
                                 + 0.0305840527736155 w_k1_FDX_s112
                                 + 0.0254539635965916 w_k1_FDX_s113
                                 + 0.0558302430540798 w_k1_FDX_s114
                                 + 0.0134353369515607 w_k1_FDX_s115
                                 + 0.0219156655429423 w_k1_FDX_s116
                                 + 0.025330326072543 w_k1_FDX_s117
                                 + 0.00752912425444135 w_k1_IBM_s000
                                 + 0.00736400596080535 w_k1_IBM_s001
                                 + 0.00596627769968674 w_k1_IBM_s002
                                 + 0.0103714167856192 w_k1_IBM_s003
                                 + 0.0202015780000456 w_k1_IBM_s004
                                 + 0.00661003606706082 w_k1_IBM_s005
                                 + 0.0128334173806589 w_k1_IBM_s006
                                 + 0.00675364320518302 w_k1_IBM_s007
                                 + 0.0357600411923896 w_k1_IBM_s010
                                 + 0.0122469874720044 w_k1_IBM_s011
                                 + 0.0110317524445715 w_k1_IBM_s012
                                 + 0.0229289321079273 w_k1_IBM_s013
                                 + 0.0381341918814094 w_k1_IBM_s014
                                 + 0.0243457210572795 w_k1_IBM_s015
                                 + 0.0218631593935004 w_k1_IBM_s016
                                 + 0.0263716525412734 w_k1_IBM_s017
                                 + 0.0429004823153568 w_k1_IBM_s100
                                 + 0.144607587903833 w_k1_IBM_s101
                                 + 0.0489170559985414 w_k1_IBM_s102
                                 + 0.0329698856343229 w_k1_IBM_s103
                                 + 0.0509672013755981 w_k1_IBM_s104
                                 + 0.0366335110732217 w_k1_IBM_s105
                                 + 0.0598002301770424 w_k1_IBM_s106
                                 + 0.107855708063944 w_k1_IBM_s107
                                 + 0.0290619396963282 w_k1_IBM_s110
                                 + 0.0338559400913089 w_k1_IBM_s111
                                 + 0.0322491919012803 w_k1_IBM_s112
                                 + 0.0256593382544834 w_k1_IBM_s113
                                 + 0.0563837632445942 w_k1_IBM_s114
                                 + 0.0135193467782058 w_k1_IBM_s115
                                 + 0.0214199408187853 w_k1_IBM_s116
                                 + 0.0242566060399347 w_k1_IBM_s117
                                 + 0.00640275868591619 w_k1_INTC_s000
                                 + 0.00768416209530065 w_k1_INTC_s001
                                 + 0.00684889863226297 w_k1_INTC_s002
                                 + 0.0102748865319839 w_k1_INTC_s003
                                 + 0.0212346143658865 w_k1_INTC_s004
                                 + 0.00636673743961457 w_k1_INTC_s005
                                 + 0.0122163521375111 w_k1_INTC_s006
                                 + 0.00738752911556837 w_k1_INTC_s007
                                 + 0.032785966006531 w_k1_INTC_s010
                                 + 0.0115320994335812 w_k1_INTC_s011
                                 + 0.0129502573188368 w_k1_INTC_s012
                                 + 0.021151369557953 w_k1_INTC_s013
                                 + 0.0368929677647355 w_k1_INTC_s014
                                 + 0.0271716207493108 w_k1_INTC_s015
                                 + 0.0246161921054373 w_k1_INTC_s016
                                 + 0.0278564118626279 w_k1_INTC_s017
                                 + 0.042894675206516 w_k1_INTC_s100
                                 + 0.135551956902752 w_k1_INTC_s101
                                 + 0.0451325026639435 w_k1_INTC_s102
                                 + 0.0330059103681128 w_k1_INTC_s103
                                 + 0.0468644862902156 w_k1_INTC_s104
                                 + 0.0362755992931469 w_k1_INTC_s105
                                 + 0.0581359785965885 w_k1_INTC_s106
                                 + 0.101389329627755 w_k1_INTC_s107
                                 + 0.0264822570978034 w_k1_INTC_s110
                                 + 0.0346833228402293 w_k1_INTC_s111
                                 + 0.0306643746732106 w_k1_INTC_s112
                                 + 0.0295640081457296 w_k1_INTC_s113
                                 + 0.0508248212875389 w_k1_INTC_s114
                                 + 0.0133277650842083 w_k1_INTC_s115
                                 + 0.021571996302078 w_k1_INTC_s116
                                 + 0.0271551376052759 w_k1_INTC_s117
                                 + 0.00802164314522552 w_k1_KO_s000
                                 + 0.00828754907070917 w_k1_KO_s001
                                 + 0.00629359727070601 w_k1_KO_s002
                                 + 0.0100485858665358 w_k1_KO_s003
                                 + 0.0226812102413582 w_k1_KO_s004
                                 + 0.0068724762568696 w_k1_KO_s005
                                 + 0.0121434347234349 w_k1_KO_s006
                                 + 0.00726287293312396 w_k1_KO_s007
                                 + 0.0306673842546883 w_k1_KO_s010
                                 + 0.011567063113608 w_k1_KO_s011
                                 + 0.0116368584810602 w_k1_KO_s012
                                 + 0.0217171440794939 w_k1_KO_s013
                                 + 0.0390686093870872 w_k1_KO_s014
                                 + 0.0250997972971812 w_k1_KO_s015
                                 + 0.0206335822404119 w_k1_KO_s016
                                 + 0.028655639503047 w_k1_KO_s017
                                 + 0.0438789682685935 w_k1_KO_s100
                                 + 0.140573081555477 w_k1_KO_s101
                                 + 0.0516555943344467 w_k1_KO_s102
                                 + 0.0352404531097938 w_k1_KO_s103
                                 + 0.0531186997261113 w_k1_KO_s104
                                 + 0.0346754278911151 w_k1_KO_s105
                                 + 0.0558190984917037 w_k1_KO_s106
                                 + 0.110897396773423 w_k1_KO_s107
                                 + 0.0307825956244846 w_k1_KO_s110
                                 + 0.0309843087699664 w_k1_KO_s111
                                 + 0.0297914301419455 w_k1_KO_s112
                                 + 0.0271590598943195 w_k1_KO_s113
                                 + 0.0558098807705064 w_k1_KO_s114
                                 + 0.0143296697001322 w_k1_KO_s115
                                 + 0.0216861324109889 w_k1_KO_s116
                                 + 0.0274807218013352 w_k1_KO_s117
                                 + 0.00727682999305845 w_k1_MAT_s000
                                 + 0.00788357074047216 w_k1_MAT_s001
                                 + 0.00645440526410992 w_k1_MAT_s002
                                 + 0.0101100916672402 w_k1_MAT_s003
                                 + 0.0272506906280295 w_k1_MAT_s004
                                 + 0.0079613640835504 w_k1_MAT_s005
                                 + 0.0148515053100943 w_k1_MAT_s006
                                 + 0.00804092364661445 w_k1_MAT_s007
                                 + 0.0322994639646347 w_k1_MAT_s010
                                 + 0.0127862866663207 w_k1_MAT_s011
                                 + 0.0103027111871214 w_k1_MAT_s012
                                 + 0.0202822410893546 w_k1_MAT_s013
                                 + 0.0306278926306091 w_k1_MAT_s014
                                 + 0.022620142381394 w_k1_MAT_s015
                                 + 0.0228189211942452 w_k1_MAT_s016
                                 + 0.0256578848087789 w_k1_MAT_s017
                                 + 0.0368049832000129 w_k1_MAT_s100
                                 + 0.159380890732961 w_k1_MAT_s101
                                 + 0.0550212919468753 w_k1_MAT_s102
                                 + 0.0359322425075002 w_k1_MAT_s103
                                 + 0.0591407426908955 w_k1_MAT_s104
                                 + 0.0371436379708612 w_k1_MAT_s105
                                 + 0.0571881373870371 w_k1_MAT_s106
                                 + 0.115745665640033 w_k1_MAT_s107
                                 + 0.0350014221773054 w_k1_MAT_s110
                                 + 0.0283539164690997 w_k1_MAT_s111
                                 + 0.0330622800370989 w_k1_MAT_s112
                                 + 0.0243044651031007 w_k1_MAT_s113
                                 + 0.0648757678671871 w_k1_MAT_s114
                                 + 0.013457037697063 w_k1_MAT_s115
                                 + 0.0268218922658014 w_k1_MAT_s116
                                 + 0.0294293756059963 w_k1_MAT_s117
                                 + 0.0075257239632293 w_k1_MSFT_s000
                                 + 0.00797856007343565 w_k1_MSFT_s001
                                 + 0.00652391509207762 w_k1_MSFT_s002
                                 + 0.00976411877035569 w_k1_MSFT_s003
                                 + 0.0214605886656029 w_k1_MSFT_s004
                                 + 0.00641459492661862 w_k1_MSFT_s005
                                 + 0.0138957330802175 w_k1_MSFT_s006
                                 + 0.00693105748113325 w_k1_MSFT_s007
                                 + 0.032196772126185 w_k1_MSFT_s010
                                 + 0.0126611116580236 w_k1_MSFT_s011
                                 + 0.0106711779123898 w_k1_MSFT_s012
                                 + 0.0207098599653666 w_k1_MSFT_s013
                                 + 0.0400868971283566 w_k1_MSFT_s014
                                 + 0.0259744820865423 w_k1_MSFT_s015
                                 + 0.0217554637302414 w_k1_MSFT_s016
                                 + 0.0244061262382969 w_k1_MSFT_s017
                                 + 0.039332941463818 w_k1_MSFT_s100
                                 + 0.15537316379878 w_k1_MSFT_s101
                                 + 0.0482034404544143 w_k1_MSFT_s102
                                 + 0.035975426182387 w_k1_MSFT_s103
                                 + 0.0484865977409537 w_k1_MSFT_s104
                                 + 0.0377762212821729 w_k1_MSFT_s105
                                 + 0.0592291087690748 w_k1_MSFT_s106
                                 + 0.102774054384482 w_k1_MSFT_s107
                                 + 0.030400766526323 w_k1_MSFT_s110
                                 + 0.0296168720244127 w_k1_MSFT_s111
                                 + 0.0326149626870205 w_k1_MSFT_s112
                                 + 0.0252848668580859 w_k1_MSFT_s113
                                 + 0.0538814779938209 w_k1_MSFT_s114
                                 + 0.0132093136810434 w_k1_MSFT_s115
                                 + 0.0227046939511934 w_k1_MSFT_s116
                                 + 0.0268808095957523 w_k1_MSFT_s117
                                 + 0.00721929897316295 w_k1_PG_s000
                                 + 0.00732195892754969 w_k1_PG_s001
                                 + 0.00612443827721038 w_k1_PG_s002
                                 + 0.0104183694872849 w_k1_PG_s003
                                 + 0.0191021399807872 w_k1_PG_s004
                                 + 0.00670252790481331 w_k1_PG_s005
                                 + 0.0124868968795035 w_k1_PG_s006
                                 + 0.00682886184504791 w_k1_PG_s007
                                 + 0.0340676273774418 w_k1_PG_s010
                                 + 0.011934490269164 w_k1_PG_s011
                                 + 0.0108890816081544 w_k1_PG_s012
                                 + 0.0215489446095679 w_k1_PG_s013
                                 + 0.035933063833717 w_k1_PG_s014
                                 + 0.0254572199526157 w_k1_PG_s015
                                 + 0.0207519241064739 w_k1_PG_s016
                                 + 0.0254873140676596 w_k1_PG_s017
                                 + 0.0426293284131665 w_k1_PG_s100
                                 + 0.144393705540486 w_k1_PG_s101
                                 + 0.0511771690473566 w_k1_PG_s102
                                 + 0.036837892653754 w_k1_PG_s103
                                 + 0.0483608104701115 w_k1_PG_s104
                                 + 0.0350414631032955 w_k1_PG_s105
                                 + 0.0545921342060562 w_k1_PG_s106
                                 + 0.106653633462663 w_k1_PG_s107
                                 + 0.0282508171548072 w_k1_PG_s110
                                 + 0.0321046939060528 w_k1_PG_s111
                                 + 0.0303748074624368 w_k1_PG_s112
                                 + 0.0253535458053129 w_k1_PG_s113
                                 + 0.0598897184739036 w_k1_PG_s114
                                 + 0.0135803042448348 w_k1_PG_s115
                                 + 0.0209699155210874 w_k1_PG_s116
                                 + 0.0234974137702175 w_k1_PG_s117 >= 1.06
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
