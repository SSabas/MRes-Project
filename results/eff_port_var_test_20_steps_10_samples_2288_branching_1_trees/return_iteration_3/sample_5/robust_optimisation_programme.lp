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
 w_k1_ADM_s0_return_constraint:  1.05358 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.856578 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.0278 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.15364 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.24228 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.03765 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.00466 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.880967 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.782449 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.27124 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.873736 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.20962 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.726502 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.25183 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.79207 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.776499 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.03574 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.17706 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.21022 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.65523 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.46679 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.970614 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.849894 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.11819 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.943466 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.20664 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.36778 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.11323 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.7307 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.767999 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.14374 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.40281 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.88041 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.844229 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.77187 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.2396 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.921018 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.01144 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.22547 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.93367 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.900245 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.2856 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.26707 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.919382 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.913256 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.812109 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.787004 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.58202 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.725098 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.41104 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.734719 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.13367 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.742028 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.729216 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.907119 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.957337 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.22121 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.668098 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.29356 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.01347 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.678248 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.19218 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.10098 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.14431 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.61157 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.3779 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.80316 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.795774 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.07733 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.3369 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.696768 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.577971 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.650675 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.31931 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.94287 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.768584 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.26754 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.918392 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.896775 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.24045 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.3086 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.886531 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.04078 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.869103 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.692457 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.92315 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.680831 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.51716 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.623461 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.17494 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.627557 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.73722 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.25842 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.00003 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.48029 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.526333 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.45265 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.924414 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.570185 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.094 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.907351 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.53118 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.75796 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.3643 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.10224 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.815901 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.11709 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.68019 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.718242 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.580259 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.545176 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.21531 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.842374 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.818405 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.22035 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.914598 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.928909 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.07571 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.15875 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.962862 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.01307 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.853312 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.767554 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.56542 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.873461 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.22205 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.763784 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.20446 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.707633 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.822618 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.09099 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.923205 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.17486 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.661882 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.37353 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.965755 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.712934 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.11628 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.875847 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.29339 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.39628 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.07846 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.832331 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.837046 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.20249 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.26489 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.780993 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.690215 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.721587 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.19247 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.80908 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.938732 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.00676 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.03771 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.819081 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.44019 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.48517 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.844174 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.957137 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.21647 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.30776 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.25219 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.649462 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.657 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.820264 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.969723 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.11718 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.749226 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.868822 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.27785 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.53299 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.681812 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.869773 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.971259 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.680125 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.794593 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.18703 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.883469 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.30026 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.82307 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.18388 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.866179 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.728764 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.45275 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.970798 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.780496 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.544681 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.08245 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.4324 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.70959 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09738 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.946988 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.996457 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.15346 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.11449 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.859059 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.851242 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.909188 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.861895 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.44972 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.960539 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.20072 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.765654 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.08752 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.836889 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.771115 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.946163 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.867217 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.938262 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.712198 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.31638 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.08958 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.745004 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.1606 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.872092 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.15819 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.23156 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.04648 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.14446 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.834636 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.23236 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.32385 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.95582 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.81609 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.803475 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.25776 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.915525 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.11097 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.10566 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.970276 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.906914 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.31986 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.34665 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.835834 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.980352 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.07493 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.834707 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.32704 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.699122 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.33617 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.742692 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.13375 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.84831 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.77497 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.930345 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.03945 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.36609 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.578361 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.07825 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.965182 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.66735 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.98247 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.04625 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.09988 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.37603 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.24691 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.13408 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.861166 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.02596 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.50608 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.940791 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.816897 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.691103 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.21584 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.14016 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.802863 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.03679 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.04426 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.940097 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.75189 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.53855 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.779059 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.918236 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.02456 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.15921 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.22516 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.401196 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.70113 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.785199 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.922764 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.82405 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.696039 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.802401 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.01614 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.43795 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.56642 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.753029 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.964501 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.620894 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.926635 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.12105 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.9659 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.30638 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.78737 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.914906 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.811036 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.76164 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.61931 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.999526 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.62377 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.590291 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.22484 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.46788 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.599262 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1765 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.909172 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.965497 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.09819 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.12046 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.979837 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.983011 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.895026 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.811311 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.45969 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.84707 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.2745 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.718188 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.11157 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.767239 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.822617 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.10232 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.978571 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.11915 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.620071 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.27576 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.963551 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.816246 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.19528 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.94005 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.36162 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.30899 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.12121 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.08626 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.965898 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.20584 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.23349 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.837081 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.68364 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.743129 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.19124 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.805258 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.942712 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08506 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.949789 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.916009 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.15918 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.1698 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.932275 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.02423 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.08145 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.885054 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.25956 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.850825 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.15709 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.885288 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.17124 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.813893 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.841323 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.901522 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.00265 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.21497 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.757557 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.14599 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.940443 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.82372 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.991406 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.01682 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.15477 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.24784 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.19552 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.966195 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.932492 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.03172 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.26446 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.915262 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.798273 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.786012 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.14553 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.04324 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.922955 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.678000279583603 z_k1_s0000
                                 - 0.551999646078284 z_k1_s0001
                                 - 0.379999126316244 z_k1_s0002
                                 - 0.327999474677693 z_k1_s0003
                                 - 0.779999829566365 z_k1_s0004
                                 - 0.570996291376888 z_k1_s0005
                                 - 0.872996445734116 z_k1_s0006
                                 - 0.892999033665615 z_k1_s0007
                                 - 0.100999962363028 z_k1_s0010
                                 - 0.200999280509234 z_k1_s0011
                                 - 0.237999312155051 z_k1_s0012
                                 - 0.162999751845467 z_k1_s0013
                                 - 0.310999878428735 z_k1_s0014
                                 - 0.0719999220568622 z_k1_s0015
                                 - 0.197999355456794 z_k1_s0016
                                 - 0.150000554617758 z_k1_s0017
                                 - 0.396999084083509 z_k1_s0020
                                 - 0.388999282993689 z_k1_s0021
                                 - 0.214999629289575 z_k1_s0022
                                 - 0.216000599425938 z_k1_s0023
                                 - 0.183000026930243 z_k1_s0024
                                 - 0.243999903240306 z_k1_s0025
                                 - 0.208000798336118 z_k1_s0026
                                 - 0.137999951799846 z_k1_s0027
                                 - 0.158999866179169 z_k1_s0030
                                 - 0.314000008831513 z_k1_s0031
                                 - 0.341999017518235 z_k1_s0032
                                 - 0.226999650130187 z_k1_s0033
                                 - 0.396998756008126 z_k1_s0034
                                 - 0.144999889836336 z_k1_s0035
                                 - 0.24699968376273 z_k1_s0036
                                 - 0.529000488381223 z_k1_s0037
                                 - 0.274000586075059 z_k1_s0040
                                 - 0.0969999229215413 z_k1_s0041
                                 - 0.295000442256046 z_k1_s0042
                                 - 0.442999902245732 z_k1_s0043
                                 - 0.191000036874323 z_k1_s0044
                                 - 0.284999753027215 z_k1_s0045
                                 - 0.122000054794061 z_k1_s0046
                                 - 0.282000341858345 z_k1_s0047
                                 - 0.0719999944654115 z_k1_s0050
                                 - 0.0759998936691358 z_k1_s0051
                                 - 0.161999858347292 z_k1_s0052
                                 - 0.211000206291492 z_k1_s0053
                                 - 0.148999766035566 z_k1_s0054
                                 - 0.298999539172033 z_k1_s0055
                                 - 0.201000135282472 z_k1_s0056
                                 - 0.121999832710978 z_k1_s0057
                                 - 0.244000860187257 z_k1_s0060
                                 - 0.247000906002795 z_k1_s0061
                                 - 0.500001366234819 z_k1_s0062
                                 - 0.367002469923402 z_k1_s0063
                                 - 0.29000084615638 z_k1_s0064
                                 - 0.411002246214883 z_k1_s0065
                                 - 0.445000974118158 z_k1_s0066
                                 - 0.183000645140443 z_k1_s0067
                                 - 0.349999224900099 z_k1_s0070
                                 - 0.221999592440435 z_k1_s0071
                                 - 0.851998524219839 z_k1_s0072
                                 - 0.523997116022516 z_k1_s0073
                                 - 0.710997358151586 z_k1_s0074
                                 - 0.251999327492342 z_k1_s0075
                                 - 0.583998220922458 z_k1_s0076
                                 - 0.592000547970622 z_k1_s0077
                                 - 0.0559999965100898 z_k1_s0100
                                 - 0.048999978796333 z_k1_s0101
                                 - 0.107000032224627 z_k1_s0102
                                 - 0.0669999828602891 z_k1_s0103
                                 - 0.102999773188254 z_k1_s0104
                                 - 0.048999978796333 z_k1_s0105
                                 - 0.170999820807636 z_k1_s0106
                                 - 0.124000044129472 z_k1_s0107
                                 - 0.331999296008718 z_k1_s0110
                                 - 0.217999863771836 z_k1_s0111
                                 - 0.299999583043919 z_k1_s0112
                                 - 0.335000974598367 z_k1_s0113
                                 - 0.315999439526319 z_k1_s0114
                                 - 0.20499986667994 z_k1_s0115
                                 - 0.3649995685034 z_k1_s0116
                                 - 0.20299988461964 z_k1_s0117
                                 - 0.270999402673313 z_k1_s0120
                                 - 0.309998956534705 z_k1_s0121
                                 - 0.179999281665518 z_k1_s0122
                                 - 0.155999784142707 z_k1_s0123
                                 - 0.143999773931778 z_k1_s0124
                                 - 0.16399961665031 z_k1_s0125
                                 - 0.354999648449494 z_k1_s0126
                                 - 0.16299965937432 z_k1_s0127
                                 - 0.0650001064014071 z_k1_s0130
                                 - 0.185000565742825 z_k1_s0131
                                 - 0.0460000844440852 z_k1_s0132
                                 - 0.0670001165337674 z_k1_s0133
                                 - 0.0490000996426257 z_k1_s0134
                                 - 0.123000103039451 z_k1_s0135
                                 - 0.0660001114675872 z_k1_s0136
                                 - 0.142000124996773 z_k1_s0137
                                 - 0.357999093637435 z_k1_s0140
                                 - 0.264998718850181 z_k1_s0141
                                 - 0.188999704573022 z_k1_s0142
                                 - 0.28999859798699 z_k1_s0143
                                 - 0.336000024395365 z_k1_s0144
                                 - 0.231000532020765 z_k1_s0145
                                 - 0.227000139159715 z_k1_s0146
                                 - 0.164999614502104 z_k1_s0147
                                 - 0.105999549038188 z_k1_s0150
                                 - 0.0840000264528933 z_k1_s0151
                                 - 0.179000400998418 z_k1_s0152
                                 - 0.228999671601351 z_k1_s0153
                                 - 0.125000132507279 z_k1_s0154
                                 - 0.0999995192058397 z_k1_s0155
                                 - 0.0779999966205445 z_k1_s0156
                                 - 0.0769999916484864 z_k1_s0157
                                 - 0.229998756766571 z_k1_s0160
                                 - 0.271999877442946 z_k1_s0161
                                 - 0.199999091709711 z_k1_s0162
                                 - 0.231000202732173 z_k1_s0163
                                 - 0.216999829173381 z_k1_s0164
                                 - 0.286999709971376 z_k1_s0165
                                 - 0.16199958220422 z_k1_s0166
                                 - 0.38799871479587 z_k1_s0167
                                 - 0.137999938875383 z_k1_s0170
                                 - 0.240998773602488 z_k1_s0171
                                 - 0.179999802254982 z_k1_s0172
                                 - 0.132999810968452 z_k1_s0173
                                 - 0.216999328927888 z_k1_s0174
                                 - 0.443000015606968 z_k1_s0175
                                 - 0.50000026270736 z_k1_s0176
                                 - 0.236000107293893 z_k1_s0177
                                 - 0.363000792496919 z_k1_s1000
                                 - 0.380999387631956 z_k1_s1001
                                 - 0.209999466276283 z_k1_s1002
                                 - 0.21299955573146 z_k1_s1003
                                 - 0.986997394418729 z_k1_s1004
                                 - 0.225999511904776 z_k1_s1005
                                 - 0.420999501703187 z_k1_s1006
                                 - 0.434998409030438 z_k1_s1007
                                 - 0.183999533738329 z_k1_s1010
                                 - 0.222999679875344 z_k1_s1011
                                 - 0.507998290919654 z_k1_s1012
                                 - 0.251999497695718 z_k1_s1013
                                 - 0.12599989809758 z_k1_s1014
                                 - 0.5520000937122 z_k1_s1015
                                 - 0.579999341401914 z_k1_s1016
                                 - 0.559999878766404 z_k1_s1017
                                 - 0.567000347964321 z_k1_s1020
                                 - 0.211999708737859 z_k1_s1021
                                 - 0.704997904462835 z_k1_s1022
                                 - 0.351999404630987 z_k1_s1023
                                 - 0.810996771583543 z_k1_s1024
                                 - 0.223999513429004 z_k1_s1025
                                 - 0.576999197958719 z_k1_s1026
                                 - 0.501000829814094 z_k1_s1027
                                 - 0.191000907820334 z_k1_s1030
                                 - 0.400001039671937 z_k1_s1031
                                 - 0.423003046637466 z_k1_s1032
                                 - 0.171000746136208 z_k1_s1033
                                 - 0.235000473681601 z_k1_s1034
                                 - 0.14200075303539 z_k1_s1035
                                 - 0.386001058133683 z_k1_s1036
                                 - 0.246001242751152 z_k1_s1037
                                 - 0.242000413448897 z_k1_s1040
                                 - 0.285999424532545 z_k1_s1041
                                 - 0.127999783374627 z_k1_s1042
                                 - 0.285999424532545 z_k1_s1043
                                 - 0.389998151182772 z_k1_s1044
                                 - 0.159999787743172 z_k1_s1045
                                 - 0.427999531705296 z_k1_s1046
                                 - 0.420999955055118 z_k1_s1047
                                 - 0.0770000947015312 z_k1_s1050
                                 - 0.148000763751737 z_k1_s1051
                                 - 0.065000058397369 z_k1_s1052
                                 - 0.16900017553319 z_k1_s1053
                                 - 0.171000458084236 z_k1_s1054
                                 - 0.128000308244296 z_k1_s1055
                                 - 0.148000763751737 z_k1_s1056
                                 - 0.279000310821091 z_k1_s1057
                                 - 0.175999671983487 z_k1_s1060
                                 - 0.127999983260758 z_k1_s1061
                                 - 0.133999944351099 z_k1_s1062
                                 - 0.0739999674476217 z_k1_s1063
                                 - 0.313999997061554 z_k1_s1064
                                 - 0.13699992489627 z_k1_s1065
                                 - 0.119000041625246 z_k1_s1066
                                 - 0.138000325078067 z_k1_s1067
                                 - 0.426000787539497 z_k1_s1070
                                 - 0.542999870719557 z_k1_s1071
                                 - 0.631002029846248 z_k1_s1072
                                 - 0.456001436536978 z_k1_s1073
                                 - 0.438999408666869 z_k1_s1074
                                 - 0.485002191601583 z_k1_s1075
                                 - 0.599001592983018 z_k1_s1076
                                 - 0.252000854365349 z_k1_s1077
                                 - 1.12499569060681 z_k1_s1100
                                 - 0.663999211292212 z_k1_s1101
                                 - 1.01700136868407 z_k1_s1102
                                 - 0.457999312945762 z_k1_s1103
                                 - 1.46999545449781 z_k1_s1104
                                 - 0.632999094861647 z_k1_s1105
                                 - 0.865996556090422 z_k1_s1106
                                 - 0.550998983838548 z_k1_s1107
                                 - 1.71999645634689 z_k1_s1110
                                 - 1.04599970267848 z_k1_s1111
                                 - 1.32599531510926 z_k1_s1112
                                 - 1.26100028384625 z_k1_s1113
                                 - 0.802999560267638 z_k1_s1114
                                 - 0.409999603048708 z_k1_s1115
                                 - 0.815999385819765 z_k1_s1116
                                 - 0.810999200822939 z_k1_s1117
                                 - 1.31600056498298 z_k1_s1120
                                 - 0.970003273107807 z_k1_s1121
                                 - 0.591002023156455 z_k1_s1122
                                 - 0.569001254775511 z_k1_s1123
                                 - 0.817002265147698 z_k1_s1124
                                 - 0.380000570749272 z_k1_s1125
                                 - 0.497000780441342 z_k1_s1126
                                 - 0.471001045565856 z_k1_s1127
                                 - 0.388999621441769 z_k1_s1130
                                 - 0.402998616629962 z_k1_s1131
                                 - 0.195999898233894 z_k1_s1132
                                 - 0.398001360919755 z_k1_s1133
                                 - 0.217999967358102 z_k1_s1134
                                 - 0.354000615471374 z_k1_s1135
                                 - 0.858000874244214 z_k1_s1136
                                 - 0.220000084042114 z_k1_s1137
                                 - 0.187000073208138 z_k1_s1140
                                 - 0.80200110049766 z_k1_s1141
                                 - 0.550001313258748 z_k1_s1142
                                 - 0.668998101956268 z_k1_s1143
                                 - 0.23699985323181 z_k1_s1144
                                 - 0.182999792166376 z_k1_s1145
                                 - 0.234999712710929 z_k1_s1146
                                 - 0.870001398609591 z_k1_s1147
                                 - 2.09600053604727 z_k1_s1150
                                 - 2.33300489962892 z_k1_s1151
                                 - 0.856999802547771 z_k1_s1152
                                 - 1.91699525781385 z_k1_s1153
                                 - 0.818000259503202 z_k1_s1154
                                 - 1.56400418354074 z_k1_s1155
                                 - 1.05700095595734 z_k1_s1156
                                 - 0.818000259503202 z_k1_s1157
                                 - 0.510999578342243 z_k1_s1160
                                 - 0.741001235217188 z_k1_s1161
                                 - 0.407998820750413 z_k1_s1162
                                 - 0.561001966533267 z_k1_s1163
                                 - 0.359998537367841 z_k1_s1164
                                 - 0.58899914183856 z_k1_s1165
                                 - 0.207000055986871 z_k1_s1166
                                 - 0.210999960002037 z_k1_s1167
                                 - 0.66599852157706 z_k1_s1170
                                 - 0.240999957519687 z_k1_s1171
                                 - 0.555000347981076 z_k1_s1172
                                 - 0.733998569906273 z_k1_s1173
                                 - 0.256999987479504 z_k1_s1174
                                 - 0.305000077358956 z_k1_s1175
                                 - 0.238999993274715 z_k1_s1176
                                 - 0.163000087965612 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.901702 w_k1_ADM_s000
                                 + 0.754064 w_k1_AXP_s000
                                 + 0.642033 w_k1_DIS_s000
                                 + 0.78246 w_k1_FDX_s000
                                 + 0.733973 w_k1_IBM_s000
                                 + 0.871015 w_k1_INTC_s000
                                 + 0.70348 w_k1_KO_s000
                                 + 0.716888 w_k1_MAT_s000
                                 + 0.786561 w_k1_MSFT_s000
                                 + 0.811636 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.35576 w_k1_ADM_s000
                                 + 1.4932 w_k1_AXP_s000 + 1.17575 w_k1_DIS_s000
                                 + 1.28993 w_k1_FDX_s000 + 1.0419 w_k1_IBM_s000
                                 + 1.22495 w_k1_INTC_s000
                                 + 0.997074 w_k1_KO_s000
                                 + 0.909253 w_k1_MAT_s000
                                 + 1.06951 w_k1_MSFT_s000 + 1.2136 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.00155 w_k1_ADM_s000
                                 + 0.897483 w_k1_AXP_s000
                                 + 0.758375 w_k1_DIS_s000
                                 + 1.00948 w_k1_FDX_s000
                                 + 0.666615 w_k1_IBM_s000
                                 + 1.19111 w_k1_INTC_s000
                                 + 0.706182 w_k1_KO_s000
                                 + 0.451375 w_k1_MAT_s000
                                 + 1.00019 w_k1_MSFT_s000
                                 + 0.915118 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.895418 w_k1_ADM_s000
                                 + 0.689612 w_k1_AXP_s000
                                 + 0.806056 w_k1_DIS_s000
                                 + 0.790873 w_k1_FDX_s000
                                 + 1.22793 w_k1_IBM_s000
                                 + 0.915964 w_k1_INTC_s000
                                 + 0.884084 w_k1_KO_s000
                                 + 0.999419 w_k1_MAT_s000
                                 + 0.766247 w_k1_MSFT_s000
                                 + 0.930344 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.07009 w_k1_ADM_s000
                                 + 0.888651 w_k1_AXP_s000
                                 + 1.13289 w_k1_DIS_s000
                                 + 1.03352 w_k1_FDX_s000
                                 + 0.905018 w_k1_IBM_s000
                                 + 1.23341 w_k1_INTC_s000
                                 + 1.01224 w_k1_KO_s000
                                 + 0.651393 w_k1_MAT_s000
                                 + 1.17516 w_k1_MSFT_s000 + 1.0966 w_k1_PG_s000
                                 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.23386 w_k1_ADM_s000
                                 + 1.53158 w_k1_AXP_s000
                                 + 1.74677 w_k1_DIS_s000
                                 + 1.31933 w_k1_FDX_s000 + 1.7419 w_k1_IBM_s000
                                 + 1.42582 w_k1_INTC_s000
                                 + 1.71113 w_k1_KO_s000 + 1.6708 w_k1_MAT_s000
                                 + 1.30176 w_k1_MSFT_s000
                                 + 1.28853 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.13452 w_k1_ADM_s000
                                 + 1.01339 w_k1_AXP_s000 + 1.1588 w_k1_DIS_s000
                                 + 1.06744 w_k1_FDX_s000
                                 + 1.29218 w_k1_IBM_s000
                                 + 1.17136 w_k1_INTC_s000
                                 + 1.14642 w_k1_KO_s000 + 1.28168 w_k1_MAT_s000
                                 + 1.1177 w_k1_MSFT_s000 + 1.08692 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.968664 w_k1_ADM_s000
                                 + 0.970837 w_k1_AXP_s000
                                 + 0.810379 w_k1_DIS_s000
                                 + 0.907166 w_k1_FDX_s000
                                 + 1.20588 w_k1_IBM_s000
                                 + 1.21788 w_k1_INTC_s000
                                 + 0.993674 w_k1_KO_s000
                                 + 1.02196 w_k1_MAT_s000
                                 + 0.875333 w_k1_MSFT_s000
                                 + 0.928253 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.1092 w_k1_ADM_s001
                                 + 1.42438 w_k1_AXP_s001
                                 + 1.44203 w_k1_DIS_s001
                                 + 1.19739 w_k1_FDX_s001
                                 + 1.67441 w_k1_IBM_s001
                                 + 1.16937 w_k1_INTC_s001 + 1.198 w_k1_KO_s001
                                 + 1.95662 w_k1_MAT_s001
                                 + 1.11091 w_k1_MSFT_s001
                                 + 1.05559 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.977684 w_k1_ADM_s001
                                 + 1.08895 w_k1_AXP_s001
                                 + 1.02871 w_k1_DIS_s001
                                 + 0.907774 w_k1_FDX_s001
                                 + 1.28832 w_k1_IBM_s001
                                 + 0.851448 w_k1_INTC_s001
                                 + 1.0585 w_k1_KO_s001 + 1.3288 w_k1_MAT_s001
                                 + 0.950225 w_k1_MSFT_s001
                                 + 1.03498 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.10227 w_k1_ADM_s001
                                 + 1.00162 w_k1_AXP_s001
                                 + 0.97383 w_k1_DIS_s001
                                 + 0.970147 w_k1_FDX_s001
                                 + 0.943253 w_k1_IBM_s001
                                 + 0.787426 w_k1_INTC_s001
                                 + 0.876826 w_k1_KO_s001
                                 + 0.703677 w_k1_MAT_s001
                                 + 0.8506 w_k1_MSFT_s001 + 0.90612 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.39785 w_k1_ADM_s001
                                 + 1.29111 w_k1_AXP_s001
                                 + 1.28021 w_k1_DIS_s001
                                 + 1.25179 w_k1_FDX_s001
                                 + 1.10019 w_k1_IBM_s001
                                 + 0.90128 w_k1_INTC_s001
                                 + 0.967416 w_k1_KO_s001
                                 + 1.24808 w_k1_MAT_s001
                                 + 1.14613 w_k1_MSFT_s001
                                 + 1.03991 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.963439 w_k1_ADM_s001
                                 + 0.690733 w_k1_AXP_s001
                                 + 0.897336 w_k1_DIS_s001
                                 + 0.868227 w_k1_FDX_s001
                                 + 0.871407 w_k1_IBM_s001
                                 + 0.939384 w_k1_INTC_s001
                                 + 0.909078 w_k1_KO_s001
                                 + 0.617566 w_k1_MAT_s001
                                 + 0.979265 w_k1_MSFT_s001
                                 + 0.866848 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.19778 w_k1_ADM_s001
                                 + 1.17712 w_k1_AXP_s001
                                 + 1.02111 w_k1_DIS_s001
                                 + 1.05142 w_k1_FDX_s001
                                 + 1.27619 w_k1_IBM_s001
                                 + 1.13611 w_k1_INTC_s001
                                 + 1.09115 w_k1_KO_s001 + 1.44988 w_k1_MAT_s001
                                 + 1.01876 w_k1_MSFT_s001
                                 + 0.927399 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.15948 w_k1_ADM_s001
                                 + 1.28129 w_k1_AXP_s001
                                 + 1.41004 w_k1_DIS_s001
                                 + 1.12693 w_k1_FDX_s001
                                 + 0.977384 w_k1_IBM_s001
                                 + 1.17949 w_k1_INTC_s001
                                 + 1.22409 w_k1_KO_s001 + 1.19089 w_k1_MAT_s001
                                 + 1.29229 w_k1_MSFT_s001
                                 + 1.07389 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.06493 w_k1_ADM_s001
                                 + 0.909483 w_k1_AXP_s001
                                 + 0.74638 w_k1_DIS_s001
                                 + 0.849114 w_k1_FDX_s001
                                 + 0.621315 w_k1_IBM_s001
                                 + 0.938451 w_k1_INTC_s001
                                 + 0.74508 w_k1_KO_s001
                                 + 0.677563 w_k1_MAT_s001
                                 + 0.863899 w_k1_MSFT_s001
                                 + 0.847421 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.808479 w_k1_ADM_s002
                                 + 0.952568 w_k1_AXP_s002
                                 + 0.65035 w_k1_DIS_s002
                                 + 0.771411 w_k1_FDX_s002
                                 + 0.838668 w_k1_IBM_s002
                                 + 0.830708 w_k1_INTC_s002
                                 + 0.841184 w_k1_KO_s002
                                 + 0.959981 w_k1_MAT_s002
                                 + 0.761309 w_k1_MSFT_s002
                                 + 0.927562 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.13082 w_k1_ADM_s002
                                 + 0.941866 w_k1_AXP_s002
                                 + 0.988218 w_k1_DIS_s002
                                 + 1.00359 w_k1_FDX_s002
                                 + 0.82768 w_k1_IBM_s002
                                 + 0.961191 w_k1_INTC_s002
                                 + 1.05446 w_k1_KO_s002
                                 + 0.786958 w_k1_MAT_s002
                                 + 0.906805 w_k1_MSFT_s002
                                 + 1.04192 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.14087 w_k1_ADM_s002
                                 + 1.43655 w_k1_AXP_s002
                                 + 1.29882 w_k1_DIS_s002
                                 + 1.20826 w_k1_FDX_s002
                                 + 1.20882 w_k1_IBM_s002
                                 + 1.07474 w_k1_INTC_s002
                                 + 1.25289 w_k1_KO_s002 + 1.1121 w_k1_MAT_s002
                                 + 1.16343 w_k1_MSFT_s002
                                 + 1.16069 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.41567 w_k1_ADM_s002
                                 + 1.95918 w_k1_AXP_s002
                                 + 1.70418 w_k1_DIS_s002
                                 + 1.35354 w_k1_FDX_s002
                                 + 1.47656 w_k1_IBM_s002
                                 + 1.51031 w_k1_INTC_s002
                                 + 1.53303 w_k1_KO_s002 + 1.66886 w_k1_MAT_s002
                                 + 1.21644 w_k1_MSFT_s002
                                 + 1.32829 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.750491 w_k1_ADM_s002
                                 + 0.695865 w_k1_AXP_s002
                                 + 0.855474 w_k1_DIS_s002
                                 + 0.781711 w_k1_FDX_s002
                                 + 1.05363 w_k1_IBM_s002
                                 + 0.806746 w_k1_INTC_s002
                                 + 1.12416 w_k1_KO_s002 + 1.15817 w_k1_MAT_s002
                                 + 0.939044 w_k1_MSFT_s002
                                 + 0.981599 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.98545 w_k1_ADM_s002
                                 + 1.26281 w_k1_AXP_s002
                                 + 1.21733 w_k1_DIS_s002
                                 + 1.14157 w_k1_FDX_s002
                                 + 0.879721 w_k1_IBM_s002
                                 + 1.00806 w_k1_INTC_s002
                                 + 1.07223 w_k1_KO_s002 + 0.90076 w_k1_MAT_s002
                                 + 1.01287 w_k1_MSFT_s002
                                 + 1.14228 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.705063 w_k1_ADM_s002
                                 + 0.5847 w_k1_AXP_s002
                                 + 0.538223 w_k1_DIS_s002
                                 + 0.642744 w_k1_FDX_s002
                                 + 0.588912 w_k1_IBM_s002
                                 + 0.736189 w_k1_INTC_s002
                                 + 0.682451 w_k1_KO_s002
                                 + 0.393341 w_k1_MAT_s002
                                 + 0.566165 w_k1_MSFT_s002
                                 + 0.72088 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.05919 w_k1_ADM_s002
                                 + 1.06335 w_k1_AXP_s002
                                 + 1.11559 w_k1_DIS_s002
                                 + 0.871782 w_k1_FDX_s002
                                 + 1.44547 w_k1_IBM_s002
                                 + 1.00058 w_k1_INTC_s002
                                 + 1.33809 w_k1_KO_s002 + 1.06858 w_k1_MAT_s002
                                 + 0.904828 w_k1_MSFT_s002
                                 + 1.15712 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.33043 w_k1_ADM_s003
                                 + 1.34851 w_k1_AXP_s003
                                 + 0.999209 w_k1_DIS_s003
                                 + 1.06816 w_k1_FDX_s003
                                 + 1.36404 w_k1_IBM_s003
                                 + 1.15767 w_k1_INTC_s003
                                 + 1.34749 w_k1_KO_s003 + 1.78997 w_k1_MAT_s003
                                 + 1.05708 w_k1_MSFT_s003
                                 + 1.26602 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.17754 w_k1_ADM_s003
                                 + 0.914792 w_k1_AXP_s003
                                 + 0.862044 w_k1_DIS_s003
                                 + 0.982561 w_k1_FDX_s003
                                 + 1.01409 w_k1_IBM_s003
                                 + 0.855088 w_k1_INTC_s003
                                 + 1.02466 w_k1_KO_s003 + 1.02063 w_k1_MAT_s003
                                 + 0.926594 w_k1_MSFT_s003
                                 + 1.08352 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.44335 w_k1_ADM_s003
                                 + 1.59715 w_k1_AXP_s003
                                 + 1.70426 w_k1_DIS_s003
                                 + 1.34264 w_k1_FDX_s003
                                 + 1.18789 w_k1_IBM_s003
                                 + 1.10168 w_k1_INTC_s003
                                 + 1.41529 w_k1_KO_s003 + 1.20817 w_k1_MAT_s003
                                 + 1.21032 w_k1_MSFT_s003
                                 + 1.36534 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.10624 w_k1_ADM_s003
                                 + 1.16825 w_k1_AXP_s003
                                 + 1.07676 w_k1_DIS_s003
                                 + 1.11575 w_k1_FDX_s003
                                 + 0.814107 w_k1_IBM_s003
                                 + 1.01261 w_k1_INTC_s003
                                 + 1.04106 w_k1_KO_s003
                                 + 0.655851 w_k1_MAT_s003
                                 + 1.05353 w_k1_MSFT_s003
                                 + 1.14903 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.0171 w_k1_ADM_s003
                                 + 0.949941 w_k1_AXP_s003
                                 + 1.30055 w_k1_DIS_s003 + 1.0402 w_k1_FDX_s003
                                 + 1.23571 w_k1_IBM_s003
                                 + 0.983157 w_k1_INTC_s003
                                 + 1.36319 w_k1_KO_s003 + 1.44221 w_k1_MAT_s003
                                 + 1.06788 w_k1_MSFT_s003
                                 + 1.34914 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.09427 w_k1_ADM_s003
                                 + 1.10627 w_k1_AXP_s003
                                 + 0.938372 w_k1_DIS_s003
                                 + 1.01735 w_k1_FDX_s003
                                 + 0.691133 w_k1_IBM_s003
                                 + 0.935087 w_k1_INTC_s003
                                 + 0.853603 w_k1_KO_s003
                                 + 0.786788 w_k1_MAT_s003
                                 + 0.869823 w_k1_MSFT_s003
                                 + 0.843878 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.957298 w_k1_ADM_s003
                                 + 0.569431 w_k1_AXP_s003
                                 + 0.965695 w_k1_DIS_s003
                                 + 0.793401 w_k1_FDX_s003
                                 + 0.961679 w_k1_IBM_s003
                                 + 0.922892 w_k1_INTC_s003
                                 + 1.02745 w_k1_KO_s003
                                 + 0.791312 w_k1_MAT_s003
                                 + 1.0193 w_k1_MSFT_s003 + 1.01106 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.951902 w_k1_ADM_s003
                                 + 0.757179 w_k1_AXP_s003
                                 + 0.664053 w_k1_DIS_s003
                                 + 0.803027 w_k1_FDX_s003
                                 + 0.838099 w_k1_IBM_s003
                                 + 0.642636 w_k1_INTC_s003
                                 + 0.769019 w_k1_KO_s003
                                 + 0.832109 w_k1_MAT_s003
                                 + 0.658126 w_k1_MSFT_s003
                                 + 0.974238 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.38091 w_k1_ADM_s004
                                 + 1.39861 w_k1_AXP_s004
                                 + 1.44887 w_k1_DIS_s004
                                 + 1.16165 w_k1_FDX_s004
                                 + 1.47082 w_k1_IBM_s004
                                 + 0.863094 w_k1_INTC_s004
                                 + 1.33619 w_k1_KO_s004 + 2.42072 w_k1_MAT_s004
                                 + 1.14961 w_k1_MSFT_s004
                                 + 1.22144 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.830415 w_k1_ADM_s004
                                 + 0.585976 w_k1_AXP_s004
                                 + 0.607898 w_k1_DIS_s004
                                 + 0.699855 w_k1_FDX_s004
                                 + 0.615415 w_k1_IBM_s004
                                 + 0.578823 w_k1_INTC_s004
                                 + 0.604211 w_k1_KO_s004
                                 + 0.916687 w_k1_MAT_s004
                                 + 0.695181 w_k1_MSFT_s004
                                 + 0.668394 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.14099 w_k1_ADM_s004
                                 + 0.965075 w_k1_AXP_s004
                                 + 0.876359 w_k1_DIS_s004
                                 + 1.0731 w_k1_FDX_s004 + 0.79384 w_k1_IBM_s004
                                 + 0.99282 w_k1_INTC_s004
                                 + 0.893661 w_k1_KO_s004
                                 + 1.13854 w_k1_MAT_s004
                                 + 1.0328 w_k1_MSFT_s004 + 1.06496 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.23479 w_k1_ADM_s004
                                 + 1.3174 w_k1_AXP_s004 + 1.36444 w_k1_DIS_s004
                                 + 1.14433 w_k1_FDX_s004 + 1.0218 w_k1_IBM_s004
                                 + 1.08548 w_k1_INTC_s004
                                 + 1.10085 w_k1_KO_s004 + 1.33442 w_k1_MAT_s004
                                 + 1.09072 w_k1_MSFT_s004
                                 + 1.04328 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.815602 w_k1_ADM_s004
                                 + 0.620735 w_k1_AXP_s004
                                 + 0.674494 w_k1_DIS_s004
                                 + 0.821303 w_k1_FDX_s004
                                 + 0.653816 w_k1_IBM_s004
                                 + 0.727803 w_k1_INTC_s004
                                 + 0.716703 w_k1_KO_s004
                                 + 0.744185 w_k1_MAT_s004
                                 + 0.786927 w_k1_MSFT_s004
                                 + 0.91505 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.921565 w_k1_ADM_s004
                                 + 0.802815 w_k1_AXP_s004
                                 + 1.00344 w_k1_DIS_s004
                                 + 0.904385 w_k1_FDX_s004
                                 + 0.962782 w_k1_IBM_s004
                                 + 0.844788 w_k1_INTC_s004
                                 + 0.947422 w_k1_KO_s004
                                 + 1.05731 w_k1_MAT_s004
                                 + 0.862173 w_k1_MSFT_s004
                                 + 1.03364 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.930559 w_k1_ADM_s004
                                 + 0.751309 w_k1_AXP_s004
                                 + 0.92495 w_k1_DIS_s004
                                 + 0.923124 w_k1_FDX_s004
                                 + 0.788198 w_k1_IBM_s004
                                 + 0.939518 w_k1_INTC_s004
                                 + 0.85423 w_k1_KO_s004
                                 + 0.889801 w_k1_MAT_s004
                                 + 0.943352 w_k1_MSFT_s004
                                 + 0.866151 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.725101 w_k1_ADM_s004
                                 + 0.799162 w_k1_AXP_s004
                                 + 0.766549 w_k1_DIS_s004
                                 + 0.793006 w_k1_FDX_s004
                                 + 1.16323 w_k1_IBM_s004
                                 + 0.638477 w_k1_INTC_s004
                                 + 0.911967 w_k1_KO_s004
                                 + 1.70855 w_k1_MAT_s004
                                 + 0.745601 w_k1_MSFT_s004
                                 + 0.908693 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.77883 w_k1_ADM_s005
                                 + 0.766075 w_k1_AXP_s005
                                 + 0.5547 w_k1_DIS_s005
                                 + 0.703674 w_k1_FDX_s005
                                 + 1.00936 w_k1_IBM_s005
                                 + 0.622321 w_k1_INTC_s005
                                 + 0.890052 w_k1_KO_s005
                                 + 0.911582 w_k1_MAT_s005
                                 + 0.631177 w_k1_MSFT_s005
                                 + 0.856178 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.00082 w_k1_ADM_s005
                                 + 1.14446 w_k1_AXP_s005
                                 + 1.77608 w_k1_DIS_s005
                                 + 1.26112 w_k1_FDX_s005
                                 + 1.74683 w_k1_IBM_s005
                                 + 1.24035 w_k1_INTC_s005
                                 + 1.73073 w_k1_KO_s005 + 1.60746 w_k1_MAT_s005
                                 + 1.20677 w_k1_MSFT_s005
                                 + 1.23523 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.33605 w_k1_ADM_s005
                                 + 1.5463 w_k1_AXP_s005 + 1.47191 w_k1_DIS_s005
                                 + 1.4279 w_k1_FDX_s005 + 1.13961 w_k1_IBM_s005
                                 + 1.28732 w_k1_INTC_s005
                                 + 1.40945 w_k1_KO_s005 + 1.09036 w_k1_MAT_s005
                                 + 1.29287 w_k1_MSFT_s005
                                 + 1.43964 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.685588 w_k1_ADM_s005
                                 + 0.794594 w_k1_AXP_s005
                                 + 0.702801 w_k1_DIS_s005
                                 + 0.889569 w_k1_FDX_s005
                                 + 0.687275 w_k1_IBM_s005
                                 + 0.998619 w_k1_INTC_s005
                                 + 0.774333 w_k1_KO_s005
                                 + 0.604681 w_k1_MAT_s005
                                 + 0.797332 w_k1_MSFT_s005
                                 + 0.94628 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.18458 w_k1_ADM_s005
                                 + 0.845434 w_k1_AXP_s005
                                 + 1.2283 w_k1_DIS_s005 + 1.06281 w_k1_FDX_s005
                                 + 0.941722 w_k1_IBM_s005
                                 + 1.15614 w_k1_INTC_s005
                                 + 1.02754 w_k1_KO_s005
                                 + 0.607839 w_k1_MAT_s005
                                 + 1.09136 w_k1_MSFT_s005
                                 + 1.13604 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.926756 w_k1_ADM_s005
                                 + 1.02691 w_k1_AXP_s005
                                 + 0.970567 w_k1_DIS_s005
                                 + 1.08338 w_k1_FDX_s005 + 1.1213 w_k1_IBM_s005
                                 + 0.971628 w_k1_INTC_s005
                                 + 1.08899 w_k1_KO_s005 + 1.08105 w_k1_MAT_s005
                                 + 0.959047 w_k1_MSFT_s005
                                 + 1.09071 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.805842 w_k1_ADM_s005
                                 + 0.743869 w_k1_AXP_s005
                                 + 0.857729 w_k1_DIS_s005
                                 + 0.862324 w_k1_FDX_s005
                                 + 0.999074 w_k1_IBM_s005
                                 + 0.945847 w_k1_INTC_s005
                                 + 1.01873 w_k1_KO_s005 + 1.02765 w_k1_MAT_s005
                                 + 0.918996 w_k1_MSFT_s005
                                 + 1.00489 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.869324 w_k1_ADM_s005
                                 + 0.826641 w_k1_AXP_s005
                                 + 0.75434 w_k1_DIS_s005
                                 + 0.787832 w_k1_FDX_s005
                                 + 1.33726 w_k1_IBM_s005
                                 + 0.912947 w_k1_INTC_s005
                                 + 1.03802 w_k1_KO_s005
                                 + 0.891371 w_k1_MAT_s005
                                 + 0.730876 w_k1_MSFT_s005
                                 + 1.01454 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.699529 w_k1_ADM_s006
                                 + 0.556541 w_k1_AXP_s006
                                 + 0.560209 w_k1_DIS_s006
                                 + 0.653186 w_k1_FDX_s006
                                 + 0.669254 w_k1_IBM_s006
                                 + 0.713274 w_k1_INTC_s006
                                 + 0.668411 w_k1_KO_s006
                                 + 0.657963 w_k1_MAT_s006
                                 + 0.685796 w_k1_MSFT_s006
                                 + 0.740106 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.15845 w_k1_ADM_s006
                                 + 1.14554 w_k1_AXP_s006
                                 + 1.34171 w_k1_DIS_s006
                                 + 1.07797 w_k1_FDX_s006
                                 + 1.37934 w_k1_IBM_s006
                                 + 1.16282 w_k1_INTC_s006
                                 + 1.25673 w_k1_KO_s006 + 1.17624 w_k1_MAT_s006
                                 + 1.16247 w_k1_MSFT_s006
                                 + 1.10679 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.28001 w_k1_ADM_s006
                                 + 1.18598 w_k1_AXP_s006
                                 + 1.13167 w_k1_DIS_s006
                                 + 1.05856 w_k1_FDX_s006
                                 + 0.863297 w_k1_IBM_s006
                                 + 1.12914 w_k1_INTC_s006
                                 + 1.01381 w_k1_KO_s006
                                 + 0.845009 w_k1_MAT_s006
                                 + 1.06403 w_k1_MSFT_s006
                                 + 1.01325 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.985029 w_k1_ADM_s006
                                 + 0.980974 w_k1_AXP_s006
                                 + 0.899084 w_k1_DIS_s006
                                 + 0.888765 w_k1_FDX_s006
                                 + 1.20893 w_k1_IBM_s006
                                 + 0.984772 w_k1_INTC_s006
                                 + 0.902586 w_k1_KO_s006
                                 + 0.846659 w_k1_MAT_s006
                                 + 0.907973 w_k1_MSFT_s006
                                 + 0.935574 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.409 w_k1_ADM_s006
                                 + 1.63279 w_k1_AXP_s006
                                 + 1.45895 w_k1_DIS_s006
                                 + 1.38847 w_k1_FDX_s006
                                 + 1.15661 w_k1_IBM_s006
                                 + 1.32359 w_k1_INTC_s006
                                 + 1.22522 w_k1_KO_s006 + 1.01149 w_k1_MAT_s006
                                 + 1.25539 w_k1_MSFT_s006
                                 + 0.990107 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.01446 w_k1_ADM_s006
                                 + 0.755083 w_k1_AXP_s006
                                 + 0.901088 w_k1_DIS_s006
                                 + 0.883708 w_k1_FDX_s006
                                 + 0.782492 w_k1_IBM_s006
                                 + 0.907456 w_k1_INTC_s006
                                 + 0.846158 w_k1_KO_s006
                                 + 0.527464 w_k1_MAT_s006
                                 + 0.960291 w_k1_MSFT_s006
                                 + 0.873033 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.809365 w_k1_ADM_s006
                                 + 0.83007 w_k1_AXP_s006
                                 + 0.643669 w_k1_DIS_s006
                                 + 0.791522 w_k1_FDX_s006
                                 + 0.873177 w_k1_IBM_s006
                                 + 0.954333 w_k1_INTC_s006
                                 + 0.800082 w_k1_KO_s006
                                 + 0.861316 w_k1_MAT_s006
                                 + 0.842425 w_k1_MSFT_s006
                                 + 0.836221 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.22626 w_k1_ADM_s006
                                 + 1.50651 w_k1_AXP_s006 + 1.4658 w_k1_DIS_s006
                                 + 1.19731 w_k1_FDX_s006
                                 + 1.67105 w_k1_IBM_s006
                                 + 1.54622 w_k1_INTC_s006
                                 + 1.38763 w_k1_KO_s006 + 1.63792 w_k1_MAT_s006
                                 + 1.12963 w_k1_MSFT_s006
                                 + 1.14202 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.802094 w_k1_ADM_s007
                                 + 0.848483 w_k1_AXP_s007
                                 + 0.606415 w_k1_DIS_s007
                                 + 0.798137 w_k1_FDX_s007
                                 + 0.554188 w_k1_IBM_s007
                                 + 0.713546 w_k1_INTC_s007
                                 + 0.516029 w_k1_KO_s007
                                 + 0.578552 w_k1_MAT_s007
                                 + 0.689868 w_k1_MSFT_s007
                                 + 0.805318 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.30974 w_k1_ADM_s007
                                 + 1.51653 w_k1_AXP_s007
                                 + 1.51218 w_k1_DIS_s007
                                 + 1.41791 w_k1_FDX_s007
                                 + 0.705942 w_k1_IBM_s007
                                 + 1.22333 w_k1_INTC_s007
                                 + 1.04139 w_k1_KO_s007 + 1.0988 w_k1_MAT_s007
                                 + 1.28121 w_k1_MSFT_s007
                                 + 1.17459 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.818541 w_k1_ADM_s007
                                 + 0.913226 w_k1_AXP_s007
                                 + 0.898761 w_k1_DIS_s007
                                 + 0.931619 w_k1_FDX_s007
                                 + 0.687214 w_k1_IBM_s007
                                 + 0.93131 w_k1_INTC_s007
                                 + 0.830272 w_k1_KO_s007
                                 + 0.816512 w_k1_MAT_s007
                                 + 0.925375 w_k1_MSFT_s007
                                 + 0.858929 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.773051 w_k1_ADM_s007
                                 + 0.887716 w_k1_AXP_s007
                                 + 0.736763 w_k1_DIS_s007
                                 + 0.818983 w_k1_FDX_s007
                                 + 0.993155 w_k1_IBM_s007
                                 + 0.905293 w_k1_INTC_s007
                                 + 0.770713 w_k1_KO_s007
                                 + 0.876695 w_k1_MAT_s007
                                 + 0.866551 w_k1_MSFT_s007
                                 + 0.849184 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.1124 w_k1_ADM_s007
                                 + 1.47441 w_k1_AXP_s007
                                 + 1.42105 w_k1_DIS_s007
                                 + 1.29895 w_k1_FDX_s007
                                 + 1.25467 w_k1_IBM_s007
                                 + 1.11545 w_k1_INTC_s007
                                 + 1.21159 w_k1_KO_s007 + 1.83309 w_k1_MAT_s007
                                 + 1.215 w_k1_MSFT_s007 + 1.14241 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.917591 w_k1_ADM_s007
                                 + 1.16727 w_k1_AXP_s007
                                 + 1.07689 w_k1_DIS_s007
                                 + 0.963509 w_k1_FDX_s007
                                 + 1.69804 w_k1_IBM_s007
                                 + 1.10119 w_k1_INTC_s007
                                 + 1.09233 w_k1_KO_s007 + 1.60681 w_k1_MAT_s007
                                 + 1.0961 w_k1_MSFT_s007
                                 + 0.968262 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.08168 w_k1_ADM_s007
                                 + 1.36759 w_k1_AXP_s007
                                 + 0.939767 w_k1_DIS_s007
                                 + 1.07784 w_k1_FDX_s007
                                 + 1.04486 w_k1_IBM_s007
                                 + 1.07746 w_k1_INTC_s007
                                 + 0.890605 w_k1_KO_s007
                                 + 1.07652 w_k1_MAT_s007
                                 + 1.01446 w_k1_MSFT_s007 + 1.003 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.843404 w_k1_ADM_s007
                                 + 1.03372 w_k1_AXP_s007
                                 + 1.17269 w_k1_DIS_s007
                                 + 1.06335 w_k1_FDX_s007
                                 + 0.809417 w_k1_IBM_s007
                                 + 1.20283 w_k1_INTC_s007
                                 + 0.948477 w_k1_KO_s007
                                 + 0.999683 w_k1_MAT_s007
                                 + 1.17043 w_k1_MSFT_s007
                                 + 1.00297 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.8027 w_k1_ADM_s010
                                 + 0.771557 w_k1_AXP_s010
                                 + 0.801227 w_k1_DIS_s010
                                 + 0.864518 w_k1_FDX_s010
                                 + 0.733422 w_k1_IBM_s010
                                 + 0.808776 w_k1_INTC_s010
                                 + 0.766972 w_k1_KO_s010
                                 + 1.04264 w_k1_MAT_s010
                                 + 0.923104 w_k1_MSFT_s010
                                 + 1.05227 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.805587 w_k1_ADM_s010
                                 + 0.74697 w_k1_AXP_s010
                                 + 0.763601 w_k1_DIS_s010
                                 + 0.767035 w_k1_FDX_s010
                                 + 0.991501 w_k1_IBM_s010
                                 + 0.707286 w_k1_INTC_s010
                                 + 0.749838 w_k1_KO_s010
                                 + 0.99805 w_k1_MAT_s010
                                 + 0.735834 w_k1_MSFT_s010
                                 + 0.963663 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.974855 w_k1_ADM_s010
                                 + 0.993669 w_k1_AXP_s010
                                 + 0.921846 w_k1_DIS_s010
                                 + 1.08327 w_k1_FDX_s010
                                 + 0.923279 w_k1_IBM_s010
                                 + 0.942048 w_k1_INTC_s010
                                 + 0.880134 w_k1_KO_s010
                                 + 1.08244 w_k1_MAT_s010
                                 + 0.964317 w_k1_MSFT_s010
                                 + 1.07254 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.666828 w_k1_ADM_s010
                                 + 0.699095 w_k1_AXP_s010
                                 + 0.813872 w_k1_DIS_s010
                                 + 0.788748 w_k1_FDX_s010
                                 + 0.848816 w_k1_IBM_s010
                                 + 0.839448 w_k1_INTC_s010
                                 + 0.825909 w_k1_KO_s010
                                 + 0.654029 w_k1_MAT_s010
                                 + 0.806249 w_k1_MSFT_s010
                                 + 0.818396 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.07024 w_k1_ADM_s010
                                 + 1.0237 w_k1_AXP_s010 + 1.40275 w_k1_DIS_s010
                                 + 1.23694 w_k1_FDX_s010
                                 + 0.863173 w_k1_IBM_s010
                                 + 1.01756 w_k1_INTC_s010
                                 + 0.842499 w_k1_KO_s010
                                 + 0.794203 w_k1_MAT_s010
                                 + 1.09853 w_k1_MSFT_s010
                                 + 1.08168 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.952712 w_k1_ADM_s010
                                 + 0.708737 w_k1_AXP_s010
                                 + 0.753839 w_k1_DIS_s010
                                 + 0.916692 w_k1_FDX_s010
                                 + 0.593463 w_k1_IBM_s010
                                 + 0.740442 w_k1_INTC_s010
                                 + 0.589936 w_k1_KO_s010
                                 + 0.459301 w_k1_MAT_s010
                                 + 0.730259 w_k1_MSFT_s010
                                 + 0.763634 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.908566 w_k1_ADM_s010
                                 + 1.1758 w_k1_AXP_s010 + 1.38445 w_k1_DIS_s010
                                 + 1.0537 w_k1_FDX_s010 + 1.35141 w_k1_IBM_s010
                                 + 1.10416 w_k1_INTC_s010
                                 + 1.23502 w_k1_KO_s010 + 1.34595 w_k1_MAT_s010
                                 + 1.11675 w_k1_MSFT_s010
                                 + 1.10238 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.06224 w_k1_ADM_s010
                                 + 0.889384 w_k1_AXP_s010
                                 + 1.30216 w_k1_DIS_s010
                                 + 1.09361 w_k1_FDX_s010
                                 + 0.96963 w_k1_IBM_s010
                                 + 0.895296 w_k1_INTC_s010
                                 + 0.993669 w_k1_KO_s010
                                 + 1.10682 w_k1_MAT_s010
                                 + 0.992718 w_k1_MSFT_s010
                                 + 1.06745 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.28158 w_k1_ADM_s011
                                 + 0.966689 w_k1_AXP_s011
                                 + 0.903953 w_k1_DIS_s011
                                 + 1.14591 w_k1_FDX_s011
                                 + 0.674272 w_k1_IBM_s011
                                 + 1.32579 w_k1_INTC_s011
                                 + 0.819893 w_k1_KO_s011
                                 + 0.618381 w_k1_MAT_s011
                                 + 1.17327 w_k1_MSFT_s011
                                 + 0.936537 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.02254 w_k1_ADM_s011
                                 + 1.19749 w_k1_AXP_s011
                                 + 1.00995 w_k1_DIS_s011
                                 + 1.08334 w_k1_FDX_s011 + 1.0885 w_k1_IBM_s011
                                 + 1.2419 w_k1_INTC_s011
                                 + 0.927719 w_k1_KO_s011
                                 + 1.17119 w_k1_MAT_s011
                                 + 1.14357 w_k1_MSFT_s011
                                 + 1.06636 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.988367 w_k1_ADM_s011
                                 + 0.77433 w_k1_AXP_s011
                                 + 0.648709 w_k1_DIS_s011
                                 + 0.748886 w_k1_FDX_s011
                                 + 0.892687 w_k1_IBM_s011
                                 + 0.866452 w_k1_INTC_s011
                                 + 0.78083 w_k1_KO_s011
                                 + 0.821037 w_k1_MAT_s011
                                 + 0.823699 w_k1_MSFT_s011
                                 + 0.852715 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.23824 w_k1_ADM_s011
                                 + 1.0776 w_k1_AXP_s011
                                 + 0.964429 w_k1_DIS_s011
                                 + 1.01192 w_k1_FDX_s011
                                 + 0.890347 w_k1_IBM_s011
                                 + 1.08176 w_k1_INTC_s011
                                 + 0.915522 w_k1_KO_s011
                                 + 0.93888 w_k1_MAT_s011
                                 + 0.952976 w_k1_MSFT_s011
                                 + 0.879219 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.62013 w_k1_ADM_s011
                                 + 0.981808 w_k1_AXP_s011
                                 + 1.38901 w_k1_DIS_s011
                                 + 1.19082 w_k1_FDX_s011 + 1.0626 w_k1_IBM_s011
                                 + 1.28689 w_k1_INTC_s011
                                 + 1.27455 w_k1_KO_s011
                                 + 0.682083 w_k1_MAT_s011
                                 + 1.28369 w_k1_MSFT_s011
                                 + 1.16892 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.01521 w_k1_ADM_s011
                                 + 0.794191 w_k1_AXP_s011
                                 + 1.04529 w_k1_DIS_s011
                                 + 0.921522 w_k1_FDX_s011
                                 + 0.89266 w_k1_IBM_s011
                                 + 0.793259 w_k1_INTC_s011
                                 + 0.935051 w_k1_KO_s011
                                 + 0.850665 w_k1_MAT_s011
                                 + 0.886086 w_k1_MSFT_s011
                                 + 0.939957 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.12457 w_k1_ADM_s011
                                 + 1.05788 w_k1_AXP_s011
                                 + 1.06248 w_k1_DIS_s011
                                 + 0.885866 w_k1_FDX_s011
                                 + 1.35897 w_k1_IBM_s011
                                 + 0.945024 w_k1_INTC_s011
                                 + 0.993778 w_k1_KO_s011
                                 + 1.51848 w_k1_MAT_s011
                                 + 0.98344 w_k1_MSFT_s011
                                 + 1.03928 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.23992 w_k1_ADM_s011
                                 + 1.52298 w_k1_AXP_s011
                                 + 1.57174 w_k1_DIS_s011
                                 + 1.32869 w_k1_FDX_s011
                                 + 1.57291 w_k1_IBM_s011
                                 + 1.38912 w_k1_INTC_s011 + 1.3727 w_k1_KO_s011
                                 + 1.78762 w_k1_MAT_s011 + 1.334 w_k1_MSFT_s011
                                 + 1.22806 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.11071 w_k1_ADM_s012
                                 + 1.02165 w_k1_AXP_s012
                                 + 1.16034 w_k1_DIS_s012 + 1.0322 w_k1_FDX_s012
                                 + 1.01413 w_k1_IBM_s012
                                 + 1.18976 w_k1_INTC_s012
                                 + 1.28759 w_k1_KO_s012 + 1.30171 w_k1_MAT_s012
                                 + 1.02983 w_k1_MSFT_s012 + 1.1553 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.945101 w_k1_ADM_s012
                                 + 0.691589 w_k1_AXP_s012
                                 + 0.686049 w_k1_DIS_s012
                                 + 0.83747 w_k1_FDX_s012
                                 + 0.713454 w_k1_IBM_s012
                                 + 0.905454 w_k1_INTC_s012
                                 + 0.787004 w_k1_KO_s012
                                 + 0.553225 w_k1_MAT_s012
                                 + 0.817256 w_k1_MSFT_s012
                                 + 0.925555 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.14241 w_k1_ADM_s012
                                 + 1.46012 w_k1_AXP_s012
                                 + 1.10144 w_k1_DIS_s012
                                 + 1.18772 w_k1_FDX_s012
                                 + 0.861966 w_k1_IBM_s012
                                 + 1.06556 w_k1_INTC_s012
                                 + 0.982191 w_k1_KO_s012 + 1.1462 w_k1_MAT_s012
                                 + 1.00739 w_k1_MSFT_s012 + 1.2891 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.37346 w_k1_ADM_s012
                                 + 1.42823 w_k1_AXP_s012
                                 + 1.80325 w_k1_DIS_s012
                                 + 1.40499 w_k1_FDX_s012
                                 + 1.67834 w_k1_IBM_s012
                                 + 1.6655 w_k1_INTC_s012 + 1.62884 w_k1_KO_s012
                                 + 1.56549 w_k1_MAT_s012
                                 + 1.40448 w_k1_MSFT_s012
                                 + 1.60102 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.4299 w_k1_ADM_s012
                                 + 1.0895 w_k1_AXP_s012 + 1.40895 w_k1_DIS_s012
                                 + 1.26575 w_k1_FDX_s012
                                 + 1.11083 w_k1_IBM_s012
                                 + 1.35002 w_k1_INTC_s012
                                 + 1.46722 w_k1_KO_s012 + 1.03573 w_k1_MAT_s012
                                 + 1.25889 w_k1_MSFT_s012
                                 + 1.40282 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.98808 w_k1_ADM_s012
                                 + 1.18033 w_k1_AXP_s012 + 1.1539 w_k1_DIS_s012
                                 + 1.01674 w_k1_FDX_s012
                                 + 1.42894 w_k1_IBM_s012
                                 + 1.21232 w_k1_INTC_s012
                                 + 1.34999 w_k1_KO_s012 + 1.2065 w_k1_MAT_s012
                                 + 1.06099 w_k1_MSFT_s012
                                 + 1.22933 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.915987 w_k1_ADM_s012
                                 + 0.873041 w_k1_AXP_s012
                                 + 0.823871 w_k1_DIS_s012
                                 + 0.881816 w_k1_FDX_s012
                                 + 1.02177 w_k1_IBM_s012
                                 + 0.814382 w_k1_INTC_s012
                                 + 1.03343 w_k1_KO_s012
                                 + 0.932988 w_k1_MAT_s012
                                 + 0.927224 w_k1_MSFT_s012
                                 + 0.960405 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.00434 w_k1_ADM_s012
                                 + 1.08868 w_k1_AXP_s012
                                 + 0.950123 w_k1_DIS_s012
                                 + 0.874138 w_k1_FDX_s012
                                 + 1.29187 w_k1_IBM_s012
                                 + 0.88375 w_k1_INTC_s012
                                 + 1.19239 w_k1_KO_s012 + 1.31147 w_k1_MAT_s012
                                 + 0.904618 w_k1_MSFT_s012
                                 + 1.11368 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.05966 w_k1_ADM_s013
                                 + 1.06929 w_k1_AXP_s013
                                 + 1.34481 w_k1_DIS_s013 + 1.289 w_k1_FDX_s013
                                 + 1.06415 w_k1_IBM_s013 + 1.427 w_k1_INTC_s013
                                 + 1.2172 w_k1_KO_s013 + 1.40836 w_k1_MAT_s013
                                 + 1.24651 w_k1_MSFT_s013
                                 + 1.16614 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.06794 w_k1_ADM_s013
                                 + 1.00875 w_k1_AXP_s013
                                 + 1.06864 w_k1_DIS_s013
                                 + 1.05894 w_k1_FDX_s013
                                 + 0.971015 w_k1_IBM_s013
                                 + 1.05928 w_k1_INTC_s013
                                 + 1.04977 w_k1_KO_s013 + 1.11704 w_k1_MAT_s013
                                 + 1.05337 w_k1_MSFT_s013
                                 + 1.12371 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.919105 w_k1_ADM_s013
                                 + 0.905687 w_k1_AXP_s013
                                 + 0.761803 w_k1_DIS_s013
                                 + 0.754237 w_k1_FDX_s013
                                 + 1.54348 w_k1_IBM_s013
                                 + 0.912348 w_k1_INTC_s013
                                 + 1.1063 w_k1_KO_s013 + 1.37026 w_k1_MAT_s013
                                 + 0.820827 w_k1_MSFT_s013
                                 + 1.01894 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.806996 w_k1_ADM_s013
                                 + 0.816528 w_k1_AXP_s013
                                 + 0.833174 w_k1_DIS_s013
                                 + 0.903227 w_k1_FDX_s013
                                 + 1.29073 w_k1_IBM_s013
                                 + 1.09972 w_k1_INTC_s013
                                 + 1.23652 w_k1_KO_s013 + 1.47232 w_k1_MAT_s013
                                 + 0.978386 w_k1_MSFT_s013
                                 + 1.01819 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.10402 w_k1_ADM_s013
                                 + 1.18872 w_k1_AXP_s013
                                 + 1.24779 w_k1_DIS_s013
                                 + 1.24918 w_k1_FDX_s013
                                 + 1.24015 w_k1_IBM_s013
                                 + 1.09249 w_k1_INTC_s013
                                 + 1.11475 w_k1_KO_s013 + 1.25657 w_k1_MAT_s013
                                 + 1.13186 w_k1_MSFT_s013
                                 + 1.13635 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.643738 w_k1_ADM_s013
                                 + 0.700857 w_k1_AXP_s013
                                 + 0.693996 w_k1_DIS_s013
                                 + 0.831037 w_k1_FDX_s013
                                 + 0.735086 w_k1_IBM_s013
                                 + 0.742518 w_k1_INTC_s013
                                 + 0.664365 w_k1_KO_s013
                                 + 0.818702 w_k1_MAT_s013
                                 + 0.656245 w_k1_MSFT_s013
                                 + 0.868008 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.783496 w_k1_ADM_s013
                                 + 1.50635 w_k1_AXP_s013
                                 + 1.69273 w_k1_DIS_s013
                                 + 1.35863 w_k1_FDX_s013
                                 + 1.39578 w_k1_IBM_s013
                                 + 1.59813 w_k1_INTC_s013
                                 + 1.40483 w_k1_KO_s013 + 1.85883 w_k1_MAT_s013
                                 + 1.18481 w_k1_MSFT_s013 + 1.1657 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.856416 w_k1_ADM_s013
                                 + 0.717566 w_k1_AXP_s013
                                 + 1.01792 w_k1_DIS_s013
                                 + 0.959108 w_k1_FDX_s013
                                 + 0.820437 w_k1_IBM_s013
                                 + 1.12039 w_k1_INTC_s013
                                 + 0.856328 w_k1_KO_s013
                                 + 0.747026 w_k1_MAT_s013
                                 + 0.959901 w_k1_MSFT_s013
                                 + 0.961418 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.756104 w_k1_ADM_s014
                                 + 0.98926 w_k1_AXP_s014
                                 + 0.702394 w_k1_DIS_s014
                                 + 0.885789 w_k1_FDX_s014
                                 + 0.980585 w_k1_IBM_s014
                                 + 0.997226 w_k1_INTC_s014
                                 + 0.731195 w_k1_KO_s014
                                 + 0.793502 w_k1_MAT_s014
                                 + 0.840303 w_k1_MSFT_s014
                                 + 0.918114 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.911758 w_k1_ADM_s014
                                 + 1.35142 w_k1_AXP_s014
                                 + 1.19544 w_k1_DIS_s014
                                 + 1.11681 w_k1_FDX_s014
                                 + 1.61213 w_k1_IBM_s014
                                 + 1.23974 w_k1_INTC_s014 + 1.3397 w_k1_KO_s014
                                 + 1.90099 w_k1_MAT_s014
                                 + 1.29721 w_k1_MSFT_s014
                                 + 1.05793 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.814406 w_k1_ADM_s014
                                 + 0.916917 w_k1_AXP_s014
                                 + 0.707104 w_k1_DIS_s014
                                 + 0.951844 w_k1_FDX_s014
                                 + 0.594081 w_k1_IBM_s014
                                 + 1.31885 w_k1_INTC_s014
                                 + 0.736061 w_k1_KO_s014
                                 + 0.681867 w_k1_MAT_s014
                                 + 1.05233 w_k1_MSFT_s014
                                 + 0.824419 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.870044 w_k1_ADM_s014
                                 + 0.997737 w_k1_AXP_s014
                                 + 0.825766 w_k1_DIS_s014
                                 + 0.892249 w_k1_FDX_s014
                                 + 1.27947 w_k1_IBM_s014
                                 + 1.14949 w_k1_INTC_s014
                                 + 0.975441 w_k1_KO_s014
                                 + 1.22715 w_k1_MAT_s014
                                 + 1.05612 w_k1_MSFT_s014 + 1.0482 w_k1_PG_s014
                                 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.25505 w_k1_ADM_s014
                                 + 1.35645 w_k1_AXP_s014
                                 + 1.15145 w_k1_DIS_s014
                                 + 1.25022 w_k1_FDX_s014
                                 + 0.857572 w_k1_IBM_s014
                                 + 1.50023 w_k1_INTC_s014
                                 + 0.957148 w_k1_KO_s014
                                 + 0.925744 w_k1_MAT_s014
                                 + 1.27679 w_k1_MSFT_s014
                                 + 0.945624 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.49692 w_k1_ADM_s014
                                 + 1.82157 w_k1_AXP_s014
                                 + 1.61956 w_k1_DIS_s014
                                 + 1.56305 w_k1_FDX_s014
                                 + 1.24448 w_k1_IBM_s014
                                 + 1.73814 w_k1_INTC_s014
                                 + 1.39247 w_k1_KO_s014 + 1.11231 w_k1_MAT_s014
                                 + 1.4409 w_k1_MSFT_s014 + 1.2167 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14668 w_k1_ADM_s014
                                 + 0.927418 w_k1_AXP_s014
                                 + 0.921812 w_k1_DIS_s014
                                 + 1.05659 w_k1_FDX_s014
                                 + 0.869861 w_k1_IBM_s014
                                 + 1.206 w_k1_INTC_s014 + 0.92247 w_k1_KO_s014
                                 + 0.794214 w_k1_MAT_s014
                                 + 1.14391 w_k1_MSFT_s014
                                 + 0.92892 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.711945 w_k1_ADM_s014
                                 + 0.559349 w_k1_AXP_s014
                                 + 0.673947 w_k1_DIS_s014
                                 + 0.788278 w_k1_FDX_s014
                                 + 0.760888 w_k1_IBM_s014
                                 + 1.01333 w_k1_INTC_s014
                                 + 0.832471 w_k1_KO_s014
                                 + 0.564594 w_k1_MAT_s014
                                 + 0.947838 w_k1_MSFT_s014
                                 + 0.73613 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.959054 w_k1_ADM_s015
                                 + 1.13969 w_k1_AXP_s015 + 1.113 w_k1_DIS_s015
                                 + 0.996186 w_k1_FDX_s015
                                 + 1.56913 w_k1_IBM_s015
                                 + 0.952628 w_k1_INTC_s015
                                 + 1.24714 w_k1_KO_s015 + 1.70754 w_k1_MAT_s015
                                 + 1.11277 w_k1_MSFT_s015
                                 + 1.03321 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.05066 w_k1_ADM_s015
                                 + 0.839333 w_k1_AXP_s015
                                 + 0.930522 w_k1_DIS_s015
                                 + 0.828736 w_k1_FDX_s015
                                 + 1.32596 w_k1_IBM_s015
                                 + 0.743359 w_k1_INTC_s015
                                 + 1.19988 w_k1_KO_s015 + 1.36505 w_k1_MAT_s015
                                 + 0.982224 w_k1_MSFT_s015
                                 + 0.946183 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.28465 w_k1_ADM_s015
                                 + 0.975168 w_k1_AXP_s015
                                 + 0.76311 w_k1_DIS_s015
                                 + 0.919289 w_k1_FDX_s015
                                 + 1.00447 w_k1_IBM_s015
                                 + 0.843116 w_k1_INTC_s015
                                 + 0.839358 w_k1_KO_s015 + 1.1359 w_k1_MAT_s015
                                 + 0.851737 w_k1_MSFT_s015
                                 + 0.941302 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.19503 w_k1_ADM_s015
                                 + 0.815387 w_k1_AXP_s015
                                 + 1.1991 w_k1_DIS_s015 + 1.13708 w_k1_FDX_s015
                                 + 0.763018 w_k1_IBM_s015
                                 + 1.21231 w_k1_INTC_s015
                                 + 1.08184 w_k1_KO_s015
                                 + 0.546764 w_k1_MAT_s015
                                 + 1.329 w_k1_MSFT_s015 + 0.884313 w_k1_PG_s015
                                 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.968137 w_k1_ADM_s015
                                 + 0.629539 w_k1_AXP_s015
                                 + 0.732162 w_k1_DIS_s015
                                 + 0.816217 w_k1_FDX_s015
                                 + 0.951273 w_k1_IBM_s015
                                 + 0.997341 w_k1_INTC_s015
                                 + 0.909155 w_k1_KO_s015
                                 + 0.805158 w_k1_MAT_s015
                                 + 0.958648 w_k1_MSFT_s015
                                 + 0.854929 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.4999 w_k1_ADM_s015
                                 + 1.08601 w_k1_AXP_s015
                                 + 1.64323 w_k1_DIS_s015
                                 + 1.29128 w_k1_FDX_s015
                                 + 1.30461 w_k1_IBM_s015
                                 + 1.3118 w_k1_INTC_s015 + 1.4785 w_k1_KO_s015
                                 + 1.20881 w_k1_MAT_s015
                                 + 1.51594 w_k1_MSFT_s015
                                 + 1.31085 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.13155 w_k1_ADM_s015
                                 + 0.612072 w_k1_AXP_s015
                                 + 0.639131 w_k1_DIS_s015
                                 + 0.812387 w_k1_FDX_s015
                                 + 0.61051 w_k1_IBM_s015
                                 + 0.745372 w_k1_INTC_s015
                                 + 0.722273 w_k1_KO_s015
                                 + 0.461293 w_k1_MAT_s015
                                 + 0.867908 w_k1_MSFT_s015
                                 + 0.817003 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.37164 w_k1_ADM_s015
                                 + 1.47495 w_k1_AXP_s015 + 1.3041 w_k1_DIS_s015
                                 + 1.42112 w_k1_FDX_s015
                                 + 1.01222 w_k1_IBM_s015
                                 + 1.25621 w_k1_INTC_s015
                                 + 1.15474 w_k1_KO_s015 + 1.01966 w_k1_MAT_s015
                                 + 1.22948 w_k1_MSFT_s015
                                 + 1.17773 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.994832 w_k1_ADM_s016
                                 + 1.12483 w_k1_AXP_s016 + 1.0033 w_k1_DIS_s016
                                 + 1.05892 w_k1_FDX_s016
                                 + 0.822095 w_k1_IBM_s016
                                 + 0.846721 w_k1_INTC_s016
                                 + 0.813287 w_k1_KO_s016
                                 + 0.978987 w_k1_MAT_s016
                                 + 1.0209 w_k1_MSFT_s016 + 0.97616 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.767827 w_k1_ADM_s016
                                 + 0.828332 w_k1_AXP_s016
                                 + 0.734472 w_k1_DIS_s016
                                 + 0.838954 w_k1_FDX_s016
                                 + 0.658606 w_k1_IBM_s016
                                 + 0.650522 w_k1_INTC_s016
                                 + 0.680195 w_k1_KO_s016
                                 + 0.916519 w_k1_MAT_s016
                                 + 0.809878 w_k1_MSFT_s016
                                 + 0.803766 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.887218 w_k1_ADM_s016
                                 + 1.19056 w_k1_AXP_s016
                                 + 1.22683 w_k1_DIS_s016
                                 + 1.08394 w_k1_FDX_s016
                                 + 1.30326 w_k1_IBM_s016
                                 + 0.812463 w_k1_INTC_s016 + 1.104 w_k1_KO_s016
                                 + 1.30071 w_k1_MAT_s016
                                 + 1.04416 w_k1_MSFT_s016
                                 + 0.996602 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.678779 w_k1_ADM_s016
                                 + 0.60648 w_k1_AXP_s016
                                 + 0.672844 w_k1_DIS_s016
                                 + 0.771075 w_k1_FDX_s016
                                 + 0.846593 w_k1_IBM_s016
                                 + 0.545508 w_k1_INTC_s016
                                 + 0.769508 w_k1_KO_s016
                                 + 0.679613 w_k1_MAT_s016
                                 + 0.706591 w_k1_MSFT_s016
                                 + 0.864978 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.918252 w_k1_ADM_s016
                                 + 1.06478 w_k1_AXP_s016
                                 + 0.842504 w_k1_DIS_s016
                                 + 0.92561 w_k1_FDX_s016
                                 + 1.35054 w_k1_IBM_s016
                                 + 0.730763 w_k1_INTC_s016
                                 + 0.984454 w_k1_KO_s016 + 1.4894 w_k1_MAT_s016
                                 + 0.776163 w_k1_MSFT_s016
                                 + 0.92604 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.934442 w_k1_ADM_s016
                                 + 0.885673 w_k1_AXP_s016
                                 + 0.919744 w_k1_DIS_s016
                                 + 0.983652 w_k1_FDX_s016
                                 + 0.887064 w_k1_IBM_s016
                                 + 0.940038 w_k1_INTC_s016
                                 + 0.953271 w_k1_KO_s016
                                 + 1.00937 w_k1_MAT_s016
                                 + 1.04003 w_k1_MSFT_s016 + 0.9448 w_k1_PG_s016
                                 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.24555 w_k1_ADM_s016
                                 + 1.21605 w_k1_AXP_s016
                                 + 1.24159 w_k1_DIS_s016
                                 + 1.33207 w_k1_FDX_s016
                                 + 0.848978 w_k1_IBM_s016
                                 + 1.05814 w_k1_INTC_s016
                                 + 1.00822 w_k1_KO_s016
                                 + 0.873377 w_k1_MAT_s016
                                 + 1.17994 w_k1_MSFT_s016
                                 + 0.928297 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.37221 w_k1_ADM_s016
                                 + 1.39149 w_k1_AXP_s016
                                 + 1.99436 w_k1_DIS_s016
                                 + 1.42426 w_k1_FDX_s016
                                 + 1.48346 w_k1_IBM_s016
                                 + 1.13694 w_k1_INTC_s016
                                 + 1.47268 w_k1_KO_s016 + 1.69598 w_k1_MAT_s016
                                 + 1.37563 w_k1_MSFT_s016
                                 + 1.19146 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.08723 w_k1_ADM_s017
                                 + 1.2443 w_k1_AXP_s017 + 1.50198 w_k1_DIS_s017
                                 + 1.23419 w_k1_FDX_s017
                                 + 1.07259 w_k1_IBM_s017
                                 + 0.92889 w_k1_INTC_s017
                                 + 1.00111 w_k1_KO_s017 + 1.09941 w_k1_MAT_s017
                                 + 1.21477 w_k1_MSFT_s017
                                 + 1.06248 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.929354 w_k1_ADM_s017
                                 + 0.803528 w_k1_AXP_s017
                                 + 0.909696 w_k1_DIS_s017
                                 + 0.852244 w_k1_FDX_s017
                                 + 1.24216 w_k1_IBM_s017
                                 + 0.6674 w_k1_INTC_s017 + 1.02812 w_k1_KO_s017
                                 + 1.05607 w_k1_MAT_s017
                                 + 0.861132 w_k1_MSFT_s017
                                 + 0.969196 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.30478 w_k1_ADM_s017
                                 + 1.42754 w_k1_AXP_s017
                                 + 1.43093 w_k1_DIS_s017
                                 + 1.12376 w_k1_FDX_s017
                                 + 1.66962 w_k1_IBM_s017
                                 + 0.866873 w_k1_INTC_s017
                                 + 1.38692 w_k1_KO_s017 + 1.77 w_k1_MAT_s017
                                 + 1.15355 w_k1_MSFT_s017
                                 + 1.21411 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.926473 w_k1_ADM_s017
                                 + 0.721957 w_k1_AXP_s017
                                 + 0.733639 w_k1_DIS_s017
                                 + 0.734317 w_k1_FDX_s017
                                 + 1.14183 w_k1_IBM_s017
                                 + 0.733088 w_k1_INTC_s017
                                 + 0.923757 w_k1_KO_s017
                                 + 0.995884 w_k1_MAT_s017
                                 + 0.80196 w_k1_MSFT_s017
                                 + 1.01808 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.27802 w_k1_ADM_s017
                                 + 1.45589 w_k1_AXP_s017
                                 + 1.32595 w_k1_DIS_s017
                                 + 1.36292 w_k1_FDX_s017
                                 + 1.24004 w_k1_IBM_s017
                                 + 1.12195 w_k1_INTC_s017
                                 + 1.13959 w_k1_KO_s017 + 1.16788 w_k1_MAT_s017
                                 + 1.24177 w_k1_MSFT_s017 + 1.1104 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.01111 w_k1_ADM_s017
                                 + 0.946784 w_k1_AXP_s017
                                 + 0.937169 w_k1_DIS_s017
                                 + 0.989656 w_k1_FDX_s017
                                 + 1.08535 w_k1_IBM_s017
                                 + 0.839658 w_k1_INTC_s017
                                 + 0.923126 w_k1_KO_s017
                                 + 1.04345 w_k1_MAT_s017
                                 + 0.840014 w_k1_MSFT_s017
                                 + 0.971684 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.30075 w_k1_ADM_s017
                                 + 1.16031 w_k1_AXP_s017
                                 + 0.95194 w_k1_DIS_s017
                                 + 1.05304 w_k1_FDX_s017
                                 + 0.757218 w_k1_IBM_s017
                                 + 0.771104 w_k1_INTC_s017
                                 + 0.771664 w_k1_KO_s017
                                 + 0.80443 w_k1_MAT_s017
                                 + 0.939689 w_k1_MSFT_s017
                                 + 0.985204 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.848604 w_k1_ADM_s017
                                 + 0.723539 w_k1_AXP_s017
                                 + 0.620107 w_k1_DIS_s017
                                 + 0.806291 w_k1_FDX_s017
                                 + 0.735609 w_k1_IBM_s017
                                 + 0.6908 w_k1_INTC_s017
                                 + 0.567172 w_k1_KO_s017
                                 + 0.559539 w_k1_MAT_s017
                                 + 0.744127 w_k1_MSFT_s017
                                 + 0.804243 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.758728 w_k1_ADM_s100
                                 + 0.646808 w_k1_AXP_s100
                                 + 0.683282 w_k1_DIS_s100
                                 + 0.740043 w_k1_FDX_s100
                                 + 0.873261 w_k1_IBM_s100
                                 + 0.915628 w_k1_INTC_s100
                                 + 0.836745 w_k1_KO_s100
                                 + 0.788996 w_k1_MAT_s100
                                 + 0.796908 w_k1_MSFT_s100
                                 + 0.867156 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.864377 w_k1_ADM_s100
                                 + 1.26943 w_k1_AXP_s100
                                 + 1.04735 w_k1_DIS_s100
                                 + 1.02891 w_k1_FDX_s100
                                 + 1.33009 w_k1_IBM_s100
                                 + 0.888931 w_k1_INTC_s100
                                 + 1.16984 w_k1_KO_s100 + 1.51811 w_k1_MAT_s100
                                 + 1.03088 w_k1_MSFT_s100
                                 + 1.06084 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.697827 w_k1_ADM_s100
                                 + 0.53528 w_k1_AXP_s100
                                 + 0.625774 w_k1_DIS_s100
                                 + 0.758692 w_k1_FDX_s100
                                 + 0.597924 w_k1_IBM_s100
                                 + 0.872743 w_k1_INTC_s100
                                 + 0.73551 w_k1_KO_s100
                                 + 0.423923 w_k1_MAT_s100
                                 + 0.878521 w_k1_MSFT_s100
                                 + 0.816876 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.869401 w_k1_ADM_s100
                                 + 0.974574 w_k1_AXP_s100
                                 + 1.05915 w_k1_DIS_s100
                                 + 0.803244 w_k1_FDX_s100
                                 + 1.66901 w_k1_IBM_s100
                                 + 0.937974 w_k1_INTC_s100
                                 + 1.18987 w_k1_KO_s100 + 1.63323 w_k1_MAT_s100
                                 + 0.966567 w_k1_MSFT_s100
                                 + 1.05382 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.755088 w_k1_ADM_s100
                                 + 0.877732 w_k1_AXP_s100
                                 + 0.900232 w_k1_DIS_s100
                                 + 0.864274 w_k1_FDX_s100
                                 + 1.02573 w_k1_IBM_s100
                                 + 1.0466 w_k1_INTC_s100
                                 + 0.980035 w_k1_KO_s100
                                 + 0.87251 w_k1_MAT_s100
                                 + 0.934593 w_k1_MSFT_s100
                                 + 0.838691 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.28131 w_k1_ADM_s100
                                 + 1.395 w_k1_AXP_s100 + 1.72124 w_k1_DIS_s100
                                 + 1.33417 w_k1_FDX_s100
                                 + 1.37547 w_k1_IBM_s100
                                 + 1.32921 w_k1_INTC_s100
                                 + 1.37521 w_k1_KO_s100 + 1.54006 w_k1_MAT_s100
                                 + 1.18427 w_k1_MSFT_s100
                                 + 1.05624 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.11592 w_k1_ADM_s100
                                 + 1.08631 w_k1_AXP_s100
                                 + 1.64622 w_k1_DIS_s100 + 1.2446 w_k1_FDX_s100
                                 + 1.00304 w_k1_IBM_s100
                                 + 1.11929 w_k1_INTC_s100 + 1.1738 w_k1_KO_s100
                                 + 0.885092 w_k1_MAT_s100
                                 + 1.05744 w_k1_MSFT_s100
                                 + 1.03175 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.12297 w_k1_ADM_s100
                                 + 1.02938 w_k1_AXP_s100
                                 + 1.19477 w_k1_DIS_s100
                                 + 1.25678 w_k1_FDX_s100
                                 + 0.697383 w_k1_IBM_s100
                                 + 1.06938 w_k1_INTC_s100
                                 + 1.00402 w_k1_KO_s100
                                 + 0.805977 w_k1_MAT_s100
                                 + 0.965935 w_k1_MSFT_s100
                                 + 0.950048 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.782796 w_k1_ADM_s101
                                 + 1.05804 w_k1_AXP_s101
                                 + 0.79913 w_k1_DIS_s101
                                 + 0.909726 w_k1_FDX_s101
                                 + 1.25424 w_k1_IBM_s101
                                 + 1.03679 w_k1_INTC_s101
                                 + 0.860453 w_k1_KO_s101 + 1.3529 w_k1_MAT_s101
                                 + 0.874901 w_k1_MSFT_s101
                                 + 1.07703 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.959262 w_k1_ADM_s101
                                 + 1.61437 w_k1_AXP_s101
                                 + 1.49187 w_k1_DIS_s101
                                 + 1.29844 w_k1_FDX_s101
                                 + 1.53586 w_k1_IBM_s101
                                 + 1.21751 w_k1_INTC_s101
                                 + 1.47526 w_k1_KO_s101 + 1.79296 w_k1_MAT_s101
                                 + 1.20093 w_k1_MSFT_s101
                                 + 1.37115 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.16046 w_k1_ADM_s101
                                 + 1.10343 w_k1_AXP_s101 + 1.2971 w_k1_DIS_s101
                                 + 1.34405 w_k1_FDX_s101
                                 + 1.07758 w_k1_IBM_s101
                                 + 1.1765 w_k1_INTC_s101 + 1.06074 w_k1_KO_s101
                                 + 1.10842 w_k1_MAT_s101
                                 + 1.14723 w_k1_MSFT_s101
                                 + 1.10628 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.0991 w_k1_ADM_s101
                                 + 0.923916 w_k1_AXP_s101
                                 + 1.07466 w_k1_DIS_s101
                                 + 0.991858 w_k1_FDX_s101
                                 + 1.33101 w_k1_IBM_s101
                                 + 1.05266 w_k1_INTC_s101
                                 + 1.06584 w_k1_KO_s101
                                 + 0.975264 w_k1_MAT_s101
                                 + 0.984825 w_k1_MSFT_s101
                                 + 1.16639 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.917422 w_k1_ADM_s101
                                 + 1.22316 w_k1_AXP_s101
                                 + 1.18593 w_k1_DIS_s101
                                 + 0.962291 w_k1_FDX_s101
                                 + 1.66047 w_k1_IBM_s101
                                 + 1.10843 w_k1_INTC_s101
                                 + 1.36763 w_k1_KO_s101 + 1.65236 w_k1_MAT_s101
                                 + 1.08359 w_k1_MSFT_s101
                                 + 1.18776 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.975771 w_k1_ADM_s101
                                 + 0.789617 w_k1_AXP_s101
                                 + 0.878896 w_k1_DIS_s101
                                 + 0.851027 w_k1_FDX_s101
                                 + 0.981166 w_k1_IBM_s101
                                 + 0.808074 w_k1_INTC_s101
                                 + 0.966473 w_k1_KO_s101
                                 + 0.832017 w_k1_MAT_s101
                                 + 0.853317 w_k1_MSFT_s101
                                 + 0.958265 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.871204 w_k1_ADM_s101
                                 + 0.795489 w_k1_AXP_s101
                                 + 0.717389 w_k1_DIS_s101
                                 + 0.845598 w_k1_FDX_s101
                                 + 0.74221 w_k1_IBM_s101
                                 + 0.771909 w_k1_INTC_s101
                                 + 0.695734 w_k1_KO_s101
                                 + 0.799884 w_k1_MAT_s101
                                 + 0.813749 w_k1_MSFT_s101
                                 + 0.925509 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.962128 w_k1_ADM_s101
                                 + 1.02348 w_k1_AXP_s101
                                 + 0.953076 w_k1_DIS_s101
                                 + 1.03914 w_k1_FDX_s101
                                 + 0.851459 w_k1_IBM_s101
                                 + 0.855471 w_k1_INTC_s101
                                 + 0.831869 w_k1_KO_s101
                                 + 0.94751 w_k1_MAT_s101
                                 + 0.838885 w_k1_MSFT_s101
                                 + 1.0322 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.937704 w_k1_ADM_s102
                                 + 0.774771 w_k1_AXP_s102
                                 + 0.819875 w_k1_DIS_s102
                                 + 0.817266 w_k1_FDX_s102
                                 + 1.07958 w_k1_IBM_s102
                                 + 0.948987 w_k1_INTC_s102
                                 + 0.910168 w_k1_KO_s102
                                 + 1.09523 w_k1_MAT_s102
                                 + 0.920337 w_k1_MSFT_s102
                                 + 0.860911 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.972888 w_k1_ADM_s102
                                 + 0.848337 w_k1_AXP_s102
                                 + 0.827346 w_k1_DIS_s102
                                 + 0.943144 w_k1_FDX_s102
                                 + 0.723792 w_k1_IBM_s102
                                 + 1.06455 w_k1_INTC_s102
                                 + 0.792756 w_k1_KO_s102
                                 + 0.711715 w_k1_MAT_s102
                                 + 0.989658 w_k1_MSFT_s102
                                 + 0.892124 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.00094 w_k1_ADM_s102
                                 + 0.994101 w_k1_AXP_s102
                                 + 1.19474 w_k1_DIS_s102
                                 + 1.08052 w_k1_FDX_s102
                                 + 1.07802 w_k1_IBM_s102
                                 + 0.906194 w_k1_INTC_s102
                                 + 0.942322 w_k1_KO_s102
                                 + 1.25333 w_k1_MAT_s102
                                 + 0.9972 w_k1_MSFT_s102 + 1.00454 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.30008 w_k1_ADM_s102
                                 + 1.10489 w_k1_AXP_s102
                                 + 0.87345 w_k1_DIS_s102
                                 + 1.05992 w_k1_FDX_s102
                                 + 0.843515 w_k1_IBM_s102
                                 + 1.2009 w_k1_INTC_s102
                                 + 0.896968 w_k1_KO_s102
                                 + 0.987789 w_k1_MAT_s102
                                 + 1.05122 w_k1_MSFT_s102
                                 + 0.931917 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.38424 w_k1_ADM_s102
                                 + 1.07475 w_k1_AXP_s102
                                 + 1.35502 w_k1_DIS_s102
                                 + 1.34389 w_k1_FDX_s102
                                 + 0.953027 w_k1_IBM_s102
                                 + 1.18474 w_k1_INTC_s102
                                 + 1.15999 w_k1_KO_s102 + 1.03398 w_k1_MAT_s102
                                 + 1.21091 w_k1_MSFT_s102
                                 + 1.08189 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.859939 w_k1_ADM_s102
                                 + 0.780201 w_k1_AXP_s102
                                 + 0.720937 w_k1_DIS_s102
                                 + 0.883471 w_k1_FDX_s102
                                 + 0.865175 w_k1_IBM_s102
                                 + 0.906837 w_k1_INTC_s102
                                 + 0.712986 w_k1_KO_s102
                                 + 0.922808 w_k1_MAT_s102
                                 + 0.805516 w_k1_MSFT_s102
                                 + 0.771019 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.966004 w_k1_ADM_s102
                                 + 1.00813 w_k1_AXP_s102
                                 + 1.19959 w_k1_DIS_s102
                                 + 1.11066 w_k1_FDX_s102
                                 + 1.43233 w_k1_IBM_s102
                                 + 1.21994 w_k1_INTC_s102
                                 + 1.19757 w_k1_KO_s102 + 1.63721 w_k1_MAT_s102
                                 + 1.13986 w_k1_MSFT_s102
                                 + 1.15314 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.693882 w_k1_ADM_s102
                                 + 0.693456 w_k1_AXP_s102
                                 + 0.643047 w_k1_DIS_s102
                                 + 0.82266 w_k1_FDX_s102
                                 + 0.67579 w_k1_IBM_s102
                                 + 1.00878 w_k1_INTC_s102
                                 + 0.680663 w_k1_KO_s102
                                 + 0.772954 w_k1_MAT_s102
                                 + 0.932355 w_k1_MSFT_s102
                                 + 0.822623 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.944129 w_k1_ADM_s103
                                 + 1.21879 w_k1_AXP_s103
                                 + 1.34996 w_k1_DIS_s103
                                 + 1.32384 w_k1_FDX_s103
                                 + 0.866919 w_k1_IBM_s103
                                 + 1.18568 w_k1_INTC_s103
                                 + 1.00057 w_k1_KO_s103 + 1.09005 w_k1_MAT_s103
                                 + 1.2641 w_k1_MSFT_s103 + 1.169 w_k1_PG_s103
                                 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.929836 w_k1_ADM_s103
                                 + 0.654515 w_k1_AXP_s103
                                 + 0.647142 w_k1_DIS_s103
                                 + 0.772991 w_k1_FDX_s103
                                 + 0.708445 w_k1_IBM_s103
                                 + 0.759746 w_k1_INTC_s103
                                 + 0.75804 w_k1_KO_s103
                                 + 0.635276 w_k1_MAT_s103
                                 + 0.878168 w_k1_MSFT_s103
                                 + 0.846005 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.22568 w_k1_ADM_s103
                                 + 1.43928 w_k1_AXP_s103
                                 + 1.48255 w_k1_DIS_s103
                                 + 1.24828 w_k1_FDX_s103
                                 + 1.22725 w_k1_IBM_s103
                                 + 1.14304 w_k1_INTC_s103
                                 + 1.12788 w_k1_KO_s103 + 1.24628 w_k1_MAT_s103
                                 + 1.1058 w_k1_MSFT_s103 + 1.26471 w_k1_PG_s103
                                 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.14409 w_k1_ADM_s103
                                 + 1.00794 w_k1_AXP_s103
                                 + 0.949303 w_k1_DIS_s103
                                 + 1.0566 w_k1_FDX_s103
                                 + 0.706112 w_k1_IBM_s103
                                 + 1.01763 w_k1_INTC_s103
                                 + 0.940349 w_k1_KO_s103
                                 + 0.876433 w_k1_MAT_s103
                                 + 1.01364 w_k1_MSFT_s103
                                 + 0.977396 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.927097 w_k1_ADM_s103
                                 + 1.03099 w_k1_AXP_s103
                                 + 1.04118 w_k1_DIS_s103
                                 + 0.969095 w_k1_FDX_s103
                                 + 1.15633 w_k1_IBM_s103
                                 + 1.00637 w_k1_INTC_s103
                                 + 1.04789 w_k1_KO_s103
                                 + 0.944502 w_k1_MAT_s103
                                 + 0.992846 w_k1_MSFT_s103
                                 + 0.990572 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.949407 w_k1_ADM_s103
                                 + 0.902506 w_k1_AXP_s103
                                 + 0.758283 w_k1_DIS_s103
                                 + 0.881457 w_k1_FDX_s103
                                 + 1.01024 w_k1_IBM_s103
                                 + 1.0128 w_k1_INTC_s103
                                 + 0.981872 w_k1_KO_s103
                                 + 0.865204 w_k1_MAT_s103
                                 + 0.846054 w_k1_MSFT_s103
                                 + 0.93304 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.05105 w_k1_ADM_s103
                                 + 1.08594 w_k1_AXP_s103
                                 + 0.936873 w_k1_DIS_s103
                                 + 1.03635 w_k1_FDX_s103
                                 + 0.968544 w_k1_IBM_s103
                                 + 1.06808 w_k1_INTC_s103
                                 + 1.04759 w_k1_KO_s103
                                 + 0.980507 w_k1_MAT_s103
                                 + 1.01463 w_k1_MSFT_s103
                                 + 1.00125 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.774296 w_k1_ADM_s103
                                 + 0.756205 w_k1_AXP_s103
                                 + 0.815777 w_k1_DIS_s103
                                 + 0.747326 w_k1_FDX_s103
                                 + 1.2467 w_k1_IBM_s103
                                 + 0.727944 w_k1_INTC_s103
                                 + 1.01608 w_k1_KO_s103 + 1.21113 w_k1_MAT_s103
                                 + 0.851051 w_k1_MSFT_s103
                                 + 0.916818 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.13747 w_k1_ADM_s104
                                 + 1.43681 w_k1_AXP_s104
                                 + 1.60259 w_k1_DIS_s104
                                 + 1.32971 w_k1_FDX_s104
                                 + 1.69614 w_k1_IBM_s104
                                 + 1.27024 w_k1_INTC_s104
                                 + 1.32869 w_k1_KO_s104 + 1.94681 w_k1_MAT_s104
                                 + 1.39057 w_k1_MSFT_s104
                                 + 1.21507 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.609725 w_k1_ADM_s104
                                 + 0.834409 w_k1_AXP_s104
                                 + 0.655594 w_k1_DIS_s104
                                 + 0.758231 w_k1_FDX_s104
                                 + 1.18134 w_k1_IBM_s104
                                 + 1.02857 w_k1_INTC_s104
                                 + 0.935357 w_k1_KO_s104 + 1.4301 w_k1_MAT_s104
                                 + 0.93536 w_k1_MSFT_s104
                                 + 0.839597 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.24823 w_k1_ADM_s104
                                 + 1.56919 w_k1_AXP_s104
                                 + 0.986327 w_k1_DIS_s104
                                 + 1.24611 w_k1_FDX_s104
                                 + 1.09442 w_k1_IBM_s104
                                 + 1.39006 w_k1_INTC_s104
                                 + 1.01106 w_k1_KO_s104 + 1.166 w_k1_MAT_s104
                                 + 1.19265 w_k1_MSFT_s104
                                 + 1.21788 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.804722 w_k1_ADM_s104
                                 + 0.673142 w_k1_AXP_s104
                                 + 0.633644 w_k1_DIS_s104
                                 + 0.794171 w_k1_FDX_s104
                                 + 0.73694 w_k1_IBM_s104
                                 + 0.932059 w_k1_INTC_s104
                                 + 0.658077 w_k1_KO_s104
                                 + 0.81755 w_k1_MAT_s104
                                 + 0.752968 w_k1_MSFT_s104
                                 + 0.833991 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.16189 w_k1_ADM_s104
                                 + 0.929278 w_k1_AXP_s104
                                 + 1.25431 w_k1_DIS_s104
                                 + 1.11825 w_k1_FDX_s104
                                 + 0.837806 w_k1_IBM_s104
                                 + 1.01183 w_k1_INTC_s104
                                 + 0.971503 w_k1_KO_s104
                                 + 0.900529 w_k1_MAT_s104
                                 + 1.16287 w_k1_MSFT_s104
                                 + 1.00079 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.34454 w_k1_ADM_s104
                                 + 1.41882 w_k1_AXP_s104
                                 + 1.80766 w_k1_DIS_s104
                                 + 1.59982 w_k1_FDX_s104 + 1.0115 w_k1_IBM_s104
                                 + 1.35164 w_k1_INTC_s104
                                 + 1.35489 w_k1_KO_s104 + 1.49424 w_k1_MAT_s104
                                 + 1.51423 w_k1_MSFT_s104
                                 + 1.17914 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.05292 w_k1_ADM_s104
                                 + 0.851197 w_k1_AXP_s104
                                 + 0.769316 w_k1_DIS_s104
                                 + 0.918491 w_k1_FDX_s104
                                 + 0.948268 w_k1_IBM_s104
                                 + 0.835735 w_k1_INTC_s104
                                 + 0.908068 w_k1_KO_s104
                                 + 0.915587 w_k1_MAT_s104
                                 + 0.968805 w_k1_MSFT_s104
                                 + 0.972142 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.08039 w_k1_ADM_s104
                                 + 0.9419 w_k1_AXP_s104 + 1.15284 w_k1_DIS_s104
                                 + 0.990632 w_k1_FDX_s104
                                 + 1.30303 w_k1_IBM_s104
                                 + 1.19592 w_k1_INTC_s104
                                 + 1.23868 w_k1_KO_s104 + 1.03524 w_k1_MAT_s104
                                 + 1.19156 w_k1_MSFT_s104
                                 + 0.940852 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.0497 w_k1_ADM_s105
                                 + 1.24711 w_k1_AXP_s105 + 1.7987 w_k1_DIS_s105
                                 + 1.2248 w_k1_FDX_s105 + 1.30949 w_k1_IBM_s105
                                 + 1.05771 w_k1_INTC_s105
                                 + 1.39596 w_k1_KO_s105 + 1.03891 w_k1_MAT_s105
                                 + 1.16299 w_k1_MSFT_s105
                                 + 0.978078 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.653961 w_k1_ADM_s105
                                 + 0.878459 w_k1_AXP_s105
                                 + 0.756664 w_k1_DIS_s105
                                 + 0.744224 w_k1_FDX_s105
                                 + 1.18987 w_k1_IBM_s105
                                 + 1.07714 w_k1_INTC_s105 + 1.2188 w_k1_KO_s105
                                 + 1.02555 w_k1_MAT_s105
                                 + 0.836987 w_k1_MSFT_s105
                                 + 0.898316 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.791981 w_k1_ADM_s105
                                 + 1.40292 w_k1_AXP_s105
                                 + 1.61864 w_k1_DIS_s105
                                 + 1.24624 w_k1_FDX_s105
                                 + 1.70494 w_k1_IBM_s105
                                 + 1.4359 w_k1_INTC_s105 + 1.51778 w_k1_KO_s105
                                 + 1.77934 w_k1_MAT_s105
                                 + 1.23626 w_k1_MSFT_s105
                                 + 1.00302 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.18797 w_k1_ADM_s105
                                 + 0.98693 w_k1_AXP_s105
                                 + 1.05839 w_k1_DIS_s105
                                 + 1.07219 w_k1_FDX_s105
                                 + 1.33736 w_k1_IBM_s105
                                 + 1.06779 w_k1_INTC_s105
                                 + 1.23055 w_k1_KO_s105 + 1.27243 w_k1_MAT_s105
                                 + 1.02686 w_k1_MSFT_s105
                                 + 1.07936 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.820553 w_k1_ADM_s105
                                 + 0.786314 w_k1_AXP_s105
                                 + 0.727632 w_k1_DIS_s105
                                 + 0.925017 w_k1_FDX_s105
                                 + 0.667037 w_k1_IBM_s105
                                 + 0.779732 w_k1_INTC_s105
                                 + 0.817037 w_k1_KO_s105
                                 + 0.617311 w_k1_MAT_s105
                                 + 0.826075 w_k1_MSFT_s105
                                 + 0.878976 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.06004 w_k1_ADM_s105
                                 + 1.04921 w_k1_AXP_s105
                                 + 1.09682 w_k1_DIS_s105
                                 + 0.997612 w_k1_FDX_s105
                                 + 1.05124 w_k1_IBM_s105
                                 + 0.96455 w_k1_INTC_s105
                                 + 1.24566 w_k1_KO_s105 + 1.11526 w_k1_MAT_s105
                                 + 0.931388 w_k1_MSFT_s105
                                 + 1.04208 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.28196 w_k1_ADM_s105
                                 + 1.0235 w_k1_AXP_s105
                                 + 0.999074 w_k1_DIS_s105
                                 + 1.26349 w_k1_FDX_s105
                                 + 0.85005 w_k1_IBM_s105
                                 + 1.13163 w_k1_INTC_s105
                                 + 1.02208 w_k1_KO_s105
                                 + 0.833722 w_k1_MAT_s105
                                 + 1.15943 w_k1_MSFT_s105
                                 + 1.04646 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.718454 w_k1_ADM_s105
                                 + 0.669847 w_k1_AXP_s105
                                 + 0.669649 w_k1_DIS_s105
                                 + 0.752706 w_k1_FDX_s105
                                 + 0.877717 w_k1_IBM_s105
                                 + 1.13811 w_k1_INTC_s105
                                 + 1.03098 w_k1_KO_s105
                                 + 0.734561 w_k1_MAT_s105
                                 + 0.890056 w_k1_MSFT_s105
                                 + 0.846956 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.947779 w_k1_ADM_s106
                                 + 0.996203 w_k1_AXP_s106
                                 + 0.955337 w_k1_DIS_s106
                                 + 0.991981 w_k1_FDX_s106
                                 + 0.925882 w_k1_IBM_s106
                                 + 0.889448 w_k1_INTC_s106
                                 + 0.964488 w_k1_KO_s106
                                 + 1.04301 w_k1_MAT_s106
                                 + 1.01345 w_k1_MSFT_s106
                                 + 1.02686 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.7218 w_k1_ADM_s106
                                 + 1.35481 w_k1_AXP_s106
                                 + 1.68751 w_k1_DIS_s106
                                 + 1.44369 w_k1_FDX_s106
                                 + 1.12385 w_k1_IBM_s106
                                 + 0.880187 w_k1_INTC_s106
                                 + 1.26202 w_k1_KO_s106 + 1.28876 w_k1_MAT_s106
                                 + 1.16036 w_k1_MSFT_s106
                                 + 1.24543 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.1036 w_k1_ADM_s106
                                 + 1.50795 w_k1_AXP_s106
                                 + 1.40611 w_k1_DIS_s106
                                 + 1.28066 w_k1_FDX_s106
                                 + 1.52514 w_k1_IBM_s106
                                 + 1.0211 w_k1_INTC_s106 + 1.26195 w_k1_KO_s106
                                 + 1.74426 w_k1_MAT_s106
                                 + 1.08655 w_k1_MSFT_s106
                                 + 1.26165 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.36289 w_k1_ADM_s106
                                 + 1.20044 w_k1_AXP_s106
                                 + 1.27664 w_k1_DIS_s106
                                 + 1.30167 w_k1_FDX_s106
                                 + 0.917746 w_k1_IBM_s106
                                 + 0.940318 w_k1_INTC_s106
                                 + 1.03218 w_k1_KO_s106
                                 + 0.905038 w_k1_MAT_s106
                                 + 1.04896 w_k1_MSFT_s106
                                 + 1.15465 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.21749 w_k1_ADM_s106
                                 + 0.853443 w_k1_AXP_s106
                                 + 0.979956 w_k1_DIS_s106
                                 + 1.02451 w_k1_FDX_s106
                                 + 0.754435 w_k1_IBM_s106
                                 + 0.841473 w_k1_INTC_s106
                                 + 0.825333 w_k1_KO_s106
                                 + 0.660811 w_k1_MAT_s106
                                 + 0.99378 w_k1_MSFT_s106
                                 + 0.891889 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.785109 w_k1_ADM_s106
                                 + 0.837376 w_k1_AXP_s106
                                 + 1.0429 w_k1_DIS_s106
                                 + 0.912387 w_k1_FDX_s106
                                 + 1.01213 w_k1_IBM_s106
                                 + 0.746425 w_k1_INTC_s106
                                 + 0.964729 w_k1_KO_s106 + 1.1556 w_k1_MAT_s106
                                 + 0.96395 w_k1_MSFT_s106
                                 + 0.846209 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.06782 w_k1_ADM_s106
                                 + 0.825899 w_k1_AXP_s106
                                 + 0.773965 w_k1_DIS_s106
                                 + 0.849138 w_k1_FDX_s106
                                 + 1.39986 w_k1_IBM_s106
                                 + 0.770177 w_k1_INTC_s106
                                 + 1.05433 w_k1_KO_s106 + 1.38698 w_k1_MAT_s106
                                 + 0.851871 w_k1_MSFT_s106
                                 + 1.00758 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.16528 w_k1_ADM_s106
                                 + 0.904424 w_k1_AXP_s106
                                 + 0.834987 w_k1_DIS_s106
                                 + 0.959661 w_k1_FDX_s106
                                 + 1.11573 w_k1_IBM_s106
                                 + 0.668886 w_k1_INTC_s106
                                 + 0.974033 w_k1_KO_s106
                                 + 1.32766 w_k1_MAT_s106
                                 + 0.875135 w_k1_MSFT_s106
                                 + 0.912598 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.938586 w_k1_ADM_s107
                                 + 1.05858 w_k1_AXP_s107
                                 + 1.18589 w_k1_DIS_s107
                                 + 1.16113 w_k1_FDX_s107
                                 + 0.898547 w_k1_IBM_s107
                                 + 1.13354 w_k1_INTC_s107
                                 + 0.966924 w_k1_KO_s107
                                 + 0.930246 w_k1_MAT_s107
                                 + 0.930647 w_k1_MSFT_s107
                                 + 0.952335 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.30671 w_k1_ADM_s107
                                 + 1.26778 w_k1_AXP_s107
                                 + 1.46827 w_k1_DIS_s107
                                 + 1.26224 w_k1_FDX_s107
                                 + 1.29878 w_k1_IBM_s107
                                 + 1.10289 w_k1_INTC_s107
                                 + 1.38524 w_k1_KO_s107 + 1.18532 w_k1_MAT_s107
                                 + 1.00172 w_k1_MSFT_s107
                                 + 1.09245 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.40518 w_k1_ADM_s107
                                 + 1.14756 w_k1_AXP_s107
                                 + 1.22016 w_k1_DIS_s107
                                 + 1.18883 w_k1_FDX_s107
                                 + 0.993612 w_k1_IBM_s107
                                 + 1.08711 w_k1_INTC_s107 + 1.11 w_k1_KO_s107
                                 + 1.32858 w_k1_MAT_s107
                                 + 1.10753 w_k1_MSFT_s107
                                 + 1.05848 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.17527 w_k1_ADM_s107
                                 + 0.977896 w_k1_AXP_s107
                                 + 0.889744 w_k1_DIS_s107
                                 + 0.997631 w_k1_FDX_s107
                                 + 0.667907 w_k1_IBM_s107
                                 + 1.1092 w_k1_INTC_s107
                                 + 0.833289 w_k1_KO_s107
                                 + 0.523957 w_k1_MAT_s107
                                 + 0.92157 w_k1_MSFT_s107
                                 + 0.859108 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.2117 w_k1_ADM_s107
                                 + 1.01659 w_k1_AXP_s107
                                 + 0.896566 w_k1_DIS_s107
                                 + 0.846029 w_k1_FDX_s107
                                 + 1.34447 w_k1_IBM_s107
                                 + 0.96877 w_k1_INTC_s107
                                 + 1.15962 w_k1_KO_s107 + 1.20684 w_k1_MAT_s107
                                 + 0.805563 w_k1_MSFT_s107
                                 + 0.989535 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.906583 w_k1_ADM_s107
                                 + 0.713312 w_k1_AXP_s107
                                 + 0.563969 w_k1_DIS_s107
                                 + 0.727071 w_k1_FDX_s107
                                 + 0.924454 w_k1_IBM_s107
                                 + 1.12346 w_k1_INTC_s107
                                 + 0.756858 w_k1_KO_s107
                                 + 0.736159 w_k1_MAT_s107
                                 + 0.729486 w_k1_MSFT_s107
                                 + 0.816805 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.962069 w_k1_ADM_s107
                                 + 0.901034 w_k1_AXP_s107
                                 + 0.886043 w_k1_DIS_s107
                                 + 0.883723 w_k1_FDX_s107
                                 + 0.879978 w_k1_IBM_s107
                                 + 0.989718 w_k1_INTC_s107
                                 + 0.887828 w_k1_KO_s107
                                 + 1.05914 w_k1_MAT_s107
                                 + 0.83804 w_k1_MSFT_s107
                                 + 0.858811 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.55604 w_k1_ADM_s107
                                 + 1.12672 w_k1_AXP_s107
                                 + 1.24569 w_k1_DIS_s107
                                 + 1.05345 w_k1_FDX_s107
                                 + 1.63167 w_k1_IBM_s107
                                 + 1.05246 w_k1_INTC_s107
                                 + 1.31813 w_k1_KO_s107 + 1.34597 w_k1_MAT_s107
                                 + 0.947413 w_k1_MSFT_s107
                                 + 1.07171 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.12039 w_k1_ADM_s110
                                 + 1.32395 w_k1_AXP_s110
                                 + 1.59621 w_k1_DIS_s110
                                 + 1.25239 w_k1_FDX_s110
                                 + 1.35261 w_k1_IBM_s110
                                 + 1.40534 w_k1_INTC_s110
                                 + 1.28713 w_k1_KO_s110 + 1.27581 w_k1_MAT_s110
                                 + 1.18315 w_k1_MSFT_s110
                                 + 1.13004 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.11376 w_k1_ADM_s110
                                 + 1.42691 w_k1_AXP_s110
                                 + 0.890571 w_k1_DIS_s110
                                 + 0.935004 w_k1_FDX_s110
                                 + 1.19088 w_k1_IBM_s110
                                 + 0.883771 w_k1_INTC_s110
                                 + 0.922379 w_k1_KO_s110
                                 + 1.49683 w_k1_MAT_s110
                                 + 0.868907 w_k1_MSFT_s110
                                 + 1.0526 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.27012 w_k1_ADM_s110
                                 + 1.22013 w_k1_AXP_s110
                                 + 1.32776 w_k1_DIS_s110
                                 + 1.30575 w_k1_FDX_s110
                                 + 0.969357 w_k1_IBM_s110
                                 + 1.14781 w_k1_INTC_s110
                                 + 0.940548 w_k1_KO_s110
                                 + 0.95762 w_k1_MAT_s110
                                 + 1.12808 w_k1_MSFT_s110
                                 + 1.14403 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.796603 w_k1_ADM_s110
                                 + 0.912961 w_k1_AXP_s110
                                 + 0.903692 w_k1_DIS_s110
                                 + 0.861783 w_k1_FDX_s110
                                 + 0.975952 w_k1_IBM_s110
                                 + 0.879381 w_k1_INTC_s110
                                 + 0.94094 w_k1_KO_s110 + 1.07745 w_k1_MAT_s110
                                 + 0.824942 w_k1_MSFT_s110
                                 + 0.939051 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.712816 w_k1_ADM_s110
                                 + 0.828826 w_k1_AXP_s110
                                 + 0.62262 w_k1_DIS_s110
                                 + 0.764482 w_k1_FDX_s110
                                 + 0.862456 w_k1_IBM_s110
                                 + 0.797791 w_k1_INTC_s110
                                 + 0.712841 w_k1_KO_s110
                                 + 0.947622 w_k1_MAT_s110
                                 + 0.691856 w_k1_MSFT_s110
                                 + 0.868132 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.064 w_k1_ADM_s110
                                 + 1.03398 w_k1_AXP_s110
                                 + 0.972482 w_k1_DIS_s110
                                 + 0.916026 w_k1_FDX_s110
                                 + 1.21497 w_k1_IBM_s110
                                 + 0.809312 w_k1_INTC_s110
                                 + 0.926007 w_k1_KO_s110
                                 + 1.33205 w_k1_MAT_s110
                                 + 0.813958 w_k1_MSFT_s110
                                 + 1.04083 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.08394 w_k1_ADM_s110
                                 + 0.783979 w_k1_AXP_s110
                                 + 0.893883 w_k1_DIS_s110
                                 + 0.965913 w_k1_FDX_s110
                                 + 0.709599 w_k1_IBM_s110
                                 + 0.937458 w_k1_INTC_s110
                                 + 0.798849 w_k1_KO_s110
                                 + 0.588661 w_k1_MAT_s110
                                 + 0.934898 w_k1_MSFT_s110
                                 + 0.909298 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.997183 w_k1_ADM_s110
                                 + 0.903088 w_k1_AXP_s110
                                 + 1.12442 w_k1_DIS_s110
                                 + 0.961127 w_k1_FDX_s110
                                 + 1.04512 w_k1_IBM_s110
                                 + 1.06583 w_k1_INTC_s110
                                 + 1.10976 w_k1_KO_s110
                                 + 0.899436 w_k1_MAT_s110
                                 + 0.946948 w_k1_MSFT_s110
                                 + 1.06952 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.618223 w_k1_ADM_s111
                                 + 0.675086 w_k1_AXP_s111
                                 + 0.640634 w_k1_DIS_s111
                                 + 0.711939 w_k1_FDX_s111
                                 + 0.799497 w_k1_IBM_s111
                                 + 0.894484 w_k1_INTC_s111
                                 + 0.822657 w_k1_KO_s111
                                 + 0.676149 w_k1_MAT_s111
                                 + 0.834981 w_k1_MSFT_s111
                                 + 0.791616 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.24771 w_k1_ADM_s111
                                 + 1.27827 w_k1_AXP_s111
                                 + 1.74518 w_k1_DIS_s111
                                 + 1.46077 w_k1_FDX_s111
                                 + 0.93469 w_k1_IBM_s111 + 1.377 w_k1_INTC_s111
                                 + 1.18597 w_k1_KO_s111
                                 + 0.711978 w_k1_MAT_s111
                                 + 1.59176 w_k1_MSFT_s111
                                 + 1.19118 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.01972 w_k1_ADM_s111
                                 + 1.03529 w_k1_AXP_s111
                                 + 0.915935 w_k1_DIS_s111
                                 + 0.856613 w_k1_FDX_s111
                                 + 1.46624 w_k1_IBM_s111
                                 + 0.858226 w_k1_INTC_s111
                                 + 1.1919 w_k1_KO_s111 + 1.48532 w_k1_MAT_s111
                                 + 1.04309 w_k1_MSFT_s111
                                 + 1.07184 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.903257 w_k1_ADM_s111
                                 + 1.27414 w_k1_AXP_s111
                                 + 1.35275 w_k1_DIS_s111
                                 + 1.19274 w_k1_FDX_s111
                                 + 1.27112 w_k1_IBM_s111
                                 + 1.33398 w_k1_INTC_s111
                                 + 1.39218 w_k1_KO_s111 + 1.23102 w_k1_MAT_s111
                                 + 1.21165 w_k1_MSFT_s111 + 1.1187 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.0398 w_k1_ADM_s111
                                 + 1.22222 w_k1_AXP_s111
                                 + 0.977467 w_k1_DIS_s111
                                 + 1.15797 w_k1_FDX_s111
                                 + 0.985246 w_k1_IBM_s111
                                 + 1.17512 w_k1_INTC_s111
                                 + 0.966803 w_k1_KO_s111
                                 + 0.874725 w_k1_MAT_s111
                                 + 1.17543 w_k1_MSFT_s111
                                 + 1.09142 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.900874 w_k1_ADM_s111
                                 + 0.70902 w_k1_AXP_s111
                                 + 0.728247 w_k1_DIS_s111
                                 + 0.888238 w_k1_FDX_s111
                                 + 0.724041 w_k1_IBM_s111
                                 + 0.985145 w_k1_INTC_s111
                                 + 0.751865 w_k1_KO_s111
                                 + 0.574574 w_k1_MAT_s111
                                 + 0.890413 w_k1_MSFT_s111
                                 + 0.838879 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.717677 w_k1_ADM_s111
                                 + 0.727361 w_k1_AXP_s111
                                 + 0.878047 w_k1_DIS_s111
                                 + 0.855623 w_k1_FDX_s111
                                 + 1.05658 w_k1_IBM_s111
                                 + 0.98527 w_k1_INTC_s111
                                 + 1.13176 w_k1_KO_s111
                                 + 0.746831 w_k1_MAT_s111
                                 + 0.950907 w_k1_MSFT_s111
                                 + 0.959749 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.807501 w_k1_ADM_s111
                                 + 0.922818 w_k1_AXP_s111
                                 + 0.996753 w_k1_DIS_s111
                                 + 1.01021 w_k1_FDX_s111
                                 + 0.799183 w_k1_IBM_s111
                                 + 1.11388 w_k1_INTC_s111
                                 + 0.939115 w_k1_KO_s111
                                 + 0.614543 w_k1_MAT_s111
                                 + 1.11312 w_k1_MSFT_s111
                                 + 0.904528 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.957449 w_k1_ADM_s112
                                 + 1.2064 w_k1_AXP_s112 + 1.03653 w_k1_DIS_s112
                                 + 1.08021 w_k1_FDX_s112
                                 + 1.07666 w_k1_IBM_s112
                                 + 0.948977 w_k1_INTC_s112
                                 + 1.0648 w_k1_KO_s112 + 1.1097 w_k1_MAT_s112
                                 + 0.946084 w_k1_MSFT_s112
                                 + 1.04801 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.14011 w_k1_ADM_s112
                                 + 1.46474 w_k1_AXP_s112 + 1.5159 w_k1_DIS_s112
                                 + 1.2607 w_k1_FDX_s112 + 1.3657 w_k1_IBM_s112
                                 + 1.06119 w_k1_INTC_s112
                                 + 1.25023 w_k1_KO_s112 + 1.2614 w_k1_MAT_s112
                                 + 1.12615 w_k1_MSFT_s112
                                 + 1.20388 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.703225 w_k1_ADM_s112
                                 + 0.629182 w_k1_AXP_s112
                                 + 0.790173 w_k1_DIS_s112
                                 + 0.731905 w_k1_FDX_s112
                                 + 1.02438 w_k1_IBM_s112
                                 + 0.853599 w_k1_INTC_s112
                                 + 0.890152 w_k1_KO_s112
                                 + 0.692173 w_k1_MAT_s112
                                 + 0.845637 w_k1_MSFT_s112
                                 + 0.849428 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.872808 w_k1_ADM_s112
                                 + 1.1123 w_k1_AXP_s112
                                 + 0.812188 w_k1_DIS_s112
                                 + 0.86393 w_k1_FDX_s112
                                 + 0.931283 w_k1_IBM_s112
                                 + 0.931055 w_k1_INTC_s112
                                 + 0.889306 w_k1_KO_s112
                                 + 0.896302 w_k1_MAT_s112
                                 + 0.775134 w_k1_MSFT_s112
                                 + 0.875884 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.963245 w_k1_ADM_s112
                                 + 0.996494 w_k1_AXP_s112
                                 + 0.967952 w_k1_DIS_s112
                                 + 1.00663 w_k1_FDX_s112 + 0.7548 w_k1_IBM_s112
                                 + 0.991495 w_k1_INTC_s112
                                 + 0.820633 w_k1_KO_s112
                                 + 0.700963 w_k1_MAT_s112
                                 + 0.906235 w_k1_MSFT_s112
                                 + 0.876579 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.10336 w_k1_ADM_s112
                                 + 1.56476 w_k1_AXP_s112
                                 + 1.64595 w_k1_DIS_s112
                                 + 1.32962 w_k1_FDX_s112
                                 + 0.961135 w_k1_IBM_s112
                                 + 1.28728 w_k1_INTC_s112 + 1.2412 w_k1_KO_s112
                                 + 1.03091 w_k1_MAT_s112
                                 + 1.28294 w_k1_MSFT_s112
                                 + 1.13835 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.798925 w_k1_ADM_s112
                                 + 1.03681 w_k1_AXP_s112
                                 + 0.95645 w_k1_DIS_s112
                                 + 0.861281 w_k1_FDX_s112
                                 + 1.4945 w_k1_IBM_s112
                                 + 0.859958 w_k1_INTC_s112
                                 + 1.00325 w_k1_KO_s112 + 1.45376 w_k1_MAT_s112
                                 + 0.81908 w_k1_MSFT_s112
                                 + 1.00734 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.666981 w_k1_ADM_s112
                                 + 0.733722 w_k1_AXP_s112
                                 + 0.664229 w_k1_DIS_s112
                                 + 0.809111 w_k1_FDX_s112
                                 + 0.742773 w_k1_IBM_s112
                                 + 0.795228 w_k1_INTC_s112
                                 + 0.78087 w_k1_KO_s112
                                 + 0.648712 w_k1_MAT_s112
                                 + 0.813593 w_k1_MSFT_s112
                                 + 0.856939 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.940442 w_k1_ADM_s113
                                 + 1.34924 w_k1_AXP_s113
                                 + 0.952269 w_k1_DIS_s113
                                 + 1.02634 w_k1_FDX_s113
                                 + 0.988299 w_k1_IBM_s113
                                 + 1.06669 w_k1_INTC_s113
                                 + 1.03551 w_k1_KO_s113 + 1.16943 w_k1_MAT_s113
                                 + 1.10886 w_k1_MSFT_s113
                                 + 0.976291 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.700357 w_k1_ADM_s113
                                 + 0.821202 w_k1_AXP_s113
                                 + 0.837473 w_k1_DIS_s113
                                 + 0.904152 w_k1_FDX_s113
                                 + 1.02854 w_k1_IBM_s113
                                 + 0.843788 w_k1_INTC_s113
                                 + 0.927662 w_k1_KO_s113
                                 + 1.12417 w_k1_MAT_s113
                                 + 0.932371 w_k1_MSFT_s113
                                 + 0.983331 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.764172 w_k1_ADM_s113
                                 + 0.571465 w_k1_AXP_s113
                                 + 0.57854 w_k1_DIS_s113
                                 + 0.77094 w_k1_FDX_s113
                                 + 0.664324 w_k1_IBM_s113
                                 + 0.851898 w_k1_INTC_s113
                                 + 0.615355 w_k1_KO_s113
                                 + 0.483113 w_k1_MAT_s113
                                 + 0.950182 w_k1_MSFT_s113
                                 + 0.810463 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.705552 w_k1_ADM_s113
                                 + 0.894618 w_k1_AXP_s113
                                 + 0.961048 w_k1_DIS_s113
                                 + 0.925848 w_k1_FDX_s113
                                 + 0.856751 w_k1_IBM_s113
                                 + 1.01355 w_k1_INTC_s113
                                 + 0.801777 w_k1_KO_s113
                                 + 0.785923 w_k1_MAT_s113
                                 + 1.07799 w_k1_MSFT_s113
                                 + 0.913742 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.20702 w_k1_ADM_s113
                                 + 1.53465 w_k1_AXP_s113 + 1.5632 w_k1_DIS_s113
                                 + 1.1477 w_k1_FDX_s113 + 1.63817 w_k1_IBM_s113
                                 + 1.09101 w_k1_INTC_s113
                                 + 1.39645 w_k1_KO_s113 + 1.8572 w_k1_MAT_s113
                                 + 1.24733 w_k1_MSFT_s113
                                 + 1.36996 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.602861 w_k1_ADM_s113
                                 + 0.83451 w_k1_AXP_s113
                                 + 0.648092 w_k1_DIS_s113
                                 + 0.658643 w_k1_FDX_s113
                                 + 1.01508 w_k1_IBM_s113
                                 + 0.721575 w_k1_INTC_s113
                                 + 0.782797 w_k1_KO_s113
                                 + 0.942286 w_k1_MAT_s113
                                 + 0.803332 w_k1_MSFT_s113
                                 + 0.895275 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.03208 w_k1_ADM_s113
                                 + 1.39044 w_k1_AXP_s113
                                 + 1.27214 w_k1_DIS_s113
                                 + 1.15896 w_k1_FDX_s113
                                 + 1.06423 w_k1_IBM_s113
                                 + 1.30227 w_k1_INTC_s113
                                 + 1.12785 w_k1_KO_s113 + 1.22793 w_k1_MAT_s113
                                 + 1.25868 w_k1_MSFT_s113
                                 + 1.17156 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.08533 w_k1_ADM_s113
                                 + 1.43262 w_k1_AXP_s113
                                 + 1.93073 w_k1_DIS_s113
                                 + 1.43233 w_k1_FDX_s113
                                 + 1.19455 w_k1_IBM_s113
                                 + 1.06545 w_k1_INTC_s113
                                 + 1.36731 w_k1_KO_s113 + 1.32352 w_k1_MAT_s113
                                 + 1.37073 w_k1_MSFT_s113
                                 + 1.35091 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.10343 w_k1_ADM_s114
                                 + 1.17758 w_k1_AXP_s114
                                 + 1.14959 w_k1_DIS_s114
                                 + 1.03452 w_k1_FDX_s114
                                 + 1.60775 w_k1_IBM_s114
                                 + 1.14554 w_k1_INTC_s114
                                 + 1.13824 w_k1_KO_s114 + 1.45405 w_k1_MAT_s114
                                 + 1.10525 w_k1_MSFT_s114
                                 + 1.16727 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.01945 w_k1_ADM_s114
                                 + 0.818715 w_k1_AXP_s114
                                 + 0.891401 w_k1_DIS_s114
                                 + 0.920213 w_k1_FDX_s114
                                 + 0.980815 w_k1_IBM_s114
                                 + 1.05256 w_k1_INTC_s114
                                 + 0.946549 w_k1_KO_s114
                                 + 0.896449 w_k1_MAT_s114
                                 + 0.940776 w_k1_MSFT_s114
                                 + 0.910873 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.941047 w_k1_ADM_s114
                                 + 0.778633 w_k1_AXP_s114
                                 + 0.836846 w_k1_DIS_s114
                                 + 0.921553 w_k1_FDX_s114
                                 + 0.763345 w_k1_IBM_s114
                                 + 0.971491 w_k1_INTC_s114
                                 + 0.797164 w_k1_KO_s114
                                 + 0.604943 w_k1_MAT_s114
                                 + 0.910394 w_k1_MSFT_s114
                                 + 0.849846 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.35831 w_k1_ADM_s114
                                 + 1.20609 w_k1_AXP_s114
                                 + 1.46756 w_k1_DIS_s114 + 1.2905 w_k1_FDX_s114
                                 + 1.36955 w_k1_IBM_s114
                                 + 1.2619 w_k1_INTC_s114 + 1.23619 w_k1_KO_s114
                                 + 1.43662 w_k1_MAT_s114
                                 + 1.24109 w_k1_MSFT_s114
                                 + 1.24664 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.8283 w_k1_ADM_s114
                                 + 0.749739 w_k1_AXP_s114
                                 + 0.62232 w_k1_DIS_s114
                                 + 0.729427 w_k1_FDX_s114
                                 + 1.01401 w_k1_IBM_s114
                                 + 0.90146 w_k1_INTC_s114
                                 + 0.782256 w_k1_KO_s114
                                 + 0.67855 w_k1_MAT_s114
                                 + 0.75431 w_k1_MSFT_s114
                                 + 0.802368 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.730364 w_k1_ADM_s114
                                 + 0.452917 w_k1_AXP_s114
                                 + 0.546414 w_k1_DIS_s114
                                 + 0.6013 w_k1_FDX_s114
                                 + 0.719238 w_k1_IBM_s114
                                 + 0.753219 w_k1_INTC_s114
                                 + 0.643647 w_k1_KO_s114
                                 + 0.433769 w_k1_MAT_s114
                                 + 0.744792 w_k1_MSFT_s114
                                 + 0.683629 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.73617 w_k1_ADM_s114
                                 + 0.832474 w_k1_AXP_s114
                                 + 0.929365 w_k1_DIS_s114
                                 + 0.800783 w_k1_FDX_s114
                                 + 1.41121 w_k1_IBM_s114
                                 + 1.00594 w_k1_INTC_s114
                                 + 0.977159 w_k1_KO_s114
                                 + 0.950777 w_k1_MAT_s114
                                 + 0.878322 w_k1_MSFT_s114
                                 + 0.950013 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.12954 w_k1_ADM_s114
                                 + 1.13829 w_k1_AXP_s114
                                 + 1.22791 w_k1_DIS_s114
                                 + 1.19845 w_k1_FDX_s114
                                 + 0.887003 w_k1_IBM_s114
                                 + 1.18418 w_k1_INTC_s114
                                 + 0.858552 w_k1_KO_s114 + 1.0575 w_k1_MAT_s114
                                 + 1.24315 w_k1_MSFT_s114
                                 + 1.07544 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.32375 w_k1_ADM_s115
                                 + 1.36283 w_k1_AXP_s115 + 1.6223 w_k1_DIS_s115
                                 + 1.33447 w_k1_FDX_s115
                                 + 1.00661 w_k1_IBM_s115
                                 + 1.09891 w_k1_INTC_s115
                                 + 1.10663 w_k1_KO_s115
                                 + 0.945845 w_k1_MAT_s115
                                 + 1.07852 w_k1_MSFT_s115 + 1.2062 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.768456 w_k1_ADM_s115
                                 + 0.735336 w_k1_AXP_s115
                                 + 0.674199 w_k1_DIS_s115
                                 + 0.733053 w_k1_FDX_s115
                                 + 0.966231 w_k1_IBM_s115
                                 + 0.925653 w_k1_INTC_s115
                                 + 0.778003 w_k1_KO_s115
                                 + 0.796677 w_k1_MAT_s115
                                 + 0.726187 w_k1_MSFT_s115
                                 + 0.834649 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.852057 w_k1_ADM_s115
                                 + 0.833084 w_k1_AXP_s115
                                 + 0.992114 w_k1_DIS_s115
                                 + 0.989596 w_k1_FDX_s115
                                 + 1.11853 w_k1_IBM_s115
                                 + 1.03813 w_k1_INTC_s115
                                 + 1.00489 w_k1_KO_s115 + 1.00053 w_k1_MAT_s115
                                 + 0.966199 w_k1_MSFT_s115
                                 + 1.06656 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.16408 w_k1_ADM_s115
                                 + 1.32449 w_k1_AXP_s115
                                 + 1.22823 w_k1_DIS_s115
                                 + 1.18274 w_k1_FDX_s115
                                 + 1.28503 w_k1_IBM_s115
                                 + 1.10964 w_k1_INTC_s115
                                 + 1.06273 w_k1_KO_s115 + 1.0326 w_k1_MAT_s115
                                 + 0.921531 w_k1_MSFT_s115
                                 + 1.09984 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.841166 w_k1_ADM_s115
                                 + 0.699127 w_k1_AXP_s115
                                 + 0.914766 w_k1_DIS_s115
                                 + 0.958138 w_k1_FDX_s115
                                 + 0.663632 w_k1_IBM_s115
                                 + 1.16701 w_k1_INTC_s115
                                 + 0.818902 w_k1_KO_s115
                                 + 0.559068 w_k1_MAT_s115
                                 + 0.88703 w_k1_MSFT_s115
                                 + 0.868438 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.971531 w_k1_ADM_s115
                                 + 0.839909 w_k1_AXP_s115
                                 + 0.739048 w_k1_DIS_s115
                                 + 0.909016 w_k1_FDX_s115
                                 + 0.838563 w_k1_IBM_s115
                                 + 0.801486 w_k1_INTC_s115
                                 + 0.825765 w_k1_KO_s115
                                 + 0.957846 w_k1_MAT_s115
                                 + 0.854133 w_k1_MSFT_s115
                                 + 0.933934 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.939071 w_k1_ADM_s115
                                 + 1.1114 w_k1_AXP_s115 + 1.03672 w_k1_DIS_s115
                                 + 0.883172 w_k1_FDX_s115
                                 + 1.55275 w_k1_IBM_s115
                                 + 0.970953 w_k1_INTC_s115
                                 + 1.24636 w_k1_KO_s115 + 1.78201 w_k1_MAT_s115
                                 + 0.945919 w_k1_MSFT_s115
                                 + 1.06542 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.06118 w_k1_ADM_s115
                                 + 1.20608 w_k1_AXP_s115
                                 + 1.01094 w_k1_DIS_s115
                                 + 1.08614 w_k1_FDX_s115
                                 + 0.907644 w_k1_IBM_s115
                                 + 0.997386 w_k1_INTC_s115
                                 + 0.928396 w_k1_KO_s115
                                 + 1.12745 w_k1_MAT_s115
                                 + 0.908953 w_k1_MSFT_s115
                                 + 1.0871 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.775444 w_k1_ADM_s116
                                 + 0.632912 w_k1_AXP_s116
                                 + 0.665791 w_k1_DIS_s116
                                 + 0.767193 w_k1_FDX_s116
                                 + 0.643619 w_k1_IBM_s116
                                 + 0.884504 w_k1_INTC_s116
                                 + 0.718743 w_k1_KO_s116
                                 + 0.657008 w_k1_MAT_s116
                                 + 0.871338 w_k1_MSFT_s116
                                 + 0.764715 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.888103 w_k1_ADM_s116
                                 + 0.865146 w_k1_AXP_s116
                                 + 0.764106 w_k1_DIS_s116
                                 + 0.842497 w_k1_FDX_s116
                                 + 0.943785 w_k1_IBM_s116
                                 + 0.895817 w_k1_INTC_s116
                                 + 0.919868 w_k1_KO_s116
                                 + 1.00916 w_k1_MAT_s116
                                 + 0.944936 w_k1_MSFT_s116
                                 + 0.854326 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.04271 w_k1_ADM_s116
                                 + 0.939607 w_k1_AXP_s116 + 1.144 w_k1_DIS_s116
                                 + 1.14951 w_k1_FDX_s116
                                 + 0.799206 w_k1_IBM_s116
                                 + 1.24847 w_k1_INTC_s116
                                 + 0.878332 w_k1_KO_s116
                                 + 1.11056 w_k1_MAT_s116
                                 + 1.21561 w_k1_MSFT_s116
                                 + 1.02313 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.03908 w_k1_ADM_s116
                                 + 0.952981 w_k1_AXP_s116
                                 + 0.908469 w_k1_DIS_s116
                                 + 0.915672 w_k1_FDX_s116
                                 + 1.27151 w_k1_IBM_s116
                                 + 1.06608 w_k1_INTC_s116
                                 + 1.03309 w_k1_KO_s116 + 1.01565 w_k1_MAT_s116
                                 + 0.953892 w_k1_MSFT_s116
                                 + 0.908972 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.1572 w_k1_ADM_s116
                                 + 1.00626 w_k1_AXP_s116
                                 + 1.44361 w_k1_DIS_s116
                                 + 1.11375 w_k1_FDX_s116
                                 + 1.38345 w_k1_IBM_s116
                                 + 1.05096 w_k1_INTC_s116
                                 + 1.31546 w_k1_KO_s116 + 1.40659 w_k1_MAT_s116
                                 + 1.27404 w_k1_MSFT_s116
                                 + 1.06227 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.08375 w_k1_ADM_s116
                                 + 0.975765 w_k1_AXP_s116
                                 + 0.876436 w_k1_DIS_s116
                                 + 1.08815 w_k1_FDX_s116
                                 + 0.927432 w_k1_IBM_s116
                                 + 0.948093 w_k1_INTC_s116
                                 + 1.01396 w_k1_KO_s116 + 1.00212 w_k1_MAT_s116
                                 + 1.00129 w_k1_MSFT_s116
                                 + 0.959908 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.23067 w_k1_ADM_s116
                                 + 1.6132 w_k1_AXP_s116 + 1.47342 w_k1_DIS_s116
                                 + 1.45782 w_k1_FDX_s116
                                 + 1.25659 w_k1_IBM_s116
                                 + 1.29181 w_k1_INTC_s116
                                 + 1.34914 w_k1_KO_s116 + 1.66918 w_k1_MAT_s116
                                 + 1.29017 w_k1_MSFT_s116
                                 + 1.22279 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.00478 w_k1_ADM_s116
                                 + 1.42432 w_k1_AXP_s116
                                 + 1.56667 w_k1_DIS_s116
                                 + 1.20653 w_k1_FDX_s116
                                 + 1.50507 w_k1_IBM_s116
                                 + 1.37078 w_k1_INTC_s116 + 1.4077 w_k1_KO_s116
                                 + 1.64019 w_k1_MAT_s116
                                 + 1.45533 w_k1_MSFT_s116
                                 + 1.08111 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.20973 w_k1_ADM_s117
                                 + 1.23352 w_k1_AXP_s117
                                 + 1.60653 w_k1_DIS_s117
                                 + 1.17431 w_k1_FDX_s117
                                 + 1.29633 w_k1_IBM_s117
                                 + 1.16284 w_k1_INTC_s117
                                 + 1.21778 w_k1_KO_s117 + 1.12043 w_k1_MAT_s117
                                 + 1.1118 w_k1_MSFT_s117 + 1.09525 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.755142 w_k1_ADM_s117
                                 + 0.912129 w_k1_AXP_s117
                                 + 0.748556 w_k1_DIS_s117
                                 + 0.767123 w_k1_FDX_s117
                                 + 0.991999 w_k1_IBM_s117
                                 + 0.777164 w_k1_INTC_s117
                                 + 1.00099 w_k1_KO_s117 + 1.51398 w_k1_MAT_s117
                                 + 0.82198 w_k1_MSFT_s117
                                 + 0.862365 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.01937 w_k1_ADM_s117
                                 + 1.31103 w_k1_AXP_s117
                                 + 1.23203 w_k1_DIS_s117
                                 + 1.16404 w_k1_FDX_s117
                                 + 0.824702 w_k1_IBM_s117
                                 + 0.802774 w_k1_INTC_s117
                                 + 1.07585 w_k1_KO_s117 + 1.20209 w_k1_MAT_s117
                                 + 1.08503 w_k1_MSFT_s117 + 1.0775 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.848587 w_k1_ADM_s117
                                 + 0.960747 w_k1_AXP_s117
                                 + 0.998081 w_k1_DIS_s117
                                 + 0.946724 w_k1_FDX_s117
                                 + 1.03581 w_k1_IBM_s117
                                 + 0.963354 w_k1_INTC_s117
                                 + 1.04162 w_k1_KO_s117 + 1.03355 w_k1_MAT_s117
                                 + 0.945257 w_k1_MSFT_s117
                                 + 0.990461 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.869261 w_k1_ADM_s117
                                 + 1.13076 w_k1_AXP_s117
                                 + 0.930999 w_k1_DIS_s117
                                 + 0.92444 w_k1_FDX_s117
                                 + 1.36774 w_k1_IBM_s117
                                 + 0.985609 w_k1_INTC_s117
                                 + 1.00899 w_k1_KO_s117 + 1.35787 w_k1_MAT_s117
                                 + 0.93843 w_k1_MSFT_s117
                                 + 0.867539 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.93062 w_k1_ADM_s117
                                 + 0.820571 w_k1_AXP_s117
                                 + 0.830299 w_k1_DIS_s117
                                 + 0.979925 w_k1_FDX_s117
                                 + 0.65694 w_k1_IBM_s117
                                 + 0.888583 w_k1_INTC_s117
                                 + 0.792893 w_k1_KO_s117
                                 + 0.463402 w_k1_MAT_s117
                                 + 0.979327 w_k1_MSFT_s117
                                 + 0.916868 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.936911 w_k1_ADM_s117
                                 + 0.847851 w_k1_AXP_s117
                                 + 0.738113 w_k1_DIS_s117
                                 + 0.891352 w_k1_FDX_s117
                                 + 0.81498 w_k1_IBM_s117
                                 + 0.818695 w_k1_INTC_s117
                                 + 0.880784 w_k1_KO_s117
                                 + 1.29461 w_k1_MAT_s117
                                 + 0.851068 w_k1_MSFT_s117
                                 + 0.917915 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.840055 w_k1_ADM_s117
                                 + 0.664966 w_k1_AXP_s117
                                 + 0.647894 w_k1_DIS_s117
                                 + 0.779146 w_k1_FDX_s117
                                 + 0.651775 w_k1_IBM_s117
                                 + 0.730704 w_k1_INTC_s117
                                 + 0.754663 w_k1_KO_s117
                                 + 0.632964 w_k1_MAT_s117
                                 + 0.760668 w_k1_MSFT_s117
                                 + 0.79628 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0542866545210708 w_k1_ADM_s000
                                 + 0.015739175055537 w_k1_ADM_s001
                                 + 0.0198253720103579 w_k1_ADM_s002
                                 + 0.0262849152760001 w_k1_ADM_s003
                                 + 0.0207895753255777 w_k1_ADM_s004
                                 + 0.0121487058281058 w_k1_ADM_s005
                                 + 0.0286846981748183 w_k1_ADM_s006
                                 + 0.0382710684494058 w_k1_ADM_s007
                                 + 0.0067741169922636 w_k1_ADM_s010
                                 + 0.0274196780146253 w_k1_ADM_s011
                                 + 0.0187070667210453 w_k1_ADM_s012
                                 + 0.00669394183305481 w_k1_ADM_s013
                                 + 0.0206378659931482 w_k1_ADM_s014
                                 + 0.0115841279885539 w_k1_ADM_s015
                                 + 0.0197353926001553 w_k1_ADM_s016
                                 + 0.0230798892854309 w_k1_ADM_s017
                                 + 0.0292960854776351 w_k1_ADM_s100
                                 + 0.029227436340654 w_k1_ADM_s101
                                 + 0.0412147753508119 w_k1_ADM_s102
                                 + 0.02215240414168 w_k1_ADM_s103
                                 + 0.0241332688277431 w_k1_ADM_s104
                                 + 0.0109604001069333 w_k1_ADM_s105
                                 + 0.0141366669222487 w_k1_ADM_s106
                                 + 0.0447201257519629 w_k1_ADM_s107
                                 + 0.0686600953198715 w_k1_ADM_s110
                                 + 0.0730442051641433 w_k1_ADM_s111
                                 + 0.051956159750784 w_k1_ADM_s112
                                 + 0.0267950449371556 w_k1_ADM_s113
                                 + 0.0393588996731842 w_k1_ADM_s114
                                 + 0.115973378580334 w_k1_ADM_s115
                                 + 0.0358436277691998 w_k1_ADM_s116
                                 + 0.0304436993029445 w_k1_ADM_s117
                                 + 0.0522205310344346 w_k1_AXP_s000
                                 + 0.0150126394939407 w_k1_AXP_s001
                                 + 0.0218042474112684 w_k1_AXP_s002
                                 + 0.0239180543884117 w_k1_AXP_s003
                                 + 0.0197275139053629 w_k1_AXP_s004
                                 + 0.0124367903126321 w_k1_AXP_s005
                                 + 0.0280068295302087 w_k1_AXP_s006
                                 + 0.0462995106562464 w_k1_AXP_s007
                                 + 0.00684484881845816 w_k1_AXP_s010
                                 + 0.0234363878634037 w_k1_AXP_s011
                                 + 0.0181472766941283 w_k1_AXP_s012
                                 + 0.00698259045365768 w_k1_AXP_s013
                                 + 0.0235428538191353 w_k1_AXP_s014
                                 + 0.00901195126399655 w_k1_AXP_s015
                                 + 0.0208436452146219 w_k1_AXP_s016
                                 + 0.0220460271495887 w_k1_AXP_s017
                                 + 0.0312513935965883 w_k1_AXP_s100
                                 + 0.0297256824730546 w_k1_AXP_s101
                                 + 0.0368439802502528 w_k1_AXP_s102
                                 + 0.0225141602277241 w_k1_AXP_s103
                                 + 0.0233000268841057 w_k1_AXP_s104
                                 + 0.0109114697670715 w_k1_AXP_s105
                                 + 0.0124543916554634 w_k1_AXP_s106
                                 + 0.0392529045604328 w_k1_AXP_s107
                                 + 0.0714531709092127 w_k1_AXP_s110
                                 + 0.0809176698998973 w_k1_AXP_s111
                                 + 0.062828003630836 w_k1_AXP_s112
                                 + 0.0346200934206306 w_k1_AXP_s113
                                 + 0.0355845576323083 w_k1_AXP_s114
                                 + 0.118718606465942 w_k1_AXP_s115
                                 + 0.0345391561200462 w_k1_AXP_s116
                                 + 0.0332606000095653 w_k1_AXP_s117
                                 + 0.0525322906338478 w_k1_DIS_s000
                                 + 0.0153659538049016 w_k1_DIS_s001
                                 + 0.0200943651497964 w_k1_DIS_s002
                                 + 0.0249902659944212 w_k1_DIS_s003
                                 + 0.0206274992926804 w_k1_DIS_s004
                                 + 0.0129930915275964 w_k1_DIS_s005
                                 + 0.0271201926246687 w_k1_DIS_s006
                                 + 0.0422454658674866 w_k1_DIS_s007
                                 + 0.00815081730005699 w_k1_DIS_s010
                                 + 0.02398056278823 w_k1_DIS_s011
                                 + 0.0184616222044805 w_k1_DIS_s012
                                 + 0.00778746104851036 w_k1_DIS_s013
                                 + 0.020228188104099 w_k1_DIS_s014
                                 + 0.0101344233482787 w_k1_DIS_s015
                                 + 0.0225306169617356 w_k1_DIS_s016
                                 + 0.0210686144433658 w_k1_DIS_s017
                                 + 0.0349438918869222 w_k1_DIS_s100
                                 + 0.0299384976865579 w_k1_DIS_s101
                                 + 0.0406474552879841 w_k1_DIS_s102
                                 + 0.0222082642178466 w_k1_DIS_s103
                                 + 0.0247581468425212 w_k1_DIS_s104
                                 + 0.0113408140814077 w_k1_DIS_s105
                                 + 0.0132494397802712 w_k1_DIS_s106
                                 + 0.0398988273055597 w_k1_DIS_s107
                                 + 0.0707577453790594 w_k1_DIS_s110
                                 + 0.0845602787508968 w_k1_DIS_s111
                                 + 0.0596812172601044 w_k1_DIS_s112
                                 + 0.0329028544428905 w_k1_DIS_s113
                                 + 0.0390610657311665 w_k1_DIS_s114
                                 + 0.120049200301884 w_k1_DIS_s115
                                 + 0.0355430882680849 w_k1_DIS_s116
                                 + 0.0344123944604692 w_k1_DIS_s117
                                 + 0.0518700409716367 w_k1_FDX_s000
                                 + 0.0143455367792698 w_k1_FDX_s001
                                 + 0.0192437939700341 w_k1_FDX_s002
                                 + 0.0237206309545387 w_k1_FDX_s003
                                 + 0.0196054926119611 w_k1_FDX_s004
                                 + 0.0131725952520422 w_k1_FDX_s005
                                 + 0.0261829792025898 w_k1_FDX_s006
                                 + 0.0424231958021539 w_k1_FDX_s007
                                 + 0.007428663585637 w_k1_FDX_s010
                                 + 0.0233854392383741 w_k1_FDX_s011
                                 + 0.0177684854344142 w_k1_FDX_s012
                                 + 0.00764191684616206 w_k1_FDX_s013
                                 + 0.0220276504743786 w_k1_FDX_s014
                                 + 0.010041001825994 w_k1_FDX_s015
                                 + 0.0211822204908094 w_k1_FDX_s016
                                 + 0.0212662389132305 w_k1_FDX_s017
                                 + 0.0321629704080515 w_k1_FDX_s100
                                 + 0.0305304379366695 w_k1_FDX_s101
                                 + 0.0413898270171567 w_k1_FDX_s102
                                 + 0.0220753814003793 w_k1_FDX_s103
                                 + 0.0242753429681037 w_k1_FDX_s104
                                 + 0.0114953707456837 w_k1_FDX_s105
                                 + 0.0130748660758399 w_k1_FDX_s106
                                 + 0.0390397280748062 w_k1_FDX_s107
                                 + 0.0682210685631958 w_k1_FDX_s110
                                 + 0.0820390181960576 w_k1_FDX_s111
                                 + 0.0570539949170722 w_k1_FDX_s112
                                 + 0.0307607080574595 w_k1_FDX_s113
                                 + 0.0381541338332006 w_k1_FDX_s114
                                 + 0.116500950335324 w_k1_FDX_s115
                                 + 0.0359723352485926 w_k1_FDX_s116
                                 + 0.0318439425922036 w_k1_FDX_s117
                                 + 0.0563428639065536 w_k1_IBM_s000
                                 + 0.014815030504348 w_k1_IBM_s001
                                 + 0.0196318616509537 w_k1_IBM_s002
                                 + 0.0239804439434939 w_k1_IBM_s003
                                 + 0.0197300416994743 w_k1_IBM_s004
                                 + 0.0137460868041424 w_k1_IBM_s005
                                 + 0.0273071844875819 w_k1_IBM_s006
                                 + 0.03865943558379 w_k1_IBM_s007
                                 + 0.00714628025886367 w_k1_IBM_s010
                                 + 0.0236132371911082 w_k1_IBM_s011
                                 + 0.0188057980015184 w_k1_IBM_s012
                                 + 0.00766094377774081 w_k1_IBM_s013
                                 + 0.0216021111525214 w_k1_IBM_s014
                                 + 0.0100716948495954 w_k1_IBM_s015
                                 + 0.0208520426488996 w_k1_IBM_s016
                                 + 0.0220188175169134 w_k1_IBM_s017
                                 + 0.0335370824162721 w_k1_IBM_s100
                                 + 0.0311421953598428 w_k1_IBM_s101
                                 + 0.0395420975886362 w_k1_IBM_s102
                                 + 0.0218457891706425 w_k1_IBM_s103
                                 + 0.02542195487999 w_k1_IBM_s104
                                 + 0.0123308183872905 w_k1_IBM_s105
                                 + 0.012751983814731 w_k1_IBM_s106
                                 + 0.0399643105092461 w_k1_IBM_s107
                                 + 0.0697249820120984 w_k1_IBM_s110
                                 + 0.0849824155331213 w_k1_IBM_s111
                                 + 0.0595145017069828 w_k1_IBM_s112
                                 + 0.0316251495645269 w_k1_IBM_s113
                                 + 0.0389859960092815 w_k1_IBM_s114
                                 + 0.120241316123326 w_k1_IBM_s115
                                 + 0.0368961014318385 w_k1_IBM_s116
                                 + 0.0317331404655909 w_k1_IBM_s117
                                 + 0.0590613315190757 w_k1_INTC_s000
                                 + 0.0137181659559101 w_k1_INTC_s001
                                 + 0.0194579846554694 w_k1_INTC_s002
                                 + 0.0215301101591576 w_k1_INTC_s003
                                 + 0.0174083067213972 w_k1_INTC_s004
                                 + 0.0132260371930398 w_k1_INTC_s005
                                 + 0.0285168940851663 w_k1_INTC_s006
                                 + 0.0420105685991937 w_k1_INTC_s007
                                 + 0.00677908617306891 w_k1_INTC_s010
                                 + 0.0252942033040562 w_k1_INTC_s011
                                 + 0.018811105823125 w_k1_INTC_s012
                                 + 0.00813806191863825 w_k1_INTC_s013
                                 + 0.0261469807315047 w_k1_INTC_s014
                                 + 0.0100267130747132 w_k1_INTC_s015
                                 + 0.0170110657848375 w_k1_INTC_s016
                                 + 0.0170657885961616 w_k1_INTC_s017
                                 + 0.0332391088089138 w_k1_INTC_s100
                                 + 0.0283769235602602 w_k1_INTC_s101
                                 + 0.0419859658579158 w_k1_INTC_s102
                                 + 0.0215956096618795 w_k1_INTC_s103
                                 + 0.0251811514707457 w_k1_INTC_s104
                                 + 0.0125646396865261 w_k1_INTC_s105
                                 + 0.0102605759104708 w_k1_INTC_s106
                                 + 0.0410175696729863 w_k1_INTC_s107
                                 + 0.0682205447913111 w_k1_INTC_s110
                                 + 0.0885386912168386 w_k1_INTC_s111
                                 + 0.0541363605298123 w_k1_INTC_s112
                                 + 0.0317037963013136 w_k1_INTC_s113
                                 + 0.0405501904824039 w_k1_INTC_s114
                                 + 0.115300229522301 w_k1_INTC_s115
                                 + 0.0371663923250588 w_k1_INTC_s116
                                 + 0.0295347881965056 w_k1_INTC_s117
                                 + 0.0524043818102769 w_k1_KO_s000
                                 + 0.0141554555283122 w_k1_KO_s001
                                 + 0.0213859178082418 w_k1_KO_s002
                                 + 0.0258189035081812 w_k1_KO_s003
                                 + 0.0194432854416092 w_k1_KO_s004
                                 + 0.0139744802005191 w_k1_KO_s005
                                 + 0.0262472611395138 w_k1_KO_s006
                                 + 0.037413541674244 w_k1_KO_s007
                                 + 0.0067928998657663 w_k1_KO_s010
                                 + 0.0225122484990822 w_k1_KO_s011
                                 + 0.0201770254386656 w_k1_KO_s012
                                 + 0.00757720495092134 w_k1_KO_s013
                                 + 0.0202879903951034 w_k1_KO_s014
                                 + 0.0103771869916499 w_k1_KO_s015
                                 + 0.0199256747321196 w_k1_KO_s016
                                 + 0.0193435356328403 w_k1_KO_s017
                                 + 0.0336635100850393 w_k1_KO_s100
                                 + 0.0286993693841672 w_k1_KO_s101
                                 + 0.0379666317058252 w_k1_KO_s102
                                 + 0.0217223262429604 w_k1_KO_s103
                                 + 0.0241248358672286 w_k1_KO_s104
                                 + 0.0133256226400196 w_k1_KO_s105
                                 + 0.0122797517132428 w_k1_KO_s106
                                 + 0.0398461931605519 w_k1_KO_s107
                                 + 0.0638527484635566 w_k1_KO_s110
                                 + 0.0876122829820623 w_k1_KO_s111
                                 + 0.0565462480422534 w_k1_KO_s112
                                 + 0.0306641863036555 w_k1_KO_s113
                                 + 0.0351718966301348 w_k1_KO_s114
                                 + 0.110712146295687 w_k1_KO_s115
                                 + 0.0363390658105993 w_k1_KO_s116
                                 + 0.0324858329101015 w_k1_KO_s117
                                 + 0.0498090702087629 w_k1_MAT_s000
                                 + 0.0146950223868398 w_k1_MAT_s001
                                 + 0.0194782324811977 w_k1_MAT_s002
                                 + 0.0248943335038207 w_k1_MAT_s003
                                 + 0.0271305490630015 w_k1_MAT_s004
                                 + 0.0122113289183593 w_k1_MAT_s005
                                 + 0.0237745620269991 w_k1_MAT_s006
                                 + 0.0453021440962353 w_k1_MAT_s007
                                 + 0.00738644717656303 w_k1_MAT_s010
                                 + 0.0232851304563518 w_k1_MAT_s011
                                 + 0.0186678308253557 w_k1_MAT_s012
                                 + 0.00850907527235875 w_k1_MAT_s013
                                 + 0.02114016041653 w_k1_MAT_s014
                                 + 0.00960217977437618 w_k1_MAT_s015
                                 + 0.0230400362017284 w_k1_MAT_s016
                                 + 0.0210722613929338 w_k1_MAT_s017
                                 + 0.0323414593036934 w_k1_MAT_s100
                                 + 0.0311961231088431 w_k1_MAT_s101
                                 + 0.0438035776447524 w_k1_MAT_s102
                                 + 0.0216059679551306 w_k1_MAT_s103
                                 + 0.0268119270750238 w_k1_MAT_s104
                                 + 0.0113912477654108 w_k1_MAT_s105
                                 + 0.0136331415637025 w_k1_MAT_s106
                                 + 0.0397762886872843 w_k1_MAT_s107
                                 + 0.0713810009950938 w_k1_MAT_s110
                                 + 0.0747533301259432 w_k1_MAT_s111
                                 + 0.0559549919371325 w_k1_MAT_s112
                                 + 0.033986141926428 w_k1_MAT_s113
                                 + 0.0366833072680789 w_k1_MAT_s114
                                 + 0.114393198734711 w_k1_MAT_s115
                                 + 0.0389462697572475 w_k1_MAT_s116
                                 + 0.0343975361726698 w_k1_MAT_s117
                                 + 0.0517240204151898 w_k1_MSFT_s000
                                 + 0.0145581738275348 w_k1_MSFT_s001
                                 + 0.0182948851005049 w_k1_MSFT_s002
                                 + 0.0226209231882703 w_k1_MSFT_s003
                                 + 0.0189166253797792 w_k1_MSFT_s004
                                 + 0.0123809317129549 w_k1_MSFT_s005
                                 + 0.0266005962912033 w_k1_MSFT_s006
                                 + 0.0419378601192714 w_k1_MSFT_s007
                                 + 0.00707942432483325 w_k1_MSFT_s010
                                 + 0.0242223001768834 w_k1_MSFT_s011
                                 + 0.0176475832383916 w_k1_MSFT_s012
                                 + 0.00729890525348997 w_k1_MSFT_s013
                                 + 0.0232766085041293 w_k1_MSFT_s014
                                 + 0.0109105295724094 w_k1_MSFT_s015
                                 + 0.0201895416920799 w_k1_MSFT_s016
                                 + 0.0197651751244936 w_k1_MSFT_s017
                                 + 0.0312785736607965 w_k1_MSFT_s100
                                 + 0.0280906734214103 w_k1_MSFT_s101
                                 + 0.0409197955866511 w_k1_MSFT_s102
                                 + 0.021882697685278 w_k1_MSFT_s103
                                 + 0.0258412671441305 w_k1_MSFT_s104
                                 + 0.0114772112850138 w_k1_MSFT_s105
                                 + 0.0121636316716679 w_k1_MSFT_s106
                                 + 0.0350766127721037 w_k1_MSFT_s107
                                 + 0.0629671967369916 w_k1_MSFT_s110
                                 + 0.0899978600307866 w_k1_MSFT_s111
                                 + 0.0529644219390898 w_k1_MSFT_s112
                                 + 0.0336017458308476 w_k1_MSFT_s113
                                 + 0.0389520576502254 w_k1_MSFT_s114
                                 + 0.103541986689566 w_k1_MSFT_s115
                                 + 0.0379910614944953 w_k1_MSFT_s116
                                 + 0.0310182912133427 w_k1_MSFT_s117
                                 + 0.0524199258237141 w_k1_PG_s000
                                 + 0.0137591122044183 w_k1_PG_s001
                                 + 0.0207798172593826 w_k1_PG_s002
                                 + 0.0269236964349751 w_k1_PG_s003
                                 + 0.0200712998590178 w_k1_PG_s004
                                 + 0.0140955712899891 w_k1_PG_s005
                                 + 0.0253100893466455 w_k1_PG_s006
                                 + 0.0395515625441046 w_k1_PG_s007
                                 + 0.0074544238946797 w_k1_PG_s010
                                 + 0.0228445124015474 w_k1_PG_s011
                                 + 0.0200789085215197 w_k1_PG_s012
                                 + 0.00774680527286661 w_k1_PG_s013
                                 + 0.0199993972221355 w_k1_PG_s014
                                 + 0.00952362052582881 w_k1_PG_s015
                                 + 0.0192704595073381 w_k1_PG_s016
                                 + 0.0208795700502059 w_k1_PG_s017
                                 + 0.0302909558199033 w_k1_PG_s100
                                 + 0.0315330356636891 w_k1_PG_s101
                                 + 0.0384111252765027 w_k1_PG_s102
                                 + 0.0224109376785389 w_k1_PG_s103
                                 + 0.0231972439335648 w_k1_PG_s104
                                 + 0.0113074166710082 w_k1_PG_s105
                                 + 0.0123647172996151 w_k1_PG_s106
                                 + 0.0367361166713061 w_k1_PG_s107
                                 + 0.0687552840012002 w_k1_PG_s110
                                 + 0.0817656239344587 w_k1_PG_s111
                                 + 0.0560035066354965 w_k1_PG_s112
                                 + 0.0321655804121916 w_k1_PG_s113
                                 + 0.0372436908310688 w_k1_PG_s114
                                 + 0.116843230599253 w_k1_PG_s115
                                 + 0.0338022897657989 w_k1_PG_s116
                                 + 0.0311405235355924 w_k1_PG_s117 >= 1.06
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
