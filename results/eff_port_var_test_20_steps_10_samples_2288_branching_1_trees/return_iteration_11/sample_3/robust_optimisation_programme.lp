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
 w_k1_ADM_s0_return_constraint:  1.13329 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.873902 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.1391 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.995907 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.913507 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.21618 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.822422 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.06193 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.36832 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.716622 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.5159 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.07451 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.797094 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.939798 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.888797 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.95124 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.765469 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.831298 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.936667 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.671398 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.993429 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.01443 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.35444 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.745974 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.785372 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.966689 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.11329 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.25359 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.978579 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.08452 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.24591 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.770028 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.886262 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.16131 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.879871 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.21697 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.2755 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.849237 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.28367 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.906425 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.07736 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.933198 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.813331 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.14917 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.763175 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.0775 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.50673 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.577711 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.55053 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.07517 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.72425 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.933808 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.86992 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.11861 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.958049 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.794492 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.18984 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.593373 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.40288 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.54694 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.43089 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.62181 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.742448 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.843958 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.17668 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.2694 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.874915 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.981502 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.30981 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.78044 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.1107 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.20207 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.804419 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.2274 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.79379 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.834673 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.30583 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.882003 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.12668 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.92031 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.81012 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.20524 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.683148 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.26546 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.68253 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.51742 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.93092 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.02524 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.681009 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.862281 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.871172 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.3676 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.815818 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.729573 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.18953 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.499071 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.7045 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.33437 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.7267 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.494262 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.605614 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.749733 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.09233 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.31485 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.87933 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.990228 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.30675 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.753624 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.0788 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.16224 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.690517 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.23208 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.98843 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.728618 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19345 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.914747 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.11793 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.902707 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.917883 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.08911 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.897028 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.10437 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.46078 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.654594 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.39517 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.9718 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.703641 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.952851 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.922971 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.14297 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.87399 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.738623 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.11931 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.73271 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.23356 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.28884 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.48227 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.666189 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.672081 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.807305 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.922197 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.21887 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.79601 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.02842 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.36497 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.757604 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.02951 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.1243 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.869283 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.11013 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.61114 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.782623 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.23773 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.938775 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.880806 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.23189 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.742421 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.22201 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.579431 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.0065 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.0604 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.634059 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.73815 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.42336 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.870945 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.864865 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.663412 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.16772 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.18789 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.823656 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.840836 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.518028 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.75297 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.19462 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.12926 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.698492 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.08429 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.881995 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.57284 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.12476 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.224 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.779089 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.811691 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.34072 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.16819 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.02981 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.680856 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.77084 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.22646 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.992187 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11352 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.954563 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.09971 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.933699 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.875875 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.02908 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.816953 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.11916 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.25916 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.759386 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.15595 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.07664 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.736854 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.911524 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.823618 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.36659 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.0564 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.905083 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.09114 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.820645 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.39854 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.22266 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.31121 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.696867 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.821902 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.86663 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.952362 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.3166 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.931665 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.933511 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.13045 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.891035 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.17401 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.91746 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.96196 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.0076 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.34962 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.739001 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.24453 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.888719 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.01342 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.07895 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.826091 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.13224 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.663424 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.04374 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.20549 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.671812 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.48812 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.25954 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.786731 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.908032 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.758128 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.13816 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.906322 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.832464 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.953942 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.545541 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.65918 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.07502 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.31869 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.720715 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.897832 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.817005 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.14015 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.18068 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.04375 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.855407 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.11427 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.0561 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.14017 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.19211 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.810879 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.44653 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.51718 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.964299 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.4108 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.976225 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.859319 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.21272 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.762213 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.12752 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.517562 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.984499 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.16608 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.573837 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.83094 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.38824 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.830615 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.01461 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.615739 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.988128 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.14912 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.709133 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.805042 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.557137 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.93119 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.22953 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.12335 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.657082 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.920103 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.761663 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.3007 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.20351 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.88475 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.601406 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.912778 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.05198 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.47757 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.13081 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.760502 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.91774 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.97069 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.00567 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15498 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.916052 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.14307 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.938804 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.960842 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.00252 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.840336 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.1599 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.18761 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.710787 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.20951 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.999256 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.696129 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.866588 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.94174 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.22005 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.866705 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.798275 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.05511 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.732667 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.29894 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.18913 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.29599 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.705874 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.716154 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.845166 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.944634 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.12622 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.832145 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.978637 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.33002 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.876829 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.18724 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.15386 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.959342 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.21386 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.55151 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.854479 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15507 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.940519 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.947715 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.981373 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.879814 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.04537 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.881378 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.11843 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.22565 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.79352 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.35909 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.18381 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.910764 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.0211 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.909124 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.15213 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.956718 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.827489 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.969326 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.707937 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.18131 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.12908 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.27269 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.796186 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.845998 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.896224 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.02694 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.13539 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.00384 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.07215 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.09721 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.998136 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.06317 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.13229 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.926514 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.17745 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.39438 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.981383 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.385000801728004 z_k1_s0000
                                 - 0.198999999514016 z_k1_s0001
                                 - 0.130999953652952 z_k1_s0002
                                 - 0.245999108044592 z_k1_s0003
                                 - 0.353999588259164 z_k1_s0004
                                 - 0.57999993122679 z_k1_s0005
                                 - 0.151000059129722 z_k1_s0006
                                 - 0.188000008521783 z_k1_s0007
                                 - 0.767998064611787 z_k1_s0010
                                 - 0.701999910228705 z_k1_s0011
                                 - 0.525999099881949 z_k1_s0012
                                 - 0.439997726872988 z_k1_s0013
                                 - 0.410999293406622 z_k1_s0014
                                 - 0.211999725483094 z_k1_s0015
                                 - 0.525999099881949 z_k1_s0016
                                 - 0.71399840220911 z_k1_s0017
                                 - 0.0890000470736242 z_k1_s0020
                                 - 0.313000207739835 z_k1_s0021
                                 - 0.211000145754331 z_k1_s0022
                                 - 0.208999373244615 z_k1_s0023
                                 - 0.355999830694406 z_k1_s0024
                                 - 0.208999373244615 z_k1_s0025
                                 - 0.186000323385384 z_k1_s0026
                                 - 0.214999902773308 z_k1_s0027
                                 - 0.129999955767602 z_k1_s0030
                                 - 0.37900022527606 z_k1_s0031
                                 - 0.121999861566589 z_k1_s0032
                                 - 0.220000340529487 z_k1_s0033
                                 - 0.11199996881516 z_k1_s0034
                                 - 0.109999870264961 z_k1_s0035
                                 - 0.313000273117105 z_k1_s0036
                                 - 0.113999917365467 z_k1_s0037
                                 - 0.107999911739057 z_k1_s0040
                                 - 0.446999174369981 z_k1_s0041
                                 - 0.107999911739057 z_k1_s0042
                                 - 0.156999932037334 z_k1_s0043
                                 - 0.25600079764815 z_k1_s0044
                                 - 0.441999704298736 z_k1_s0045
                                 - 0.117000075067315 z_k1_s0046
                                 - 0.227000147234875 z_k1_s0047
                                 - 0.240999488145485 z_k1_s0050
                                 - 0.618998990953239 z_k1_s0051
                                 - 0.423999128798646 z_k1_s0052
                                 - 0.307000298704051 z_k1_s0053
                                 - 0.280999954684051 z_k1_s0054
                                 - 0.27699906545162 z_k1_s0055
                                 - 0.273998398527296 z_k1_s0056
                                 - 0.295000349002159 z_k1_s0057
                                 - 0.321000344022265 z_k1_s0060
                                 - 0.117999963270529 z_k1_s0061
                                 - 0.166000036509367 z_k1_s0062
                                 - 0.236000644140888 z_k1_s0063
                                 - 0.19899940289873 z_k1_s0064
                                 - 0.10599990011083 z_k1_s0065
                                 - 0.28900077359626 z_k1_s0066
                                 - 0.358999049028333 z_k1_s0067
                                 - 0.289000030011921 z_k1_s0070
                                 - 0.617998117583193 z_k1_s0071
                                 - 0.602997914598065 z_k1_s0072
                                 - 1.19300279401187 z_k1_s0073
                                 - 0.628997955972163 z_k1_s0074
                                 - 0.289000030011921 z_k1_s0075
                                 - 0.451000359450562 z_k1_s0076
                                 - 0.584998602416282 z_k1_s0077
                                 - 0.184999808768248 z_k1_s0100
                                 - 0.148999891853288 z_k1_s0101
                                 - 0.198999740190777 z_k1_s0102
                                 - 0.302998919901379 z_k1_s0103
                                 - 0.215999719089485 z_k1_s0104
                                 - 0.379998760325019 z_k1_s0105
                                 - 0.25599955423953 z_k1_s0106
                                 - 0.488000034268006 z_k1_s0107
                                 - 0.217999563418376 z_k1_s0110
                                 - 0.0789998417892279 z_k1_s0111
                                 - 0.165999667557112 z_k1_s0112
                                 - 0.346999305074204 z_k1_s0113
                                 - 0.162999673565116 z_k1_s0114
                                 - 0.31999935914624 z_k1_s0115
                                 - 0.188999621495748 z_k1_s0116
                                 - 0.117999763685176 z_k1_s0117
                                 - 0.145999775195746 z_k1_s0120
                                 - 0.286000341950824 z_k1_s0121
                                 - 0.331999712321514 z_k1_s0122
                                 - 0.288000078097713 z_k1_s0123
                                 - 0.181999558333283 z_k1_s0124
                                 - 0.107999893311837 z_k1_s0125
                                 - 0.337998920762182 z_k1_s0126
                                 - 0.132999858543295 z_k1_s0127
                                 - 0.208000332398968 z_k1_s0130
                                 - 0.20300042845064 z_k1_s0131
                                 - 0.221000395005236 z_k1_s0132
                                 - 0.426001365096349 z_k1_s0133
                                 - 0.379000795519166 z_k1_s0134
                                 - 0.155000368904758 z_k1_s0135
                                 - 0.323000354214905 z_k1_s0136
                                 - 0.316001693429619 z_k1_s0137
                                 - 0.163000573330148 z_k1_s0140
                                 - 0.441001291426203 z_k1_s0141
                                 - 0.539001578409804 z_k1_s0142
                                 - 0.231000676461345 z_k1_s0143
                                 - 0.135000491334833 z_k1_s0144
                                 - 0.368001717654579 z_k1_s0145
                                 - 0.15200034911714 z_k1_s0146
                                 - 0.211000553893075 z_k1_s0147
                                 - 0.0580000162534372 z_k1_s0150
                                 - 0.0870000714801883 z_k1_s0151
                                 - 0.183000582375867 z_k1_s0152
                                 - 0.166000296635872 z_k1_s0153
                                 - 0.150000415586733 z_k1_s0154
                                 - 0.0780000088649583 z_k1_s0155
                                 - 0.173000303469911 z_k1_s0156
                                 - 0.0469999920570811 z_k1_s0157
                                 - 0.251000129318228 z_k1_s0160
                                 - 0.193999934552444 z_k1_s0161
                                 - 0.340001225094257 z_k1_s0162
                                 - 0.105000102278626 z_k1_s0163
                                 - 0.295000691331132 z_k1_s0164
                                 - 0.20499962380488 z_k1_s0165
                                 - 0.171000223296718 z_k1_s0166
                                 - 0.244000327066677 z_k1_s0167
                                 - 0.200999944364653 z_k1_s0170
                                 - 0.192999946578995 z_k1_s0171
                                 - 0.156999956543535 z_k1_s0172
                                 - 0.0759999789637495 z_k1_s0173
                                 - 0.120999966508075 z_k1_s0174
                                 - 0.108999969829588 z_k1_s0175
                                 - 0.0589999836692266 z_k1_s0176
                                 - 0.0839999767494073 z_k1_s0177
                                 - 0.338000403011773 z_k1_s1000
                                 - 0.25399969373047 z_k1_s1001
                                 - 0.451001357164002 z_k1_s1002
                                 - 0.107999931532668 z_k1_s1003
                                 - 0.297000056814947 z_k1_s1004
                                 - 0.419001086961601 z_k1_s1005
                                 - 0.371000681657999 z_k1_s1006
                                 - 0.21300008283397 z_k1_s1007
                                 - 0.450001409366573 z_k1_s1010
                                 - 0.201000031103413 z_k1_s1011
                                 - 0.503999346569355 z_k1_s1012
                                 - 0.35899967415166 z_k1_s1013
                                 - 0.308000274570526 z_k1_s1014
                                 - 0.222000226260589 z_k1_s1015
                                 - 0.154000137285263 z_k1_s1016
                                 - 0.228000212705459 z_k1_s1017
                                 - 0.334000395174054 z_k1_s1020
                                 - 0.129999935659531 z_k1_s1021
                                 - 0.134999967084872 z_k1_s1022
                                 - 0.067000009779879 z_k1_s1023
                                 - 0.183000533187934 z_k1_s1024
                                 - 0.220000207515885 z_k1_s1025
                                 - 0.25999957751929 z_k1_s1026
                                 - 0.139999851610326 z_k1_s1027
                                 - 0.206999548072052 z_k1_s1030
                                 - 0.477997787603602 z_k1_s1031
                                 - 0.158999800728949 z_k1_s1032
                                 - 0.631999831260935 z_k1_s1033
                                 - 0.379998607089966 z_k1_s1034
                                 - 0.477997787603602 z_k1_s1035
                                 - 0.433998869412317 z_k1_s1036
                                 - 0.146999645268629 z_k1_s1037
                                 - 0.182000052336109 z_k1_s1040
                                 - 0.355000224062198 z_k1_s1041
                                 - 0.255000039784106 z_k1_s1042
                                 - 0.829000781190334 z_k1_s1043
                                 - 0.248999934377414 z_k1_s1044
                                 - 0.190999932946128 z_k1_s1045
                                 - 0.263999920394125 z_k1_s1046
                                 - 0.44999958050133 z_k1_s1047
                                 - 0.494997107715691 z_k1_s1050
                                 - 0.203999709617035 z_k1_s1051
                                 - 0.707996728682189 z_k1_s1052
                                 - 0.298999214638085 z_k1_s1053
                                 - 0.897999705714789 z_k1_s1054
                                 - 0.738998759446966 z_k1_s1055
                                 - 0.414998777273913 z_k1_s1056
                                 - 0.208999307745359 z_k1_s1057
                                 - 0.406000976557478 z_k1_s1060
                                 - 0.369000160466948 z_k1_s1061
                                 - 0.496999416129932 z_k1_s1062
                                 - 0.45799939917001 z_k1_s1063
                                 - 0.289999326111888 z_k1_s1064
                                 - 0.115999970444859 z_k1_s1065
                                 - 0.163999991318609 z_k1_s1066
                                 - 0.100000123487013 z_k1_s1067
                                 - 0.611000324992766 z_k1_s1070
                                 - 0.844002132761498 z_k1_s1071
                                 - 0.416000263858945 z_k1_s1072
                                 - 1.13199928736865 z_k1_s1073
                                 - 1.50400367407724 z_k1_s1074
                                 - 0.507000425828189 z_k1_s1075
                                 - 0.464000679227652 z_k1_s1076
                                 - 1.19400224248887 z_k1_s1077
                                 - 1.14099559535149 z_k1_s1100
                                 - 0.693998299264136 z_k1_s1101
                                 - 0.552998046621311 z_k1_s1102
                                 - 0.526998226843897 z_k1_s1103
                                 - 1.17299379542103 z_k1_s1104
                                 - 0.629995865144285 z_k1_s1105
                                 - 0.774998457237634 z_k1_s1106
                                 - 0.540998269046677 z_k1_s1107
                                 - 0.480999628314127 z_k1_s1110
                                 - 0.355000238863578 z_k1_s1111
                                 - 0.142999953801462 z_k1_s1112
                                 - 0.581999587365514 z_k1_s1113
                                 - 0.122999814682456 z_k1_s1114
                                 - 0.399000544925392 z_k1_s1115
                                 - 0.383000433630187 z_k1_s1116
                                 - 0.508000559624382 z_k1_s1117
                                 - 0.334998790004362 z_k1_s1120
                                 - 0.606998677452558 z_k1_s1121
                                 - 0.421998306468007 z_k1_s1122
                                 - 0.212999165968124 z_k1_s1123
                                 - 0.916994887515726 z_k1_s1124
                                 - 0.799995871687333 z_k1_s1125
                                 - 0.749999011569604 z_k1_s1126
                                 - 0.566998422770155 z_k1_s1127
                                 - 0.471001423670351 z_k1_s1130
                                 - 0.666998645239696 z_k1_s1131
                                 - 0.175000064092819 z_k1_s1132
                                 - 0.358999058204078 z_k1_s1133
                                 - 0.19899997800731 z_k1_s1134
                                 - 0.66399969369988 z_k1_s1135
                                 - 0.449998385896688 z_k1_s1136
                                 - 0.474000375210167 z_k1_s1137
                                 - 1.58300387051719 z_k1_s1140
                                 - 2.18800450072978 z_k1_s1141
                                 - 0.599000158154155 z_k1_s1142
                                 - 1.38300258724001 z_k1_s1143
                                 - 1.32600027048547 z_k1_s1144
                                 - 1.17000533937094 z_k1_s1145
                                 - 2.03599832271769 z_k1_s1146
                                 - 0.553999869416788 z_k1_s1147
                                 - 0.305000685256524 z_k1_s1150
                                 - 0.334000443811496 z_k1_s1151
                                 - 0.276000499700763 z_k1_s1152
                                 - 0.837002675725109 z_k1_s1153
                                 - 0.308000365796149 z_k1_s1154
                                 - 0.33100076327187 z_k1_s1155
                                 - 0.899003190223849 z_k1_s1156
                                 - 0.980000969805571 z_k1_s1157
                                 - 0.253000718395495 z_k1_s1160
                                 - 0.440000083817246 z_k1_s1161
                                 - 0.403000543878472 z_k1_s1162
                                 - 0.430000691185562 z_k1_s1163
                                 - 0.253000718395495 z_k1_s1164
                                 - 0.1400002094511 z_k1_s1165
                                 - 0.161000050978601 z_k1_s1166
                                 - 0.154000029336036 z_k1_s1167
                                 - 0.34200044564359 z_k1_s1170
                                 - 0.318000824546668 z_k1_s1171
                                 - 0.609003536612409 z_k1_s1172
                                 - 1.34900549212989 z_k1_s1173
                                 - 1.25000267134575 z_k1_s1174
                                 - 0.585003915515486 z_k1_s1175
                                 - 0.713003843036555 z_k1_s1176
                                 - 0.679003405647174 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.0158 w_k1_ADM_s000
                                 + 1.0109 w_k1_AXP_s000 + 1.26478 w_k1_DIS_s000
                                 + 1.17004 w_k1_FDX_s000
                                 + 0.907821 w_k1_IBM_s000
                                 + 1.26824 w_k1_INTC_s000
                                 + 1.07315 w_k1_KO_s000 + 1.07566 w_k1_MAT_s000
                                 + 1.16736 w_k1_MSFT_s000 + 1.0758 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.954373 w_k1_ADM_s000
                                 + 0.955642 w_k1_AXP_s000
                                 + 0.901551 w_k1_DIS_s000
                                 + 0.949004 w_k1_FDX_s000
                                 + 0.644845 w_k1_IBM_s000
                                 + 1.1773 w_k1_INTC_s000
                                 + 0.672384 w_k1_KO_s000
                                 + 0.407872 w_k1_MAT_s000
                                 + 0.923018 w_k1_MSFT_s000
                                 + 0.911573 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.51353 w_k1_ADM_s000
                                 + 1.71173 w_k1_AXP_s000 + 1.6992 w_k1_DIS_s000
                                 + 1.40265 w_k1_FDX_s000
                                 + 0.994592 w_k1_IBM_s000
                                 + 1.36634 w_k1_INTC_s000
                                 + 1.42624 w_k1_KO_s000 + 1.29514 w_k1_MAT_s000
                                 + 1.35454 w_k1_MSFT_s000
                                 + 1.30428 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.33707 w_k1_ADM_s000
                                 + 1.08903 w_k1_AXP_s000
                                 + 1.69655 w_k1_DIS_s000
                                 + 1.22095 w_k1_FDX_s000
                                 + 1.34938 w_k1_IBM_s000
                                 + 1.38138 w_k1_INTC_s000
                                 + 1.46175 w_k1_KO_s000 + 1.11249 w_k1_MAT_s000
                                 + 1.32618 w_k1_MSFT_s000
                                 + 1.18849 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.980886 w_k1_ADM_s000
                                 + 0.731787 w_k1_AXP_s000
                                 + 0.795349 w_k1_DIS_s000
                                 + 0.782477 w_k1_FDX_s000
                                 + 0.855051 w_k1_IBM_s000
                                 + 1.0765 w_k1_INTC_s000
                                 + 0.868643 w_k1_KO_s000
                                 + 0.61166 w_k1_MAT_s000
                                 + 0.914765 w_k1_MSFT_s000
                                 + 0.836392 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.09118 w_k1_ADM_s000
                                 + 0.996892 w_k1_AXP_s000
                                 + 0.932833 w_k1_DIS_s000
                                 + 0.899587 w_k1_FDX_s000
                                 + 1.2061 w_k1_IBM_s000
                                 + 0.998607 w_k1_INTC_s000
                                 + 0.957443 w_k1_KO_s000
                                 + 1.07822 w_k1_MAT_s000
                                 + 0.857203 w_k1_MSFT_s000
                                 + 0.984169 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.844015 w_k1_ADM_s000
                                 + 0.628578 w_k1_AXP_s000
                                 + 0.63439 w_k1_DIS_s000
                                 + 0.67885 w_k1_FDX_s000
                                 + 0.762053 w_k1_IBM_s000
                                 + 0.841693 w_k1_INTC_s000
                                 + 0.802419 w_k1_KO_s000
                                 + 0.614224 w_k1_MAT_s000
                                 + 0.71601 w_k1_MSFT_s000
                                 + 0.714149 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.30857 w_k1_ADM_s000
                                 + 1.34205 w_k1_AXP_s000
                                 + 1.03899 w_k1_DIS_s000
                                 + 1.10968 w_k1_FDX_s000
                                 + 0.955486 w_k1_IBM_s000
                                 + 1.20542 w_k1_INTC_s000
                                 + 1.04275 w_k1_KO_s000
                                 + 0.966642 w_k1_MAT_s000
                                 + 1.05201 w_k1_MSFT_s000
                                 + 0.896031 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.884314 w_k1_ADM_s001
                                 + 0.742451 w_k1_AXP_s001
                                 + 0.666516 w_k1_DIS_s001
                                 + 0.776087 w_k1_FDX_s001
                                 + 0.826632 w_k1_IBM_s001
                                 + 0.880219 w_k1_INTC_s001
                                 + 0.767873 w_k1_KO_s001
                                 + 0.673572 w_k1_MAT_s001
                                 + 0.808865 w_k1_MSFT_s001
                                 + 0.972656 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.10022 w_k1_ADM_s001
                                 + 0.940294 w_k1_AXP_s001
                                 + 0.833932 w_k1_DIS_s001
                                 + 1.05514 w_k1_FDX_s001
                                 + 0.644549 w_k1_IBM_s001
                                 + 0.993209 w_k1_INTC_s001
                                 + 0.783543 w_k1_KO_s001
                                 + 0.530507 w_k1_MAT_s001
                                 + 0.968669 w_k1_MSFT_s001
                                 + 0.849675 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.15739 w_k1_ADM_s001
                                 + 1.42099 w_k1_AXP_s001
                                 + 1.12246 w_k1_DIS_s001
                                 + 1.09753 w_k1_FDX_s001
                                 + 1.22378 w_k1_IBM_s001
                                 + 1.05304 w_k1_INTC_s001
                                 + 1.06788 w_k1_KO_s001 + 1.1591 w_k1_MAT_s001
                                 + 0.981749 w_k1_MSFT_s001
                                 + 1.08153 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.879017 w_k1_ADM_s001
                                 + 0.874837 w_k1_AXP_s001
                                 + 1.21436 w_k1_DIS_s001
                                 + 1.01605 w_k1_FDX_s001
                                 + 1.24362 w_k1_IBM_s001
                                 + 0.985962 w_k1_INTC_s001
                                 + 1.25127 w_k1_KO_s001 + 1.27212 w_k1_MAT_s001
                                 + 1.04378 w_k1_MSFT_s001
                                 + 1.04107 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.00549 w_k1_ADM_s001
                                 + 1.04791 w_k1_AXP_s001
                                 + 0.916082 w_k1_DIS_s001
                                 + 1.00055 w_k1_FDX_s001
                                 + 1.21144 w_k1_IBM_s001
                                 + 0.958948 w_k1_INTC_s001
                                 + 1.08023 w_k1_KO_s001
                                 + 0.802655 w_k1_MAT_s001
                                 + 0.959031 w_k1_MSFT_s001
                                 + 0.995267 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.31453 w_k1_ADM_s001
                                 + 1.53328 w_k1_AXP_s001
                                 + 1.74331 w_k1_DIS_s001
                                 + 1.31437 w_k1_FDX_s001
                                 + 1.66646 w_k1_IBM_s001
                                 + 1.24321 w_k1_INTC_s001
                                 + 1.46819 w_k1_KO_s001 + 1.59028 w_k1_MAT_s001
                                 + 1.19743 w_k1_MSFT_s001
                                 + 1.16598 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.12558 w_k1_ADM_s001
                                 + 1.21245 w_k1_AXP_s001 + 1.3748 w_k1_DIS_s001
                                 + 1.31607 w_k1_FDX_s001
                                 + 0.934854 w_k1_IBM_s001
                                 + 1.34277 w_k1_INTC_s001
                                 + 1.33292 w_k1_KO_s001
                                 + 0.828255 w_k1_MAT_s001
                                 + 1.24361 w_k1_MSFT_s001
                                 + 1.21512 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.954583 w_k1_ADM_s001
                                 + 0.766704 w_k1_AXP_s001
                                 + 0.943998 w_k1_DIS_s001
                                 + 0.896287 w_k1_FDX_s001
                                 + 0.940899 w_k1_IBM_s001
                                 + 0.922584 w_k1_INTC_s001
                                 + 1.13316 w_k1_KO_s001
                                 + 0.794526 w_k1_MAT_s001
                                 + 0.899612 w_k1_MSFT_s001
                                 + 1.03025 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.15411 w_k1_ADM_s002
                                 + 1.61323 w_k1_AXP_s002
                                 + 2.03572 w_k1_DIS_s002
                                 + 1.60131 w_k1_FDX_s002
                                 + 1.18192 w_k1_IBM_s002
                                 + 1.29614 w_k1_INTC_s002
                                 + 1.34974 w_k1_KO_s002 + 1.22142 w_k1_MAT_s002
                                 + 1.71827 w_k1_MSFT_s002
                                 + 1.34894 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.04489 w_k1_ADM_s002
                                 + 1.27617 w_k1_AXP_s002
                                 + 1.13283 w_k1_DIS_s002
                                 + 1.13166 w_k1_FDX_s002
                                 + 1.17459 w_k1_IBM_s002
                                 + 1.1251 w_k1_INTC_s002 + 1.02954 w_k1_KO_s002
                                 + 1.42564 w_k1_MAT_s002
                                 + 1.14081 w_k1_MSFT_s002
                                 + 1.16802 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.898778 w_k1_ADM_s002
                                 + 1.15822 w_k1_AXP_s002
                                 + 1.00171 w_k1_DIS_s002
                                 + 1.01763 w_k1_FDX_s002
                                 + 1.63773 w_k1_IBM_s002
                                 + 1.15286 w_k1_INTC_s002
                                 + 1.17357 w_k1_KO_s002 + 1.85387 w_k1_MAT_s002
                                 + 1.10313 w_k1_MSFT_s002
                                 + 1.25236 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.890351 w_k1_ADM_s002
                                 + 0.943668 w_k1_AXP_s002
                                 + 0.602145 w_k1_DIS_s002
                                 + 0.758506 w_k1_FDX_s002
                                 + 1.0876 w_k1_IBM_s002
                                 + 0.779214 w_k1_INTC_s002
                                 + 0.84651 w_k1_KO_s002 + 1.27786 w_k1_MAT_s002
                                 + 0.819945 w_k1_MSFT_s002
                                 + 0.87456 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.990791 w_k1_ADM_s002
                                 + 0.952075 w_k1_AXP_s002
                                 + 0.932328 w_k1_DIS_s002
                                 + 1.03173 w_k1_FDX_s002
                                 + 0.753227 w_k1_IBM_s002
                                 + 1.04268 w_k1_INTC_s002
                                 + 0.896336 w_k1_KO_s002
                                 + 0.733523 w_k1_MAT_s002
                                 + 1.04204 w_k1_MSFT_s002
                                 + 1.02204 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.695959 w_k1_ADM_s002
                                 + 0.757653 w_k1_AXP_s002
                                 + 0.868163 w_k1_DIS_s002
                                 + 0.789735 w_k1_FDX_s002
                                 + 0.914137 w_k1_IBM_s002
                                 + 0.816411 w_k1_INTC_s002
                                 + 0.817481 w_k1_KO_s002
                                 + 1.06371 w_k1_MAT_s002
                                 + 0.922264 w_k1_MSFT_s002
                                 + 0.912316 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.45454 w_k1_ADM_s002
                                 + 1.38879 w_k1_AXP_s002
                                 + 1.40276 w_k1_DIS_s002
                                 + 1.26368 w_k1_FDX_s002 + 1.1251 w_k1_IBM_s002
                                 + 1.14738 w_k1_INTC_s002
                                 + 1.08588 w_k1_KO_s002 + 1.06923 w_k1_MAT_s002
                                 + 1.22581 w_k1_MSFT_s002
                                 + 1.18763 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.915678 w_k1_ADM_s002
                                 + 1.05947 w_k1_AXP_s002
                                 + 0.755906 w_k1_DIS_s002
                                 + 0.892704 w_k1_FDX_s002
                                 + 0.894654 w_k1_IBM_s002
                                 + 0.961607 w_k1_INTC_s002
                                 + 0.776269 w_k1_KO_s002
                                 + 1.02449 w_k1_MAT_s002
                                 + 0.893705 w_k1_MSFT_s002
                                 + 0.952015 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.20556 w_k1_ADM_s003
                                 + 1.38875 w_k1_AXP_s003
                                 + 0.989634 w_k1_DIS_s003
                                 + 1.21831 w_k1_FDX_s003
                                 + 1.07962 w_k1_IBM_s003
                                 + 0.882727 w_k1_INTC_s003
                                 + 0.883044 w_k1_KO_s003
                                 + 1.30344 w_k1_MAT_s003
                                 + 0.872826 w_k1_MSFT_s003
                                 + 1.08867 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.916631 w_k1_ADM_s003
                                 + 0.714971 w_k1_AXP_s003
                                 + 0.761676 w_k1_DIS_s003
                                 + 0.879104 w_k1_FDX_s003
                                 + 0.632047 w_k1_IBM_s003
                                 + 0.799243 w_k1_INTC_s003
                                 + 0.697292 w_k1_KO_s003
                                 + 0.554119 w_k1_MAT_s003
                                 + 0.824003 w_k1_MSFT_s003
                                 + 0.992279 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.04728 w_k1_ADM_s003
                                 + 1.59685 w_k1_AXP_s003
                                 + 1.65055 w_k1_DIS_s003
                                 + 1.33131 w_k1_FDX_s003
                                 + 1.43423 w_k1_IBM_s003
                                 + 1.03904 w_k1_INTC_s003
                                 + 1.28438 w_k1_KO_s003 + 1.67481 w_k1_MAT_s003
                                 + 1.17229 w_k1_MSFT_s003
                                 + 1.35947 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.977943 w_k1_ADM_s003
                                 + 0.833628 w_k1_AXP_s003
                                 + 0.85997 w_k1_DIS_s003
                                 + 0.818111 w_k1_FDX_s003
                                 + 1.15555 w_k1_IBM_s003
                                 + 0.701571 w_k1_INTC_s003
                                 + 0.90498 w_k1_KO_s003 + 1.24132 w_k1_MAT_s003
                                 + 0.826642 w_k1_MSFT_s003
                                 + 0.962754 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.882275 w_k1_ADM_s003
                                 + 1.10582 w_k1_AXP_s003
                                 + 1.40432 w_k1_DIS_s003
                                 + 1.17394 w_k1_FDX_s003
                                 + 0.969298 w_k1_IBM_s003
                                 + 1.15218 w_k1_INTC_s003
                                 + 1.15692 w_k1_KO_s003 + 1.26478 w_k1_MAT_s003
                                 + 1.07386 w_k1_MSFT_s003
                                 + 1.12915 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.25581 w_k1_ADM_s003
                                 + 1.2096 w_k1_AXP_s003 + 1.11044 w_k1_DIS_s003
                                 + 1.26457 w_k1_FDX_s003
                                 + 0.748958 w_k1_IBM_s003
                                 + 0.877422 w_k1_INTC_s003
                                 + 0.747625 w_k1_KO_s003
                                 + 0.785834 w_k1_MAT_s003
                                 + 0.965328 w_k1_MSFT_s003
                                 + 0.970683 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.08175 w_k1_ADM_s003
                                 + 1.02253 w_k1_AXP_s003 + 1.0366 w_k1_DIS_s003
                                 + 0.995991 w_k1_FDX_s003
                                 + 1.15304 w_k1_IBM_s003
                                 + 0.896407 w_k1_INTC_s003
                                 + 1.02255 w_k1_KO_s003 + 1.38499 w_k1_MAT_s003
                                 + 0.990799 w_k1_MSFT_s003
                                 + 0.981084 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.655069 w_k1_ADM_s003
                                 + 0.589777 w_k1_AXP_s003
                                 + 0.54557 w_k1_DIS_s003
                                 + 0.710534 w_k1_FDX_s003
                                 + 0.953818 w_k1_IBM_s003
                                 + 0.686951 w_k1_INTC_s003
                                 + 0.76745 w_k1_KO_s003
                                 + 0.887304 w_k1_MAT_s003
                                 + 0.685152 w_k1_MSFT_s003
                                 + 0.826134 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.844321 w_k1_ADM_s004
                                 + 1.44941 w_k1_AXP_s004
                                 + 1.04818 w_k1_DIS_s004
                                 + 1.09462 w_k1_FDX_s004
                                 + 1.45466 w_k1_IBM_s004
                                 + 1.25761 w_k1_INTC_s004
                                 + 1.29025 w_k1_KO_s004 + 2.387 w_k1_MAT_s004
                                 + 1.21412 w_k1_MSFT_s004
                                 + 1.20276 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.71733 w_k1_ADM_s004
                                 + 0.998607 w_k1_AXP_s004
                                 + 1.1058 w_k1_DIS_s004 + 1.08348 w_k1_FDX_s004
                                 + 0.877478 w_k1_IBM_s004
                                 + 1.16503 w_k1_INTC_s004
                                 + 0.984128 w_k1_KO_s004
                                 + 0.839389 w_k1_MAT_s004
                                 + 1.07579 w_k1_MSFT_s004 + 1.0764 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.653475 w_k1_ADM_s004
                                 + 0.882483 w_k1_AXP_s004
                                 + 0.979137 w_k1_DIS_s004
                                 + 0.920928 w_k1_FDX_s004
                                 + 1.48238 w_k1_IBM_s004
                                 + 1.13145 w_k1_INTC_s004
                                 + 1.23539 w_k1_KO_s004 + 1.13122 w_k1_MAT_s004
                                 + 1.08939 w_k1_MSFT_s004
                                 + 1.02114 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.28519 w_k1_ADM_s004
                                 + 1.543 w_k1_AXP_s004 + 1.80639 w_k1_DIS_s004
                                 + 1.30451 w_k1_FDX_s004
                                 + 1.55989 w_k1_IBM_s004
                                 + 1.28389 w_k1_INTC_s004
                                 + 1.58247 w_k1_KO_s004 + 1.65209 w_k1_MAT_s004
                                 + 1.34291 w_k1_MSFT_s004
                                 + 1.28282 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.06718 w_k1_ADM_s004
                                 + 1.30133 w_k1_AXP_s004
                                 + 1.49062 w_k1_DIS_s004
                                 + 1.27675 w_k1_FDX_s004
                                 + 1.17682 w_k1_IBM_s004
                                 + 1.32856 w_k1_INTC_s004
                                 + 1.26103 w_k1_KO_s004 + 1.27121 w_k1_MAT_s004
                                 + 1.21235 w_k1_MSFT_s004
                                 + 1.09288 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.71636 w_k1_ADM_s004
                                 + 0.762644 w_k1_AXP_s004
                                 + 0.755363 w_k1_DIS_s004
                                 + 0.824355 w_k1_FDX_s004
                                 + 0.758861 w_k1_IBM_s004
                                 + 0.820843 w_k1_INTC_s004
                                 + 0.793109 w_k1_KO_s004
                                 + 0.563985 w_k1_MAT_s004
                                 + 0.797054 w_k1_MSFT_s004
                                 + 0.888612 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.639003 w_k1_ADM_s004
                                 + 0.709186 w_k1_AXP_s004
                                 + 0.649103 w_k1_DIS_s004
                                 + 0.744536 w_k1_FDX_s004
                                 + 1.04377 w_k1_IBM_s004
                                 + 0.986508 w_k1_INTC_s004
                                 + 0.81966 w_k1_KO_s004
                                 + 0.723924 w_k1_MAT_s004
                                 + 0.793051 w_k1_MSFT_s004
                                 + 0.824629 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.791369 w_k1_ADM_s004
                                 + 1.0524 w_k1_AXP_s004
                                 + 0.834183 w_k1_DIS_s004
                                 + 0.879011 w_k1_FDX_s004
                                 + 1.25635 w_k1_IBM_s004
                                 + 1.20598 w_k1_INTC_s004
                                 + 1.00769 w_k1_KO_s004 + 1.27423 w_k1_MAT_s004
                                 + 0.922619 w_k1_MSFT_s004
                                 + 0.986179 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.949733 w_k1_ADM_s005
                                 + 0.875602 w_k1_AXP_s005
                                 + 0.937991 w_k1_DIS_s005
                                 + 1.01422 w_k1_FDX_s005
                                 + 0.714128 w_k1_IBM_s005
                                 + 0.971906 w_k1_INTC_s005
                                 + 0.810639 w_k1_KO_s005
                                 + 0.719899 w_k1_MAT_s005
                                 + 0.924209 w_k1_MSFT_s005
                                 + 0.929825 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.821542 w_k1_ADM_s005
                                 + 1.18351 w_k1_AXP_s005
                                 + 1.20809 w_k1_DIS_s005
                                 + 1.07775 w_k1_FDX_s005
                                 + 1.08054 w_k1_IBM_s005
                                 + 0.893829 w_k1_INTC_s005
                                 + 0.870614 w_k1_KO_s005
                                 + 1.07449 w_k1_MAT_s005
                                 + 1.00949 w_k1_MSFT_s005
                                 + 0.947919 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.773089 w_k1_ADM_s005
                                 + 0.771276 w_k1_AXP_s005
                                 + 0.846138 w_k1_DIS_s005
                                 + 0.713519 w_k1_FDX_s005
                                 + 1.15037 w_k1_IBM_s005
                                 + 0.644965 w_k1_INTC_s005
                                 + 0.896183 w_k1_KO_s005
                                 + 1.22258 w_k1_MAT_s005
                                 + 0.74524 w_k1_MSFT_s005
                                 + 0.867566 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.29469 w_k1_ADM_s005
                                 + 1.40901 w_k1_AXP_s005
                                 + 1.49763 w_k1_DIS_s005
                                 + 1.17894 w_k1_FDX_s005
                                 + 1.53529 w_k1_IBM_s005
                                 + 1.09625 w_k1_INTC_s005
                                 + 1.19863 w_k1_KO_s005 + 1.40351 w_k1_MAT_s005
                                 + 1.12816 w_k1_MSFT_s005 + 1.0937 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.625002 w_k1_ADM_s005
                                 + 0.721061 w_k1_AXP_s005
                                 + 0.69161 w_k1_DIS_s005
                                 + 0.741355 w_k1_FDX_s005
                                 + 0.708697 w_k1_IBM_s005
                                 + 0.691249 w_k1_INTC_s005
                                 + 0.62433 w_k1_KO_s005
                                 + 0.461749 w_k1_MAT_s005
                                 + 0.704695 w_k1_MSFT_s005
                                 + 0.733459 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.858485 w_k1_ADM_s005
                                 + 1.20255 w_k1_AXP_s005
                                 + 0.853255 w_k1_DIS_s005
                                 + 0.996794 w_k1_FDX_s005
                                 + 1.07454 w_k1_IBM_s005
                                 + 0.85474 w_k1_INTC_s005
                                 + 0.954263 w_k1_KO_s005
                                 + 1.26041 w_k1_MAT_s005
                                 + 0.7861 w_k1_MSFT_s005
                                 + 0.900823 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.67588 w_k1_ADM_s005
                                 + 0.6928 w_k1_AXP_s005 + 0.97264 w_k1_DIS_s005
                                 + 0.835346 w_k1_FDX_s005
                                 + 0.863007 w_k1_IBM_s005
                                 + 0.792405 w_k1_INTC_s005
                                 + 0.792581 w_k1_KO_s005
                                 + 0.767887 w_k1_MAT_s005
                                 + 0.84325 w_k1_MSFT_s005
                                 + 0.900933 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.989937 w_k1_ADM_s005
                                 + 1.21796 w_k1_AXP_s005
                                 + 1.70329 w_k1_DIS_s005
                                 + 1.23634 w_k1_FDX_s005
                                 + 1.05719 w_k1_IBM_s005
                                 + 1.05997 w_k1_INTC_s005 + 1.086 w_k1_KO_s005
                                 + 1.1891 w_k1_MAT_s005
                                 + 1.17089 w_k1_MSFT_s005
                                 + 0.920703 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.05055 w_k1_ADM_s006
                                 + 0.908735 w_k1_AXP_s006
                                 + 0.902938 w_k1_DIS_s006
                                 + 0.990849 w_k1_FDX_s006
                                 + 0.787081 w_k1_IBM_s006
                                 + 1.10426 w_k1_INTC_s006
                                 + 0.894292 w_k1_KO_s006
                                 + 0.521501 w_k1_MAT_s006
                                 + 1.09403 w_k1_MSFT_s006
                                 + 0.941748 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.729185 w_k1_ADM_s006
                                 + 0.646724 w_k1_AXP_s006
                                 + 0.637128 w_k1_DIS_s006
                                 + 0.821639 w_k1_FDX_s006
                                 + 0.661491 w_k1_IBM_s006
                                 + 0.814791 w_k1_INTC_s006
                                 + 0.617098 w_k1_KO_s006
                                 + 0.593828 w_k1_MAT_s006
                                 + 0.884877 w_k1_MSFT_s006
                                 + 0.716685 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.838723 w_k1_ADM_s006
                                 + 0.90687 w_k1_AXP_s006
                                 + 0.895048 w_k1_DIS_s006
                                 + 0.944857 w_k1_FDX_s006
                                 + 1.0479 w_k1_IBM_s006
                                 + 0.90028 w_k1_INTC_s006
                                 + 0.90089 w_k1_KO_s006
                                 + 0.884286 w_k1_MAT_s006
                                 + 0.944204 w_k1_MSFT_s006
                                 + 0.866139 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.04425 w_k1_ADM_s006
                                 + 1.34228 w_k1_AXP_s006 + 1.1729 w_k1_DIS_s006
                                 + 1.14896 w_k1_FDX_s006
                                 + 1.59625 w_k1_IBM_s006
                                 + 1.1133 w_k1_INTC_s006 + 1.17052 w_k1_KO_s006
                                 + 1.46032 w_k1_MAT_s006
                                 + 1.04386 w_k1_MSFT_s006
                                 + 0.957156 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.897389 w_k1_ADM_s006
                                 + 0.71099 w_k1_AXP_s006
                                 + 0.724091 w_k1_DIS_s006
                                 + 0.78813 w_k1_FDX_s006
                                 + 0.96103 w_k1_IBM_s006
                                 + 0.811966 w_k1_INTC_s006
                                 + 0.843847 w_k1_KO_s006
                                 + 0.663028 w_k1_MAT_s006
                                 + 0.787998 w_k1_MSFT_s006
                                 + 0.815156 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.731905 w_k1_ADM_s006
                                 + 0.806791 w_k1_AXP_s006
                                 + 0.693511 w_k1_DIS_s006
                                 + 0.799887 w_k1_FDX_s006
                                 + 1.38948 w_k1_IBM_s006
                                 + 0.67237 w_k1_INTC_s006
                                 + 0.953006 w_k1_KO_s006
                                 + 1.35433 w_k1_MAT_s006
                                 + 0.718784 w_k1_MSFT_s006
                                 + 0.951449 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.941537 w_k1_ADM_s006
                                 + 1.23175 w_k1_AXP_s006
                                 + 1.32244 w_k1_DIS_s006
                                 + 1.26324 w_k1_FDX_s006
                                 + 0.940617 w_k1_IBM_s006
                                 + 1.29669 w_k1_INTC_s006
                                 + 1.22008 w_k1_KO_s006 + 1.09299 w_k1_MAT_s006
                                 + 1.34451 w_k1_MSFT_s006
                                 + 1.19555 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.827639 w_k1_ADM_s006
                                 + 1.05853 w_k1_AXP_s006
                                 + 1.07813 w_k1_DIS_s006
                                 + 0.98171 w_k1_FDX_s006
                                 + 1.15668 w_k1_IBM_s006
                                 + 1.08511 w_k1_INTC_s006
                                 + 1.08096 w_k1_KO_s006 + 1.08554 w_k1_MAT_s006
                                 + 0.992538 w_k1_MSFT_s006
                                 + 1.01285 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.12508 w_k1_ADM_s007
                                 + 1.88592 w_k1_AXP_s007
                                 + 1.40478 w_k1_DIS_s007
                                 + 1.32896 w_k1_FDX_s007 + 1.5331 w_k1_IBM_s007
                                 + 1.30592 w_k1_INTC_s007
                                 + 1.21504 w_k1_KO_s007 + 1.95073 w_k1_MAT_s007
                                 + 1.30766 w_k1_MSFT_s007
                                 + 1.28948 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.08644 w_k1_ADM_s007
                                 + 1.01176 w_k1_AXP_s007
                                 + 0.95971 w_k1_DIS_s007
                                 + 1.05131 w_k1_FDX_s007
                                 + 0.799471 w_k1_IBM_s007
                                 + 1.0553 w_k1_INTC_s007
                                 + 0.807793 w_k1_KO_s007
                                 + 0.992354 w_k1_MAT_s007
                                 + 1.09754 w_k1_MSFT_s007
                                 + 0.981004 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.845248 w_k1_ADM_s007
                                 + 1.05678 w_k1_AXP_s007
                                 + 1.33502 w_k1_DIS_s007
                                 + 1.07554 w_k1_FDX_s007 + 1.3289 w_k1_IBM_s007
                                 + 1.2521 w_k1_INTC_s007 + 1.11043 w_k1_KO_s007
                                 + 1.20143 w_k1_MAT_s007
                                 + 1.26216 w_k1_MSFT_s007
                                 + 1.01072 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.866642 w_k1_ADM_s007
                                 + 0.709865 w_k1_AXP_s007
                                 + 0.870791 w_k1_DIS_s007
                                 + 0.819335 w_k1_FDX_s007
                                 + 1.15944 w_k1_IBM_s007
                                 + 0.946753 w_k1_INTC_s007
                                 + 0.95072 w_k1_KO_s007
                                 + 0.764058 w_k1_MAT_s007
                                 + 0.830877 w_k1_MSFT_s007
                                 + 0.911099 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.816961 w_k1_ADM_s007
                                 + 0.698155 w_k1_AXP_s007
                                 + 0.638432 w_k1_DIS_s007
                                 + 0.779042 w_k1_FDX_s007
                                 + 0.738762 w_k1_IBM_s007
                                 + 0.80995 w_k1_INTC_s007
                                 + 0.82571 w_k1_KO_s007
                                 + 0.861086 w_k1_MAT_s007
                                 + 0.758174 w_k1_MSFT_s007
                                 + 0.796682 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.33438 w_k1_ADM_s007
                                 + 1.40418 w_k1_AXP_s007
                                 + 1.49622 w_k1_DIS_s007
                                 + 1.33619 w_k1_FDX_s007
                                 + 1.09575 w_k1_IBM_s007
                                 + 1.32097 w_k1_INTC_s007
                                 + 1.19427 w_k1_KO_s007 + 1.0974 w_k1_MAT_s007
                                 + 1.34028 w_k1_MSFT_s007
                                 + 1.14807 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.06012 w_k1_ADM_s007
                                 + 1.24848 w_k1_AXP_s007
                                 + 1.12762 w_k1_DIS_s007
                                 + 1.21032 w_k1_FDX_s007
                                 + 0.955946 w_k1_IBM_s007
                                 + 1.09455 w_k1_INTC_s007
                                 + 0.882949 w_k1_KO_s007
                                 + 0.83603 w_k1_MAT_s007
                                 + 1.19225 w_k1_MSFT_s007
                                 + 1.01941 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.857916 w_k1_ADM_s007
                                 + 1.12678 w_k1_AXP_s007
                                 + 0.91138 w_k1_DIS_s007
                                 + 0.918531 w_k1_FDX_s007
                                 + 1.26012 w_k1_IBM_s007
                                 + 0.963406 w_k1_INTC_s007
                                 + 1.08037 w_k1_KO_s007 + 1.33468 w_k1_MAT_s007
                                 + 0.857693 w_k1_MSFT_s007
                                 + 1.08508 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.692403 w_k1_ADM_s010
                                 + 0.641563 w_k1_AXP_s010
                                 + 0.59167 w_k1_DIS_s010
                                 + 0.808903 w_k1_FDX_s010
                                 + 0.662275 w_k1_IBM_s010
                                 + 0.886559 w_k1_INTC_s010
                                 + 0.595953 w_k1_KO_s010
                                 + 0.646555 w_k1_MAT_s010
                                 + 0.72077 w_k1_MSFT_s010
                                 + 0.815789 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.833938 w_k1_ADM_s010
                                 + 0.891299 w_k1_AXP_s010
                                 + 0.663716 w_k1_DIS_s010
                                 + 0.796362 w_k1_FDX_s010
                                 + 0.799614 w_k1_IBM_s010
                                 + 1.21159 w_k1_INTC_s010
                                 + 0.838626 w_k1_KO_s010
                                 + 0.887652 w_k1_MAT_s010
                                 + 0.939458 w_k1_MSFT_s010
                                 + 0.840676 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.22703 w_k1_ADM_s010
                                 + 1.12904 w_k1_AXP_s010
                                 + 1.44475 w_k1_DIS_s010
                                 + 1.32797 w_k1_FDX_s010
                                 + 0.99561 w_k1_IBM_s010
                                 + 1.34859 w_k1_INTC_s010
                                 + 1.23968 w_k1_KO_s010
                                 + 0.970479 w_k1_MAT_s010
                                 + 1.2366 w_k1_MSFT_s010 + 1.28637 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.954843 w_k1_ADM_s010
                                 + 0.812661 w_k1_AXP_s010
                                 + 0.710324 w_k1_DIS_s010
                                 + 0.897537 w_k1_FDX_s010
                                 + 1.02075 w_k1_IBM_s010
                                 + 0.96089 w_k1_INTC_s010
                                 + 0.828543 w_k1_KO_s010
                                 + 1.11613 w_k1_MAT_s010
                                 + 0.859114 w_k1_MSFT_s010
                                 + 0.866062 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.691881 w_k1_ADM_s010
                                 + 0.776373 w_k1_AXP_s010
                                 + 0.749741 w_k1_DIS_s010
                                 + 0.817579 w_k1_FDX_s010
                                 + 1.24825 w_k1_IBM_s010
                                 + 1.14977 w_k1_INTC_s010 + 1.0196 w_k1_KO_s010
                                 + 1.28061 w_k1_MAT_s010
                                 + 0.916903 w_k1_MSFT_s010
                                 + 0.90968 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.841616 w_k1_ADM_s010
                                 + 1.09087 w_k1_AXP_s010
                                 + 0.955723 w_k1_DIS_s010
                                 + 1.10612 w_k1_FDX_s010
                                 + 0.892707 w_k1_IBM_s010
                                 + 1.17493 w_k1_INTC_s010
                                 + 1.05565 w_k1_KO_s010 + 1.2018 w_k1_MAT_s010
                                 + 1.11004 w_k1_MSFT_s010
                                 + 0.98749 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.901084 w_k1_ADM_s010
                                 + 0.845593 w_k1_AXP_s010
                                 + 1.0373 w_k1_DIS_s010
                                 + 0.986758 w_k1_FDX_s010
                                 + 1.03453 w_k1_IBM_s010
                                 + 1.14087 w_k1_INTC_s010
                                 + 1.10547 w_k1_KO_s010 + 1.16897 w_k1_MAT_s010
                                 + 1.19689 w_k1_MSFT_s010
                                 + 1.07241 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.0839 w_k1_ADM_s010
                                 + 1.22115 w_k1_AXP_s010
                                 + 1.22352 w_k1_DIS_s010 + 1.0898 w_k1_FDX_s010
                                 + 1.4409 w_k1_IBM_s010
                                 + 1.49436 w_k1_INTC_s010
                                 + 1.33757 w_k1_KO_s010 + 1.41424 w_k1_MAT_s010
                                 + 1.28029 w_k1_MSFT_s010
                                 + 1.12026 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.01324 w_k1_ADM_s011
                                 + 0.734681 w_k1_AXP_s011
                                 + 0.72964 w_k1_DIS_s011
                                 + 0.85432 w_k1_FDX_s011
                                 + 0.715491 w_k1_IBM_s011
                                 + 0.771704 w_k1_INTC_s011
                                 + 0.698215 w_k1_KO_s011
                                 + 0.515875 w_k1_MAT_s011
                                 + 0.712409 w_k1_MSFT_s011
                                 + 0.789006 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.72352 w_k1_ADM_s011
                                 + 1.53806 w_k1_AXP_s011 + 1.9729 w_k1_DIS_s011
                                 + 1.48047 w_k1_FDX_s011
                                 + 1.23433 w_k1_IBM_s011
                                 + 0.898586 w_k1_INTC_s011
                                 + 1.45724 w_k1_KO_s011 + 1.26798 w_k1_MAT_s011
                                 + 1.31682 w_k1_MSFT_s011
                                 + 1.25308 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.22723 w_k1_ADM_s011
                                 + 1.03428 w_k1_AXP_s011
                                 + 1.05873 w_k1_DIS_s011
                                 + 1.10042 w_k1_FDX_s011
                                 + 0.863243 w_k1_IBM_s011
                                 + 0.869866 w_k1_INTC_s011
                                 + 0.975048 w_k1_KO_s011
                                 + 1.13769 w_k1_MAT_s011
                                 + 1.02569 w_k1_MSFT_s011
                                 + 0.83607 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.17069 w_k1_ADM_s011
                                 + 0.782443 w_k1_AXP_s011
                                 + 0.890671 w_k1_DIS_s011
                                 + 0.796827 w_k1_FDX_s011
                                 + 1.15375 w_k1_IBM_s011
                                 + 0.727565 w_k1_INTC_s011
                                 + 1.06139 w_k1_KO_s011 + 1.11519 w_k1_MAT_s011
                                 + 0.853848 w_k1_MSFT_s011
                                 + 0.921532 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.18932 w_k1_ADM_s011
                                 + 1.00489 w_k1_AXP_s011
                                 + 1.10533 w_k1_DIS_s011
                                 + 1.12374 w_k1_FDX_s011 + 0.7176 w_k1_IBM_s011
                                 + 1.17032 w_k1_INTC_s011
                                 + 0.893452 w_k1_KO_s011
                                 + 0.553854 w_k1_MAT_s011
                                 + 1.08716 w_k1_MSFT_s011
                                 + 0.871713 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.44197 w_k1_ADM_s011
                                 + 1.25717 w_k1_AXP_s011
                                 + 1.42565 w_k1_DIS_s011
                                 + 1.23388 w_k1_FDX_s011
                                 + 1.21279 w_k1_IBM_s011
                                 + 0.866622 w_k1_INTC_s011
                                 + 1.21589 w_k1_KO_s011 + 1.32807 w_k1_MAT_s011
                                 + 1.0645 w_k1_MSFT_s011 + 1.08388 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.06849 w_k1_ADM_s011
                                 + 0.858124 w_k1_AXP_s011
                                 + 1.11762 w_k1_DIS_s011
                                 + 0.930215 w_k1_FDX_s011
                                 + 1.03205 w_k1_IBM_s011
                                 + 0.762475 w_k1_INTC_s011
                                 + 1.11114 w_k1_KO_s011
                                 + 0.953828 w_k1_MAT_s011
                                 + 1.03868 w_k1_MSFT_s011
                                 + 0.996458 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.868635 w_k1_ADM_s011
                                 + 0.649138 w_k1_AXP_s011
                                 + 0.760396 w_k1_DIS_s011
                                 + 0.766066 w_k1_FDX_s011
                                 + 0.793126 w_k1_IBM_s011
                                 + 0.82299 w_k1_INTC_s011
                                 + 0.859817 w_k1_KO_s011
                                 + 0.876152 w_k1_MAT_s011
                                 + 0.929898 w_k1_MSFT_s011
                                 + 0.726158 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.844613 w_k1_ADM_s012
                                 + 1.06818 w_k1_AXP_s012
                                 + 0.806283 w_k1_DIS_s012
                                 + 0.858896 w_k1_FDX_s012
                                 + 0.981534 w_k1_IBM_s012
                                 + 0.8712 w_k1_INTC_s012
                                 + 0.982422 w_k1_KO_s012 + 1.0614 w_k1_MAT_s012
                                 + 0.87586 w_k1_MSFT_s012
                                 + 0.967131 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.17634 w_k1_ADM_s012
                                 + 1.34877 w_k1_AXP_s012
                                 + 1.47602 w_k1_DIS_s012
                                 + 1.21412 w_k1_FDX_s012
                                 + 1.52347 w_k1_IBM_s012
                                 + 1.30388 w_k1_INTC_s012
                                 + 1.39588 w_k1_KO_s012 + 1.54811 w_k1_MAT_s012
                                 + 1.27093 w_k1_MSFT_s012
                                 + 1.23437 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.24737 w_k1_ADM_s012
                                 + 1.41194 w_k1_AXP_s012
                                 + 1.45113 w_k1_DIS_s012
                                 + 1.27724 w_k1_FDX_s012
                                 + 1.07314 w_k1_IBM_s012
                                 + 1.06835 w_k1_INTC_s012
                                 + 1.33603 w_k1_KO_s012
                                 + 0.961571 w_k1_MAT_s012
                                 + 1.18364 w_k1_MSFT_s012
                                 + 1.25973 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.39868 w_k1_ADM_s012
                                 + 0.962393 w_k1_AXP_s012
                                 + 1.08743 w_k1_DIS_s012
                                 + 1.06542 w_k1_FDX_s012
                                 + 0.87445 w_k1_IBM_s012
                                 + 0.965778 w_k1_INTC_s012
                                 + 0.861247 w_k1_KO_s012
                                 + 0.818837 w_k1_MAT_s012
                                 + 1.09566 w_k1_MSFT_s012
                                 + 1.05805 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.0346 w_k1_ADM_s012
                                 + 1.08554 w_k1_AXP_s012
                                 + 0.994703 w_k1_DIS_s012
                                 + 0.943367 w_k1_FDX_s012
                                 + 1.31165 w_k1_IBM_s012
                                 + 0.864326 w_k1_INTC_s012
                                 + 1.11692 w_k1_KO_s012 + 1.27722 w_k1_MAT_s012
                                 + 0.948079 w_k1_MSFT_s012
                                 + 0.997455 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.08475 w_k1_ADM_s012
                                 + 0.584702 w_k1_AXP_s012
                                 + 0.692198 w_k1_DIS_s012
                                 + 0.729687 w_k1_FDX_s012
                                 + 0.957365 w_k1_IBM_s012
                                 + 0.760805 w_k1_INTC_s012
                                 + 0.891829 w_k1_KO_s012
                                 + 0.640102 w_k1_MAT_s012
                                 + 0.725252 w_k1_MSFT_s012
                                 + 0.866756 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.794602 w_k1_ADM_s012
                                 + 0.872302 w_k1_AXP_s012
                                 + 0.816243 w_k1_DIS_s012
                                 + 0.898209 w_k1_FDX_s012
                                 + 0.728104 w_k1_IBM_s012
                                 + 0.871967 w_k1_INTC_s012
                                 + 0.811474 w_k1_KO_s012
                                 + 0.649294 w_k1_MAT_s012
                                 + 0.930092 w_k1_MSFT_s012
                                 + 0.792765 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.10059 w_k1_ADM_s012
                                 + 0.912715 w_k1_AXP_s012
                                 + 1.1403 w_k1_DIS_s012
                                 + 0.965938 w_k1_FDX_s012
                                 + 1.13954 w_k1_IBM_s012
                                 + 0.873393 w_k1_INTC_s012
                                 + 1.10809 w_k1_KO_s012 + 1.05258 w_k1_MAT_s012
                                 + 0.999253 w_k1_MSFT_s012
                                 + 1.01532 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.984297 w_k1_ADM_s013
                                 + 1.02053 w_k1_AXP_s013
                                 + 1.32396 w_k1_DIS_s013
                                 + 1.23693 w_k1_FDX_s013
                                 + 0.885109 w_k1_IBM_s013
                                 + 1.0652 w_k1_INTC_s013
                                 + 0.935232 w_k1_KO_s013
                                 + 1.16037 w_k1_MAT_s013
                                 + 1.2279 w_k1_MSFT_s013 + 1.08883 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.1445 w_k1_ADM_s013
                                 + 1.11221 w_k1_AXP_s013
                                 + 1.25039 w_k1_DIS_s013
                                 + 1.08351 w_k1_FDX_s013 + 1.7287 w_k1_IBM_s013
                                 + 0.882193 w_k1_INTC_s013
                                 + 1.12491 w_k1_KO_s013 + 1.78231 w_k1_MAT_s013
                                 + 1.07447 w_k1_MSFT_s013
                                 + 1.10521 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.674968 w_k1_ADM_s013
                                 + 0.886147 w_k1_AXP_s013
                                 + 0.94846 w_k1_DIS_s013
                                 + 1.07595 w_k1_FDX_s013
                                 + 1.00739 w_k1_IBM_s013
                                 + 0.98439 w_k1_INTC_s013
                                 + 1.03659 w_k1_KO_s013 + 1.27438 w_k1_MAT_s013
                                 + 1.20003 w_k1_MSFT_s013 + 1.0794 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.946926 w_k1_ADM_s013
                                 + 0.845264 w_k1_AXP_s013
                                 + 0.766577 w_k1_DIS_s013
                                 + 0.922484 w_k1_FDX_s013
                                 + 0.917577 w_k1_IBM_s013
                                 + 0.737608 w_k1_INTC_s013
                                 + 0.813263 w_k1_KO_s013
                                 + 0.957861 w_k1_MAT_s013
                                 + 0.861073 w_k1_MSFT_s013
                                 + 1.0071 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15043 w_k1_ADM_s013
                                 + 1.22238 w_k1_AXP_s013
                                 + 1.43177 w_k1_DIS_s013
                                 + 1.31993 w_k1_FDX_s013
                                 + 1.19153 w_k1_IBM_s013
                                 + 0.978946 w_k1_INTC_s013
                                 + 1.22794 w_k1_KO_s013 + 1.07398 w_k1_MAT_s013
                                 + 1.17241 w_k1_MSFT_s013 + 1.1054 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.752317 w_k1_ADM_s013
                                 + 0.526674 w_k1_AXP_s013
                                 + 0.583215 w_k1_DIS_s013
                                 + 0.726338 w_k1_FDX_s013
                                 + 0.623296 w_k1_IBM_s013
                                 + 0.675876 w_k1_INTC_s013
                                 + 0.579033 w_k1_KO_s013
                                 + 0.501544 w_k1_MAT_s013
                                 + 0.707382 w_k1_MSFT_s013
                                 + 0.734656 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.857347 w_k1_ADM_s013
                                 + 0.92718 w_k1_AXP_s013
                                 + 0.83503 w_k1_DIS_s013
                                 + 0.854424 w_k1_FDX_s013
                                 + 1.25252 w_k1_IBM_s013
                                 + 0.946868 w_k1_INTC_s013
                                 + 0.982674 w_k1_KO_s013
                                 + 1.12867 w_k1_MAT_s013
                                 + 0.887157 w_k1_MSFT_s013
                                 + 0.952961 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.884503 w_k1_ADM_s013
                                 + 0.810925 w_k1_AXP_s013
                                 + 0.725066 w_k1_DIS_s013
                                 + 0.924594 w_k1_FDX_s013
                                 + 0.740457 w_k1_IBM_s013
                                 + 0.814693 w_k1_INTC_s013
                                 + 0.758866 w_k1_KO_s013
                                 + 0.913639 w_k1_MAT_s013
                                 + 0.87848 w_k1_MSFT_s013 + 1.0191 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.60921 w_k1_ADM_s014
                                 + 1.3987 w_k1_AXP_s014 + 1.61557 w_k1_DIS_s014
                                 + 1.45923 w_k1_FDX_s014
                                 + 1.05709 w_k1_IBM_s014
                                 + 1.15157 w_k1_INTC_s014
                                 + 1.15639 w_k1_KO_s014 + 1.30549 w_k1_MAT_s014
                                 + 1.54369 w_k1_MSFT_s014
                                 + 1.39668 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.07198 w_k1_ADM_s014
                                 + 1.1506 w_k1_AXP_s014
                                 + 0.920065 w_k1_DIS_s014
                                 + 1.05904 w_k1_FDX_s014
                                 + 0.928001 w_k1_IBM_s014
                                 + 0.885061 w_k1_INTC_s014
                                 + 0.855133 w_k1_KO_s014
                                 + 1.01317 w_k1_MAT_s014
                                 + 0.99839 w_k1_MSFT_s014
                                 + 0.952799 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.05389 w_k1_ADM_s014
                                 + 0.793807 w_k1_AXP_s014
                                 + 1.01393 w_k1_DIS_s014
                                 + 0.903739 w_k1_FDX_s014
                                 + 1.18021 w_k1_IBM_s014
                                 + 0.950856 w_k1_INTC_s014
                                 + 1.06821 w_k1_KO_s014 + 1.10472 w_k1_MAT_s014
                                 + 1.07018 w_k1_MSFT_s014
                                 + 0.984573 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.23478 w_k1_ADM_s014
                                 + 1.30808 w_k1_AXP_s014
                                 + 1.39949 w_k1_DIS_s014
                                 + 1.26582 w_k1_FDX_s014 + 1.416 w_k1_IBM_s014
                                 + 1.26897 w_k1_INTC_s014
                                 + 1.26059 w_k1_KO_s014 + 1.51171 w_k1_MAT_s014
                                 + 1.17293 w_k1_MSFT_s014
                                 + 1.16427 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.04257 w_k1_ADM_s014
                                 + 0.787229 w_k1_AXP_s014
                                 + 0.544596 w_k1_DIS_s014
                                 + 0.716596 w_k1_FDX_s014
                                 + 0.97447 w_k1_IBM_s014
                                 + 0.707623 w_k1_INTC_s014
                                 + 0.772901 w_k1_KO_s014
                                 + 0.999074 w_k1_MAT_s014
                                 + 0.725144 w_k1_MSFT_s014
                                 + 0.786499 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.03775 w_k1_ADM_s014
                                 + 0.786286 w_k1_AXP_s014
                                 + 0.928842 w_k1_DIS_s014
                                 + 1.03587 w_k1_FDX_s014
                                 + 0.732547 w_k1_IBM_s014
                                 + 0.969073 w_k1_INTC_s014
                                 + 0.823206 w_k1_KO_s014
                                 + 0.808459 w_k1_MAT_s014
                                 + 1.01023 w_k1_MSFT_s014
                                 + 0.992201 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.832518 w_k1_ADM_s014
                                 + 0.719233 w_k1_AXP_s014
                                 + 0.641199 w_k1_DIS_s014
                                 + 0.763421 w_k1_FDX_s014
                                 + 0.694074 w_k1_IBM_s014
                                 + 0.840399 w_k1_INTC_s014
                                 + 0.704851 w_k1_KO_s014
                                 + 0.594485 w_k1_MAT_s014
                                 + 0.816407 w_k1_MSFT_s014
                                 + 0.837612 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.30487 w_k1_ADM_s014
                                 + 1.42684 w_k1_AXP_s014 + 1.1039 w_k1_DIS_s014
                                 + 1.15259 w_k1_FDX_s014
                                 + 1.25855 w_k1_IBM_s014
                                 + 1.03425 w_k1_INTC_s014
                                 + 1.06364 w_k1_KO_s014 + 1.34033 w_k1_MAT_s014
                                 + 1.05657 w_k1_MSFT_s014
                                 + 1.17727 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.18785 w_k1_ADM_s015
                                 + 0.999982 w_k1_AXP_s015
                                 + 0.928411 w_k1_DIS_s015
                                 + 1.06446 w_k1_FDX_s015
                                 + 0.609355 w_k1_IBM_s015
                                 + 0.976729 w_k1_INTC_s015
                                 + 0.718804 w_k1_KO_s015
                                 + 0.684201 w_k1_MAT_s015
                                 + 0.968704 w_k1_MSFT_s015
                                 + 0.984185 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.47762 w_k1_ADM_s015
                                 + 1.47977 w_k1_AXP_s015
                                 + 2.08947 w_k1_DIS_s015
                                 + 1.41735 w_k1_FDX_s015 + 1.2307 w_k1_IBM_s015
                                 + 1.02457 w_k1_INTC_s015
                                 + 1.45508 w_k1_KO_s015 + 1.67475 w_k1_MAT_s015
                                 + 1.45039 w_k1_MSFT_s015
                                 + 1.33809 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.86073 w_k1_ADM_s015
                                 + 0.840892 w_k1_AXP_s015
                                 + 0.838804 w_k1_DIS_s015
                                 + 0.905832 w_k1_FDX_s015
                                 + 0.782053 w_k1_IBM_s015
                                 + 1.02018 w_k1_INTC_s015
                                 + 0.973332 w_k1_KO_s015
                                 + 0.614863 w_k1_MAT_s015
                                 + 0.986905 w_k1_MSFT_s015
                                 + 0.979917 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.954097 w_k1_ADM_s015
                                 + 0.945324 w_k1_AXP_s015
                                 + 0.763844 w_k1_DIS_s015
                                 + 0.727048 w_k1_FDX_s015
                                 + 0.933455 w_k1_IBM_s015
                                 + 0.782217 w_k1_INTC_s015
                                 + 0.856947 w_k1_KO_s015
                                 + 0.801161 w_k1_MAT_s015
                                 + 0.720584 w_k1_MSFT_s015
                                 + 0.999855 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.992963 w_k1_ADM_s015
                                 + 0.924511 w_k1_AXP_s015
                                 + 1.03841 w_k1_DIS_s015
                                 + 0.916648 w_k1_FDX_s015
                                 + 1.19881 w_k1_IBM_s015
                                 + 0.861466 w_k1_INTC_s015
                                 + 1.13127 w_k1_KO_s015 + 1.01539 w_k1_MAT_s015
                                 + 0.991782 w_k1_MSFT_s015
                                 + 1.1271 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.32754 w_k1_ADM_s015
                                 + 1.35639 w_k1_AXP_s015
                                 + 1.08787 w_k1_DIS_s015
                                 + 1.17151 w_k1_FDX_s015
                                 + 0.757978 w_k1_IBM_s015
                                 + 0.901403 w_k1_INTC_s015
                                 + 0.889682 w_k1_KO_s015
                                 + 0.745007 w_k1_MAT_s015
                                 + 0.925117 w_k1_MSFT_s015
                                 + 0.968474 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.05287 w_k1_ADM_s015
                                 + 1.4096 w_k1_AXP_s015 + 1.30349 w_k1_DIS_s015
                                 + 1.09095 w_k1_FDX_s015
                                 + 1.07463 w_k1_IBM_s015
                                 + 1.12885 w_k1_INTC_s015
                                 + 1.20718 w_k1_KO_s015 + 1.13144 w_k1_MAT_s015
                                 + 1.12473 w_k1_MSFT_s015
                                 + 1.09856 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.03565 w_k1_ADM_s015
                                 + 1.10323 w_k1_AXP_s015
                                 + 0.84203 w_k1_DIS_s015
                                 + 0.814374 w_k1_FDX_s015
                                 + 1.54211 w_k1_IBM_s015
                                 + 0.805429 w_k1_INTC_s015
                                 + 1.02562 w_k1_KO_s015 + 1.12917 w_k1_MAT_s015
                                 + 0.873904 w_k1_MSFT_s015
                                 + 1.02242 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.28124 w_k1_ADM_s016
                                 + 1.00305 w_k1_AXP_s016
                                 + 1.00295 w_k1_DIS_s016
                                 + 1.04638 w_k1_FDX_s016
                                 + 0.746007 w_k1_IBM_s016
                                 + 0.778545 w_k1_INTC_s016
                                 + 0.845928 w_k1_KO_s016
                                 + 0.549107 w_k1_MAT_s016
                                 + 1.04085 w_k1_MSFT_s016
                                 + 1.11627 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.59523 w_k1_ADM_s016
                                 + 1.27352 w_k1_AXP_s016
                                 + 0.913869 w_k1_DIS_s016
                                 + 1.00078 w_k1_FDX_s016
                                 + 0.885323 w_k1_IBM_s016
                                 + 0.760655 w_k1_INTC_s016
                                 + 0.823573 w_k1_KO_s016
                                 + 0.746316 w_k1_MAT_s016
                                 + 0.940661 w_k1_MSFT_s016
                                 + 0.986909 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.8215 w_k1_ADM_s016
                                 + 1.04596 w_k1_AXP_s016
                                 + 0.788446 w_k1_DIS_s016
                                 + 0.828141 w_k1_FDX_s016
                                 + 1.15999 w_k1_IBM_s016
                                 + 0.961518 w_k1_INTC_s016
                                 + 0.778318 w_k1_KO_s016
                                 + 0.964636 w_k1_MAT_s016
                                 + 0.863222 w_k1_MSFT_s016
                                 + 0.893641 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.36449 w_k1_ADM_s016
                                 + 2.18808 w_k1_AXP_s016
                                 + 1.96543 w_k1_DIS_s016 + 1.7037 w_k1_FDX_s016
                                 + 1.26034 w_k1_IBM_s016
                                 + 1.17871 w_k1_INTC_s016
                                 + 1.19725 w_k1_KO_s016 + 1.16097 w_k1_MAT_s016
                                 + 1.29783 w_k1_MSFT_s016
                                 + 1.44167 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.32274 w_k1_ADM_s016
                                 + 1.30143 w_k1_AXP_s016
                                 + 1.32298 w_k1_DIS_s016
                                 + 1.42614 w_k1_FDX_s016
                                 + 0.942858 w_k1_IBM_s016
                                 + 1.09803 w_k1_INTC_s016
                                 + 0.960768 w_k1_KO_s016
                                 + 0.969163 w_k1_MAT_s016
                                 + 1.29546 w_k1_MSFT_s016
                                 + 1.27673 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.14023 w_k1_ADM_s016
                                 + 1.31185 w_k1_AXP_s016
                                 + 1.15601 w_k1_DIS_s016
                                 + 1.15212 w_k1_FDX_s016
                                 + 1.69951 w_k1_IBM_s016
                                 + 1.03544 w_k1_INTC_s016
                                 + 1.20531 w_k1_KO_s016 + 1.85594 w_k1_MAT_s016
                                 + 1.07414 w_k1_MSFT_s016
                                 + 1.22545 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.943019 w_k1_ADM_s016
                                 + 0.8546 w_k1_AXP_s016 + 0.57538 w_k1_DIS_s016
                                 + 0.758873 w_k1_FDX_s016
                                 + 0.632557 w_k1_IBM_s016
                                 + 0.704561 w_k1_INTC_s016
                                 + 0.58957 w_k1_KO_s016
                                 + 0.465012 w_k1_MAT_s016
                                 + 0.624812 w_k1_MSFT_s016
                                 + 0.915451 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.988339 w_k1_ADM_s016
                                 + 0.740458 w_k1_AXP_s016
                                 + 0.650519 w_k1_DIS_s016
                                 + 0.719661 w_k1_FDX_s016
                                 + 0.932484 w_k1_IBM_s016
                                 + 0.701804 w_k1_INTC_s016
                                 + 0.761137 w_k1_KO_s016
                                 + 0.749574 w_k1_MAT_s016
                                 + 0.795897 w_k1_MSFT_s016
                                 + 0.969171 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.06918 w_k1_ADM_s017
                                 + 0.814859 w_k1_AXP_s017
                                 + 0.862823 w_k1_DIS_s017
                                 + 0.817588 w_k1_FDX_s017
                                 + 1.14306 w_k1_IBM_s017
                                 + 0.840418 w_k1_INTC_s017
                                 + 1.08217 w_k1_KO_s017 + 1.09539 w_k1_MAT_s017
                                 + 0.773884 w_k1_MSFT_s017
                                 + 0.880613 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.32427 w_k1_ADM_s017
                                 + 0.989943 w_k1_AXP_s017
                                 + 1.00798 w_k1_DIS_s017
                                 + 0.962656 w_k1_FDX_s017
                                 + 1.01396 w_k1_IBM_s017
                                 + 0.935037 w_k1_INTC_s017
                                 + 1.25884 w_k1_KO_s017 + 1.18561 w_k1_MAT_s017
                                 + 1.01244 w_k1_MSFT_s017
                                 + 1.05255 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.25391 w_k1_ADM_s017
                                 + 1.02931 w_k1_AXP_s017 + 1.0032 w_k1_DIS_s017
                                 + 1.07134 w_k1_FDX_s017
                                 + 0.80163 w_k1_IBM_s017
                                 + 1.10143 w_k1_INTC_s017
                                 + 1.07305 w_k1_KO_s017 + 1.01782 w_k1_MAT_s017
                                 + 1.02797 w_k1_MSFT_s017
                                 + 1.08201 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.33926 w_k1_ADM_s017
                                 + 1.39972 w_k1_AXP_s017
                                 + 1.10909 w_k1_DIS_s017
                                 + 1.06551 w_k1_FDX_s017
                                 + 1.10178 w_k1_IBM_s017
                                 + 1.17439 w_k1_INTC_s017
                                 + 1.24403 w_k1_KO_s017 + 1.41742 w_k1_MAT_s017
                                 + 0.988106 w_k1_MSFT_s017
                                 + 1.08431 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.966327 w_k1_ADM_s017
                                 + 0.69504 w_k1_AXP_s017
                                 + 0.619599 w_k1_DIS_s017
                                 + 0.797175 w_k1_FDX_s017
                                 + 0.681846 w_k1_IBM_s017
                                 + 0.816641 w_k1_INTC_s017
                                 + 0.775158 w_k1_KO_s017
                                 + 0.658936 w_k1_MAT_s017
                                 + 0.798825 w_k1_MSFT_s017
                                 + 0.879346 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.73067 w_k1_ADM_s017
                                 + 1.42063 w_k1_AXP_s017
                                 + 1.62885 w_k1_DIS_s017
                                 + 1.42314 w_k1_FDX_s017
                                 + 1.28911 w_k1_IBM_s017
                                 + 1.12658 w_k1_INTC_s017
                                 + 1.32382 w_k1_KO_s017 + 1.42081 w_k1_MAT_s017
                                 + 1.29005 w_k1_MSFT_s017
                                 + 1.36385 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.31313 w_k1_ADM_s017
                                 + 1.16903 w_k1_AXP_s017
                                 + 1.13575 w_k1_DIS_s017
                                 + 1.02405 w_k1_FDX_s017
                                 + 1.52134 w_k1_IBM_s017
                                 + 0.895333 w_k1_INTC_s017
                                 + 1.21206 w_k1_KO_s017 + 1.36276 w_k1_MAT_s017
                                 + 1.02183 w_k1_MSFT_s017
                                 + 1.09491 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.27534 w_k1_ADM_s017
                                 + 1.49992 w_k1_AXP_s017
                                 + 1.43944 w_k1_DIS_s017
                                 + 1.23259 w_k1_FDX_s017
                                 + 0.906463 w_k1_IBM_s017
                                 + 1.19782 w_k1_INTC_s017
                                 + 1.25969 w_k1_KO_s017 + 1.62024 w_k1_MAT_s017
                                 + 1.25769 w_k1_MSFT_s017
                                 + 1.17024 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.32199 w_k1_ADM_s100
                                 + 0.91735 w_k1_AXP_s100
                                 + 1.28319 w_k1_DIS_s100
                                 + 1.04657 w_k1_FDX_s100
                                 + 0.723691 w_k1_IBM_s100
                                 + 1.02993 w_k1_INTC_s100
                                 + 1.00609 w_k1_KO_s100
                                 + 0.711657 w_k1_MAT_s100
                                 + 1.16206 w_k1_MSFT_s100
                                 + 1.01404 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.23696 w_k1_ADM_s100
                                 + 0.946571 w_k1_AXP_s100
                                 + 0.950883 w_k1_DIS_s100
                                 + 1.07535 w_k1_FDX_s100
                                 + 0.776292 w_k1_IBM_s100
                                 + 1.02505 w_k1_INTC_s100
                                 + 0.941624 w_k1_KO_s100
                                 + 0.818265 w_k1_MAT_s100
                                 + 0.961609 w_k1_MSFT_s100
                                 + 0.976804 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.09109 w_k1_ADM_s100
                                 + 0.867847 w_k1_AXP_s100
                                 + 1.0974 w_k1_DIS_s100
                                 + 0.879372 w_k1_FDX_s100
                                 + 1.06012 w_k1_IBM_s100
                                 + 1.16428 w_k1_INTC_s100
                                 + 1.03641 w_k1_KO_s100
                                 + 0.900051 w_k1_MAT_s100
                                 + 0.975424 w_k1_MSFT_s100
                                 + 0.970176 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.55021 w_k1_ADM_s100
                                 + 1.62229 w_k1_AXP_s100
                                 + 1.91621 w_k1_DIS_s100
                                 + 1.45748 w_k1_FDX_s100
                                 + 1.34566 w_k1_IBM_s100
                                 + 1.42927 w_k1_INTC_s100
                                 + 1.59756 w_k1_KO_s100 + 1.1399 w_k1_MAT_s100
                                 + 1.5895 w_k1_MSFT_s100 + 1.34434 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.30649 w_k1_ADM_s100
                                 + 1.37186 w_k1_AXP_s100
                                 + 1.34909 w_k1_DIS_s100
                                 + 1.31951 w_k1_FDX_s100
                                 + 0.786239 w_k1_IBM_s100
                                 + 1.11825 w_k1_INTC_s100
                                 + 1.03992 w_k1_KO_s100 + 1.04927 w_k1_MAT_s100
                                 + 1.28968 w_k1_MSFT_s100
                                 + 1.10392 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.1009 w_k1_ADM_s100
                                 + 1.29879 w_k1_AXP_s100
                                 + 1.26173 w_k1_DIS_s100
                                 + 1.05785 w_k1_FDX_s100 + 1.6061 w_k1_IBM_s100
                                 + 1.14949 w_k1_INTC_s100
                                 + 1.32756 w_k1_KO_s100 + 1.90796 w_k1_MAT_s100
                                 + 1.15251 w_k1_MSFT_s100
                                 + 1.19079 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.846418 w_k1_ADM_s100
                                 + 0.899034 w_k1_AXP_s100
                                 + 0.746878 w_k1_DIS_s100
                                 + 0.806151 w_k1_FDX_s100
                                 + 0.793678 w_k1_IBM_s100
                                 + 0.99849 w_k1_INTC_s100
                                 + 0.970129 w_k1_KO_s100
                                 + 0.807972 w_k1_MAT_s100
                                 + 0.837824 w_k1_MSFT_s100
                                 + 0.919319 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.952047 w_k1_ADM_s100
                                 + 0.65435 w_k1_AXP_s100
                                 + 0.672757 w_k1_DIS_s100
                                 + 0.730248 w_k1_FDX_s100
                                 + 0.665479 w_k1_IBM_s100
                                 + 0.791099 w_k1_INTC_s100
                                 + 0.791166 w_k1_KO_s100
                                 + 0.555405 w_k1_MAT_s100
                                 + 0.821472 w_k1_MSFT_s100
                                 + 0.752819 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.09132 w_k1_ADM_s101
                                 + 1.08872 w_k1_AXP_s101
                                 + 1.38923 w_k1_DIS_s101 + 1.1315 w_k1_FDX_s101
                                 + 1.00171 w_k1_IBM_s101
                                 + 1.09861 w_k1_INTC_s101
                                 + 1.16572 w_k1_KO_s101 + 1.01867 w_k1_MAT_s101
                                 + 1.16425 w_k1_MSFT_s101
                                 + 1.06378 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.838669 w_k1_ADM_s101
                                 + 0.642715 w_k1_AXP_s101
                                 + 0.625014 w_k1_DIS_s101
                                 + 0.817954 w_k1_FDX_s101
                                 + 0.683684 w_k1_IBM_s101
                                 + 0.933858 w_k1_INTC_s101
                                 + 0.714448 w_k1_KO_s101
                                 + 0.645131 w_k1_MAT_s101
                                 + 0.74943 w_k1_MSFT_s101
                                 + 0.851501 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.905698 w_k1_ADM_s101
                                 + 0.768016 w_k1_AXP_s101
                                 + 0.866749 w_k1_DIS_s101
                                 + 0.806098 w_k1_FDX_s101
                                 + 0.893908 w_k1_IBM_s101
                                 + 0.933504 w_k1_INTC_s101
                                 + 0.82561 w_k1_KO_s101
                                 + 0.576427 w_k1_MAT_s101
                                 + 0.829636 w_k1_MSFT_s101
                                 + 0.905443 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.08347 w_k1_ADM_s101
                                 + 0.906498 w_k1_AXP_s101
                                 + 0.950665 w_k1_DIS_s101
                                 + 1.07769 w_k1_FDX_s101
                                 + 0.664817 w_k1_IBM_s101
                                 + 1.04754 w_k1_INTC_s101
                                 + 0.831781 w_k1_KO_s101
                                 + 0.532544 w_k1_MAT_s101
                                 + 1.08514 w_k1_MSFT_s101
                                 + 0.898975 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.885565 w_k1_ADM_s101
                                 + 0.734067 w_k1_AXP_s101
                                 + 1.00058 w_k1_DIS_s101
                                 + 0.818047 w_k1_FDX_s101
                                 + 1.29557 w_k1_IBM_s101
                                 + 1.02596 w_k1_INTC_s101
                                 + 1.02171 w_k1_KO_s101 + 0.90639 w_k1_MAT_s101
                                 + 0.812795 w_k1_MSFT_s101
                                 + 0.867241 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.11472 w_k1_ADM_s101
                                 + 1.36955 w_k1_AXP_s101
                                 + 1.10132 w_k1_DIS_s101
                                 + 1.11346 w_k1_FDX_s101 + 1.2854 w_k1_IBM_s101
                                 + 1.18094 w_k1_INTC_s101
                                 + 1.04662 w_k1_KO_s101 + 1.42024 w_k1_MAT_s101
                                 + 1.02499 w_k1_MSFT_s101
                                 + 1.07748 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.07583 w_k1_ADM_s101
                                 + 1.05715 w_k1_AXP_s101
                                 + 1.26174 w_k1_DIS_s101
                                 + 0.986842 w_k1_FDX_s101
                                 + 1.51383 w_k1_IBM_s101
                                 + 1.24978 w_k1_INTC_s101
                                 + 1.35491 w_k1_KO_s101 + 1.82775 w_k1_MAT_s101
                                 + 1.06518 w_k1_MSFT_s101
                                 + 1.11232 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.53041 w_k1_ADM_s101
                                 + 1.61554 w_k1_AXP_s101
                                 + 1.69889 w_k1_DIS_s101
                                 + 1.50803 w_k1_FDX_s101
                                 + 1.28444 w_k1_IBM_s101
                                 + 1.1611 w_k1_INTC_s101 + 1.12095 w_k1_KO_s101
                                 + 1.6034 w_k1_MAT_s101
                                 + 1.24839 w_k1_MSFT_s101
                                 + 1.15399 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.08607 w_k1_ADM_s102
                                 + 0.887578 w_k1_AXP_s102
                                 + 1.04965 w_k1_DIS_s102
                                 + 1.01559 w_k1_FDX_s102
                                 + 1.05135 w_k1_IBM_s102
                                 + 0.779421 w_k1_INTC_s102
                                 + 1.02009 w_k1_KO_s102 + 1.21039 w_k1_MAT_s102
                                 + 0.933932 w_k1_MSFT_s102
                                 + 1.05217 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.877803 w_k1_ADM_s102
                                 + 0.714212 w_k1_AXP_s102
                                 + 0.732751 w_k1_DIS_s102
                                 + 0.791472 w_k1_FDX_s102
                                 + 1.21444 w_k1_IBM_s102
                                 + 0.901591 w_k1_INTC_s102
                                 + 0.968159 w_k1_KO_s102
                                 + 0.868558 w_k1_MAT_s102
                                 + 0.894681 w_k1_MSFT_s102
                                 + 0.989966 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.10978 w_k1_ADM_s102
                                 + 1.39669 w_k1_AXP_s102
                                 + 1.45044 w_k1_DIS_s102
                                 + 1.29195 w_k1_FDX_s102
                                 + 1.45195 w_k1_IBM_s102
                                 + 1.11287 w_k1_INTC_s102
                                 + 1.18383 w_k1_KO_s102 + 1.29077 w_k1_MAT_s102
                                 + 1.10534 w_k1_MSFT_s102
                                 + 1.19757 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.744031 w_k1_ADM_s102
                                 + 0.627201 w_k1_AXP_s102
                                 + 0.622627 w_k1_DIS_s102
                                 + 0.716813 w_k1_FDX_s102
                                 + 0.948677 w_k1_IBM_s102
                                 + 0.684793 w_k1_INTC_s102
                                 + 0.683177 w_k1_KO_s102
                                 + 0.607452 w_k1_MAT_s102
                                 + 0.711443 w_k1_MSFT_s102
                                 + 0.891473 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.857776 w_k1_ADM_s102
                                 + 1.12747 w_k1_AXP_s102
                                 + 1.09453 w_k1_DIS_s102
                                 + 0.955951 w_k1_FDX_s102
                                 + 1.41955 w_k1_IBM_s102
                                 + 0.985275 w_k1_INTC_s102
                                 + 1.16438 w_k1_KO_s102 + 1.31025 w_k1_MAT_s102
                                 + 0.98425 w_k1_MSFT_s102
                                 + 1.09948 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.829728 w_k1_ADM_s102
                                 + 0.73817 w_k1_AXP_s102
                                 + 0.595571 w_k1_DIS_s102
                                 + 0.799582 w_k1_FDX_s102
                                 + 0.723923 w_k1_IBM_s102
                                 + 0.765999 w_k1_INTC_s102
                                 + 0.718315 w_k1_KO_s102
                                 + 0.614595 w_k1_MAT_s102
                                 + 0.827643 w_k1_MSFT_s102
                                 + 0.812173 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.27312 w_k1_ADM_s102
                                 + 1.06867 w_k1_AXP_s102
                                 + 0.949066 w_k1_DIS_s102
                                 + 1.0167 w_k1_FDX_s102
                                 + 0.770698 w_k1_IBM_s102
                                 + 1.04509 w_k1_INTC_s102
                                 + 0.907587 w_k1_KO_s102
                                 + 1.01457 w_k1_MAT_s102
                                 + 0.973151 w_k1_MSFT_s102
                                 + 1.09948 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.5013 w_k1_ADM_s102
                                 + 1.38295 w_k1_AXP_s102
                                 + 1.46715 w_k1_DIS_s102
                                 + 1.37871 w_k1_FDX_s102
                                 + 0.947317 w_k1_IBM_s102
                                 + 1.10778 w_k1_INTC_s102
                                 + 1.08976 w_k1_KO_s102 + 1.19726 w_k1_MAT_s102
                                 + 1.32659 w_k1_MSFT_s102
                                 + 1.24523 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.16106 w_k1_ADM_s103
                                 + 1.43104 w_k1_AXP_s103
                                 + 1.65185 w_k1_DIS_s103
                                 + 1.44169 w_k1_FDX_s103
                                 + 1.34732 w_k1_IBM_s103
                                 + 1.21456 w_k1_INTC_s103
                                 + 1.49263 w_k1_KO_s103 + 1.30837 w_k1_MAT_s103
                                 + 1.17078 w_k1_MSFT_s103
                                 + 1.30027 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.14248 w_k1_ADM_s103
                                 + 1.14949 w_k1_AXP_s103
                                 + 1.23922 w_k1_DIS_s103
                                 + 1.21031 w_k1_FDX_s103
                                 + 1.02311 w_k1_IBM_s103
                                 + 1.21916 w_k1_INTC_s103
                                 + 1.05325 w_k1_KO_s103
                                 + 0.825576 w_k1_MAT_s103
                                 + 1.12883 w_k1_MSFT_s103
                                 + 1.06627 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.741265 w_k1_ADM_s103
                                 + 0.602623 w_k1_AXP_s103
                                 + 0.907869 w_k1_DIS_s103
                                 + 0.798365 w_k1_FDX_s103
                                 + 0.794036 w_k1_IBM_s103
                                 + 0.947813 w_k1_INTC_s103
                                 + 0.97535 w_k1_KO_s103
                                 + 0.513668 w_k1_MAT_s103
                                 + 0.896412 w_k1_MSFT_s103
                                 + 0.846853 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.969172 w_k1_ADM_s103
                                 + 0.938652 w_k1_AXP_s103
                                 + 1.06777 w_k1_DIS_s103
                                 + 1.04176 w_k1_FDX_s103
                                 + 1.07299 w_k1_IBM_s103
                                 + 0.997121 w_k1_INTC_s103
                                 + 1.09419 w_k1_KO_s103 + 1.09545 w_k1_MAT_s103
                                 + 1.00733 w_k1_MSFT_s103
                                 + 0.944302 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.906694 w_k1_ADM_s103
                                 + 0.918311 w_k1_AXP_s103
                                 + 1.1082 w_k1_DIS_s103
                                 + 0.997755 w_k1_FDX_s103
                                 + 1.53314 w_k1_IBM_s103
                                 + 1.19498 w_k1_INTC_s103
                                 + 1.37606 w_k1_KO_s103 + 1.44273 w_k1_MAT_s103
                                 + 1.03892 w_k1_MSFT_s103
                                 + 1.16313 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.955059 w_k1_ADM_s103
                                 + 0.881172 w_k1_AXP_s103
                                 + 0.962697 w_k1_DIS_s103
                                 + 0.985944 w_k1_FDX_s103
                                 + 0.74288 w_k1_IBM_s103
                                 + 0.86245 w_k1_INTC_s103
                                 + 0.922291 w_k1_KO_s103
                                 + 0.936601 w_k1_MAT_s103
                                 + 0.842246 w_k1_MSFT_s103
                                 + 0.896297 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.832203 w_k1_ADM_s103
                                 + 0.755676 w_k1_AXP_s103
                                 + 0.716915 w_k1_DIS_s103
                                 + 0.841022 w_k1_FDX_s103
                                 + 1.02888 w_k1_IBM_s103
                                 + 0.773997 w_k1_INTC_s103
                                 + 0.919122 w_k1_KO_s103
                                 + 0.906418 w_k1_MAT_s103
                                 + 0.81488 w_k1_MSFT_s103
                                 + 0.916327 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.750668 w_k1_ADM_s103
                                 + 0.632346 w_k1_AXP_s103
                                 + 0.625793 w_k1_DIS_s103
                                 + 0.763103 w_k1_FDX_s103
                                 + 0.695752 w_k1_IBM_s103
                                 + 0.913406 w_k1_INTC_s103
                                 + 0.769039 w_k1_KO_s103
                                 + 0.462842 w_k1_MAT_s103
                                 + 0.657291 w_k1_MSFT_s103
                                 + 0.822952 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.01028 w_k1_ADM_s104
                                 + 1.01759 w_k1_AXP_s104
                                 + 1.35008 w_k1_DIS_s104
                                 + 1.13045 w_k1_FDX_s104
                                 + 1.79995 w_k1_IBM_s104
                                 + 1.38329 w_k1_INTC_s104
                                 + 1.51537 w_k1_KO_s104 + 1.53248 w_k1_MAT_s104
                                 + 1.17244 w_k1_MSFT_s104
                                 + 1.21301 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.857001 w_k1_ADM_s104
                                 + 1.14963 w_k1_AXP_s104
                                 + 1.08711 w_k1_DIS_s104
                                 + 1.13901 w_k1_FDX_s104
                                 + 1.11073 w_k1_IBM_s104
                                 + 1.08957 w_k1_INTC_s104
                                 + 1.08069 w_k1_KO_s104 + 1.2246 w_k1_MAT_s104
                                 + 0.937644 w_k1_MSFT_s104
                                 + 1.01097 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.43067 w_k1_ADM_s104
                                 + 1.28429 w_k1_AXP_s104
                                 + 1.72327 w_k1_DIS_s104
                                 + 1.44892 w_k1_FDX_s104
                                 + 1.29311 w_k1_IBM_s104
                                 + 1.24957 w_k1_INTC_s104 + 1.6642 w_k1_KO_s104
                                 + 1.40396 w_k1_MAT_s104
                                 + 1.19933 w_k1_MSFT_s104
                                 + 1.41026 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.718933 w_k1_ADM_s104
                                 + 0.82189 w_k1_AXP_s104
                                 + 0.825487 w_k1_DIS_s104
                                 + 0.914534 w_k1_FDX_s104
                                 + 0.750194 w_k1_IBM_s104
                                 + 0.904985 w_k1_INTC_s104
                                 + 1.00113 w_k1_KO_s104
                                 + 0.794271 w_k1_MAT_s104
                                 + 0.880827 w_k1_MSFT_s104
                                 + 1.00515 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.31748 w_k1_ADM_s104
                                 + 1.02616 w_k1_AXP_s104
                                 + 1.22884 w_k1_DIS_s104
                                 + 1.30455 w_k1_FDX_s104
                                 + 0.908423 w_k1_IBM_s104
                                 + 1.46166 w_k1_INTC_s104
                                 + 1.30614 w_k1_KO_s104
                                 + 0.825003 w_k1_MAT_s104
                                 + 1.18319 w_k1_MSFT_s104
                                 + 1.25108 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.623152 w_k1_ADM_s104
                                 + 0.879925 w_k1_AXP_s104
                                 + 0.906258 w_k1_DIS_s104
                                 + 0.860121 w_k1_FDX_s104
                                 + 1.29004 w_k1_IBM_s104
                                 + 1.08725 w_k1_INTC_s104
                                 + 1.24599 w_k1_KO_s104 + 1.46341 w_k1_MAT_s104
                                 + 0.949941 w_k1_MSFT_s104
                                 + 1.08875 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.85415 w_k1_ADM_s104
                                 + 0.920258 w_k1_AXP_s104
                                 + 0.935563 w_k1_DIS_s104
                                 + 1.05043 w_k1_FDX_s104
                                 + 1.12843 w_k1_IBM_s104
                                 + 1.17183 w_k1_INTC_s104
                                 + 1.18457 w_k1_KO_s104 + 1.00246 w_k1_MAT_s104
                                 + 1.07148 w_k1_MSFT_s104
                                 + 1.09815 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.729597 w_k1_ADM_s104
                                 + 0.740331 w_k1_AXP_s104
                                 + 0.819956 w_k1_DIS_s104
                                 + 0.867596 w_k1_FDX_s104
                                 + 1.01225 w_k1_IBM_s104
                                 + 0.978071 w_k1_INTC_s104
                                 + 0.97471 w_k1_KO_s104 + 1.29045 w_k1_MAT_s104
                                 + 0.948313 w_k1_MSFT_s104
                                 + 1.03908 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.93892 w_k1_ADM_s105
                                 + 1.09087 w_k1_AXP_s105
                                 + 1.03844 w_k1_DIS_s105
                                 + 0.943232 w_k1_FDX_s105
                                 + 1.42671 w_k1_IBM_s105
                                 + 1.10554 w_k1_INTC_s105
                                 + 1.20635 w_k1_KO_s105 + 1.43717 w_k1_MAT_s105
                                 + 1.04335 w_k1_MSFT_s105
                                 + 1.02951 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.42135 w_k1_ADM_s105
                                 + 1.83421 w_k1_AXP_s105
                                 + 1.63433 w_k1_DIS_s105
                                 + 1.33642 w_k1_FDX_s105
                                 + 1.68647 w_k1_IBM_s105
                                 + 1.06237 w_k1_INTC_s105
                                 + 1.52346 w_k1_KO_s105 + 1.42027 w_k1_MAT_s105
                                 + 1.28604 w_k1_MSFT_s105
                                 + 1.20744 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.32677 w_k1_ADM_s105
                                 + 1.30332 w_k1_AXP_s105
                                 + 1.32106 w_k1_DIS_s105
                                 + 1.26075 w_k1_FDX_s105 + 1.1542 w_k1_IBM_s105
                                 + 1.09872 w_k1_INTC_s105 + 1.367 w_k1_KO_s105
                                 + 0.951039 w_k1_MAT_s105
                                 + 1.1699 w_k1_MSFT_s105 + 1.22754 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.811048 w_k1_ADM_s105
                                 + 0.689924 w_k1_AXP_s105
                                 + 0.633179 w_k1_DIS_s105
                                 + 0.697633 w_k1_FDX_s105
                                 + 0.804379 w_k1_IBM_s105
                                 + 0.683276 w_k1_INTC_s105
                                 + 0.740562 w_k1_KO_s105
                                 + 0.646085 w_k1_MAT_s105
                                 + 0.688112 w_k1_MSFT_s105
                                 + 0.810792 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.848324 w_k1_ADM_s105
                                 + 0.984707 w_k1_AXP_s105
                                 + 0.832862 w_k1_DIS_s105
                                 + 0.910372 w_k1_FDX_s105
                                 + 1.00896 w_k1_IBM_s105
                                 + 0.894603 w_k1_INTC_s105
                                 + 0.869784 w_k1_KO_s105
                                 + 0.992864 w_k1_MAT_s105
                                 + 0.866843 w_k1_MSFT_s105
                                 + 0.974701 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.902434 w_k1_ADM_s105
                                 + 1.00869 w_k1_AXP_s105
                                 + 0.854764 w_k1_DIS_s105
                                 + 1.05783 w_k1_FDX_s105
                                 + 0.683827 w_k1_IBM_s105
                                 + 0.908649 w_k1_INTC_s105
                                 + 0.866504 w_k1_KO_s105
                                 + 0.678343 w_k1_MAT_s105
                                 + 1.0236 w_k1_MSFT_s105
                                 + 0.894657 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.28804 w_k1_ADM_s105
                                 + 0.941044 w_k1_AXP_s105
                                 + 1.19169 w_k1_DIS_s105
                                 + 1.11329 w_k1_FDX_s105
                                 + 0.896701 w_k1_IBM_s105
                                 + 1.04827 w_k1_INTC_s105 + 1.0836 w_k1_KO_s105
                                 + 0.628602 w_k1_MAT_s105
                                 + 1.10726 w_k1_MSFT_s105
                                 + 1.02551 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.87905 w_k1_ADM_s105
                                 + 0.921647 w_k1_AXP_s105
                                 + 0.740553 w_k1_DIS_s105
                                 + 0.760507 w_k1_FDX_s105
                                 + 1.20203 w_k1_IBM_s105
                                 + 0.797959 w_k1_INTC_s105
                                 + 1.08701 w_k1_KO_s105
                                 + 0.996474 w_k1_MAT_s105
                                 + 0.80497 w_k1_MSFT_s105
                                 + 0.916138 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.889536 w_k1_ADM_s106
                                 + 0.888226 w_k1_AXP_s106
                                 + 0.862853 w_k1_DIS_s106
                                 + 1.03089 w_k1_FDX_s106
                                 + 0.700405 w_k1_IBM_s106
                                 + 0.828103 w_k1_INTC_s106
                                 + 0.869959 w_k1_KO_s106
                                 + 1.16622 w_k1_MAT_s106
                                 + 1.02913 w_k1_MSFT_s106
                                 + 0.921816 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.03637 w_k1_ADM_s106
                                 + 0.896683 w_k1_AXP_s106
                                 + 1.10391 w_k1_DIS_s106
                                 + 1.09807 w_k1_FDX_s106
                                 + 1.01538 w_k1_IBM_s106
                                 + 1.12576 w_k1_INTC_s106
                                 + 1.08621 w_k1_KO_s106 + 1.21903 w_k1_MAT_s106
                                 + 1.19309 w_k1_MSFT_s106
                                 + 1.01081 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.985677 w_k1_ADM_s106
                                 + 1.066 w_k1_AXP_s106 + 0.89572 w_k1_DIS_s106
                                 + 1.04846 w_k1_FDX_s106
                                 + 0.950209 w_k1_IBM_s106
                                 + 0.821716 w_k1_INTC_s106
                                 + 0.876658 w_k1_KO_s106
                                 + 1.31296 w_k1_MAT_s106
                                 + 0.818188 w_k1_MSFT_s106
                                 + 0.88566 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.37946 w_k1_ADM_s106
                                 + 1.31436 w_k1_AXP_s106
                                 + 1.55031 w_k1_DIS_s106 + 1.3343 w_k1_FDX_s106
                                 + 1.25089 w_k1_IBM_s106
                                 + 1.09557 w_k1_INTC_s106
                                 + 1.49918 w_k1_KO_s106 + 1.69709 w_k1_MAT_s106
                                 + 1.30171 w_k1_MSFT_s106
                                 + 1.18532 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.991979 w_k1_ADM_s106
                                 + 0.827289 w_k1_AXP_s106
                                 + 0.983485 w_k1_DIS_s106
                                 + 0.9395 w_k1_FDX_s106 + 1.3411 w_k1_IBM_s106
                                 + 0.993259 w_k1_INTC_s106
                                 + 1.11535 w_k1_KO_s106 + 1.1608 w_k1_MAT_s106
                                 + 1.05041 w_k1_MSFT_s106
                                 + 0.959745 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.829802 w_k1_ADM_s106
                                 + 1.11225 w_k1_AXP_s106
                                 + 0.783567 w_k1_DIS_s106
                                 + 0.925466 w_k1_FDX_s106
                                 + 1.56076 w_k1_IBM_s106
                                 + 0.751355 w_k1_INTC_s106
                                 + 1.10881 w_k1_KO_s106 + 2.51235 w_k1_MAT_s106
                                 + 0.848028 w_k1_MSFT_s106
                                 + 0.914501 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.780101 w_k1_ADM_s106
                                 + 0.702899 w_k1_AXP_s106
                                 + 0.57158 w_k1_DIS_s106
                                 + 0.682541 w_k1_FDX_s106
                                 + 0.952549 w_k1_IBM_s106
                                 + 0.79709 w_k1_INTC_s106
                                 + 0.776613 w_k1_KO_s106
                                 + 0.734334 w_k1_MAT_s106
                                 + 0.721639 w_k1_MSFT_s106
                                 + 0.762484 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.670451 w_k1_ADM_s106
                                 + 0.816196 w_k1_AXP_s106
                                 + 0.574546 w_k1_DIS_s106
                                 + 0.666007 w_k1_FDX_s106
                                 + 1.25391 w_k1_IBM_s106
                                 + 0.765705 w_k1_INTC_s106
                                 + 0.949502 w_k1_KO_s106
                                 + 1.49507 w_k1_MAT_s106
                                 + 0.723497 w_k1_MSFT_s106
                                 + 0.812394 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.07392 w_k1_ADM_s107
                                 + 0.96877 w_k1_AXP_s107
                                 + 1.12511 w_k1_DIS_s107
                                 + 1.18375 w_k1_FDX_s107
                                 + 0.790747 w_k1_IBM_s107
                                 + 1.20732 w_k1_INTC_s107
                                 + 1.05792 w_k1_KO_s107 + 1.08184 w_k1_MAT_s107
                                 + 1.21902 w_k1_MSFT_s107
                                 + 1.02576 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.17416 w_k1_ADM_s107
                                 + 1.21172 w_k1_AXP_s107
                                 + 1.14911 w_k1_DIS_s107
                                 + 1.04653 w_k1_FDX_s107
                                 + 1.64162 w_k1_IBM_s107
                                 + 1.02415 w_k1_INTC_s107 + 1.3491 w_k1_KO_s107
                                 + 1.87331 w_k1_MAT_s107
                                 + 1.02774 w_k1_MSFT_s107
                                 + 1.02751 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.48351 w_k1_ADM_s107
                                 + 1.12271 w_k1_AXP_s107
                                 + 1.31577 w_k1_DIS_s107
                                 + 1.40792 w_k1_FDX_s107
                                 + 0.756851 w_k1_IBM_s107
                                 + 1.27377 w_k1_INTC_s107
                                 + 1.22625 w_k1_KO_s107 + 1.17456 w_k1_MAT_s107
                                 + 1.3464 w_k1_MSFT_s107 + 1.08886 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.872144 w_k1_ADM_s107
                                 + 0.876024 w_k1_AXP_s107
                                 + 0.742851 w_k1_DIS_s107
                                 + 0.877251 w_k1_FDX_s107
                                 + 1.22894 w_k1_IBM_s107
                                 + 1.00248 w_k1_INTC_s107
                                 + 1.01732 w_k1_KO_s107 + 1.71529 w_k1_MAT_s107
                                 + 0.882237 w_k1_MSFT_s107
                                 + 0.835884 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.882648 w_k1_ADM_s107
                                 + 0.643619 w_k1_AXP_s107
                                 + 0.667376 w_k1_DIS_s107
                                 + 0.851603 w_k1_FDX_s107
                                 + 0.750152 w_k1_IBM_s107
                                 + 1.03795 w_k1_INTC_s107
                                 + 0.938331 w_k1_KO_s107
                                 + 0.580399 w_k1_MAT_s107
                                 + 0.843876 w_k1_MSFT_s107
                                 + 0.797952 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.44694 w_k1_ADM_s107
                                 + 1.37395 w_k1_AXP_s107
                                 + 1.59169 w_k1_DIS_s107
                                 + 1.40503 w_k1_FDX_s107
                                 + 1.30838 w_k1_IBM_s107
                                 + 1.31766 w_k1_INTC_s107
                                 + 1.51048 w_k1_KO_s107 + 1.68176 w_k1_MAT_s107
                                 + 1.22814 w_k1_MSFT_s107
                                 + 1.22366 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.2638 w_k1_ADM_s107
                                 + 1.36122 w_k1_AXP_s107
                                 + 1.05126 w_k1_DIS_s107
                                 + 1.15246 w_k1_FDX_s107
                                 + 1.23882 w_k1_IBM_s107
                                 + 1.1802 w_k1_INTC_s107 + 1.14632 w_k1_KO_s107
                                 + 1.68049 w_k1_MAT_s107
                                 + 1.02201 w_k1_MSFT_s107 + 0.9813 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.12985 w_k1_ADM_s107
                                 + 0.87966 w_k1_AXP_s107
                                 + 1.01641 w_k1_DIS_s107
                                 + 1.03549 w_k1_FDX_s107 + 1.0797 w_k1_IBM_s107
                                 + 1.18537 w_k1_INTC_s107
                                 + 1.13246 w_k1_KO_s107 + 1.16748 w_k1_MAT_s107
                                 + 1.09698 w_k1_MSFT_s107
                                 + 0.944603 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.02924 w_k1_ADM_s110
                                 + 0.816247 w_k1_AXP_s110
                                 + 1.08522 w_k1_DIS_s110 + 0.9411 w_k1_FDX_s110
                                 + 0.836718 w_k1_IBM_s110
                                 + 1.10474 w_k1_INTC_s110
                                 + 0.893697 w_k1_KO_s110
                                 + 0.709622 w_k1_MAT_s110
                                 + 1.07726 w_k1_MSFT_s110
                                 + 1.01567 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.02456 w_k1_ADM_s110
                                 + 0.992864 w_k1_AXP_s110
                                 + 0.764278 w_k1_DIS_s110
                                 + 0.987362 w_k1_FDX_s110
                                 + 0.811013 w_k1_IBM_s110
                                 + 0.927163 w_k1_INTC_s110
                                 + 0.769003 w_k1_KO_s110
                                 + 0.962915 w_k1_MAT_s110
                                 + 0.951467 w_k1_MSFT_s110
                                 + 0.888307 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.722897 w_k1_ADM_s110
                                 + 0.567872 w_k1_AXP_s110
                                 + 0.66259 w_k1_DIS_s110
                                 + 0.697592 w_k1_FDX_s110
                                 + 0.699493 w_k1_IBM_s110
                                 + 0.880968 w_k1_INTC_s110
                                 + 0.717693 w_k1_KO_s110
                                 + 0.543759 w_k1_MAT_s110
                                 + 0.767189 w_k1_MSFT_s110
                                 + 0.841031 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.26802 w_k1_ADM_s110
                                 + 1.24162 w_k1_AXP_s110
                                 + 1.18282 w_k1_DIS_s110
                                 + 1.23573 w_k1_FDX_s110
                                 + 0.815943 w_k1_IBM_s110
                                 + 1.20301 w_k1_INTC_s110
                                 + 1.08326 w_k1_KO_s110 + 1.07982 w_k1_MAT_s110
                                 + 1.18174 w_k1_MSFT_s110
                                 + 1.11975 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.01387 w_k1_ADM_s110
                                 + 1.02983 w_k1_AXP_s110
                                 + 0.857778 w_k1_DIS_s110
                                 + 0.912078 w_k1_FDX_s110
                                 + 1.31496 w_k1_IBM_s110
                                 + 1.02916 w_k1_INTC_s110
                                 + 1.02355 w_k1_KO_s110 + 1.35186 w_k1_MAT_s110
                                 + 0.992074 w_k1_MSFT_s110
                                 + 0.910548 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.50125 w_k1_ADM_s110
                                 + 1.4071 w_k1_AXP_s110 + 1.55016 w_k1_DIS_s110
                                 + 1.31783 w_k1_FDX_s110
                                 + 1.26533 w_k1_IBM_s110
                                 + 1.15109 w_k1_INTC_s110
                                 + 1.29836 w_k1_KO_s110 + 1.86776 w_k1_MAT_s110
                                 + 1.36178 w_k1_MSFT_s110
                                 + 1.27383 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.01813 w_k1_ADM_s110
                                 + 1.07965 w_k1_AXP_s110
                                 + 1.42653 w_k1_DIS_s110
                                 + 1.00378 w_k1_FDX_s110
                                 + 1.50395 w_k1_IBM_s110
                                 + 1.24665 w_k1_INTC_s110
                                 + 1.36676 w_k1_KO_s110 + 1.36727 w_k1_MAT_s110
                                 + 1.09019 w_k1_MSFT_s110
                                 + 1.07708 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.860282 w_k1_ADM_s110
                                 + 0.559938 w_k1_AXP_s110
                                 + 0.778234 w_k1_DIS_s110
                                 + 0.817174 w_k1_FDX_s110
                                 + 0.784857 w_k1_IBM_s110
                                 + 0.899667 w_k1_INTC_s110
                                 + 0.860276 w_k1_KO_s110
                                 + 0.812769 w_k1_MAT_s110
                                 + 0.898545 w_k1_MSFT_s110
                                 + 0.957884 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.77162 w_k1_ADM_s111
                                 + 0.894571 w_k1_AXP_s111
                                 + 0.831514 w_k1_DIS_s111
                                 + 0.765196 w_k1_FDX_s111
                                 + 1.20405 w_k1_IBM_s111
                                 + 0.774936 w_k1_INTC_s111
                                 + 1.03245 w_k1_KO_s111 + 1.30197 w_k1_MAT_s111
                                 + 0.843982 w_k1_MSFT_s111
                                 + 0.921176 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.04741 w_k1_ADM_s111
                                 + 0.918889 w_k1_AXP_s111
                                 + 0.673815 w_k1_DIS_s111
                                 + 0.923993 w_k1_FDX_s111
                                 + 0.647471 w_k1_IBM_s111
                                 + 0.985657 w_k1_INTC_s111
                                 + 0.774621 w_k1_KO_s111
                                 + 0.740826 w_k1_MAT_s111
                                 + 0.891123 w_k1_MSFT_s111
                                 + 0.857549 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.23804 w_k1_ADM_s111
                                 + 1.12934 w_k1_AXP_s111 + 1.5456 w_k1_DIS_s111
                                 + 1.14474 w_k1_FDX_s111
                                 + 1.40601 w_k1_IBM_s111
                                 + 1.0307 w_k1_INTC_s111 + 1.40322 w_k1_KO_s111
                                 + 1.44207 w_k1_MAT_s111
                                 + 1.20471 w_k1_MSFT_s111
                                 + 1.07997 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.09734 w_k1_ADM_s111
                                 + 1.1193 w_k1_AXP_s111
                                 + 0.994664 w_k1_DIS_s111
                                 + 0.999142 w_k1_FDX_s111
                                 + 0.965549 w_k1_IBM_s111
                                 + 0.987709 w_k1_INTC_s111
                                 + 0.904816 w_k1_KO_s111
                                 + 1.15163 w_k1_MAT_s111
                                 + 1.05078 w_k1_MSFT_s111
                                 + 0.856223 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.98469 w_k1_ADM_s111
                                 + 1.15209 w_k1_AXP_s111
                                 + 1.26186 w_k1_DIS_s111
                                 + 1.13382 w_k1_FDX_s111
                                 + 1.78291 w_k1_IBM_s111
                                 + 0.934848 w_k1_INTC_s111
                                 + 1.34074 w_k1_KO_s111 + 2.14956 w_k1_MAT_s111
                                 + 1.05812 w_k1_MSFT_s111
                                 + 1.04251 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.05573 w_k1_ADM_s111
                                 + 1.21259 w_k1_AXP_s111
                                 + 1.39015 w_k1_DIS_s111
                                 + 1.23274 w_k1_FDX_s111
                                 + 1.16912 w_k1_IBM_s111
                                 + 1.05857 w_k1_INTC_s111 + 1.1711 w_k1_KO_s111
                                 + 1.30798 w_k1_MAT_s111
                                 + 1.15285 w_k1_MSFT_s111
                                 + 1.01195 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.703919 w_k1_ADM_s111
                                 + 0.764869 w_k1_AXP_s111
                                 + 0.709858 w_k1_DIS_s111
                                 + 0.79684 w_k1_FDX_s111
                                 + 0.867608 w_k1_IBM_s111
                                 + 0.881463 w_k1_INTC_s111
                                 + 0.794829 w_k1_KO_s111
                                 + 0.886554 w_k1_MAT_s111
                                 + 0.888812 w_k1_MSFT_s111
                                 + 0.754694 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.21618 w_k1_ADM_s111
                                 + 1.05394 w_k1_AXP_s111
                                 + 1.21141 w_k1_DIS_s111
                                 + 1.25757 w_k1_FDX_s111
                                 + 0.794367 w_k1_IBM_s111
                                 + 1.19462 w_k1_INTC_s111
                                 + 1.07075 w_k1_KO_s111 + 1.09207 w_k1_MAT_s111
                                 + 1.19193 w_k1_MSFT_s111
                                 + 0.866706 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.998043 w_k1_ADM_s112
                                 + 1.8025 w_k1_AXP_s112 + 1.65 w_k1_DIS_s112
                                 + 1.36423 w_k1_FDX_s112
                                 + 1.22726 w_k1_IBM_s112
                                 + 1.21709 w_k1_INTC_s112
                                 + 1.42576 w_k1_KO_s112 + 1.74267 w_k1_MAT_s112
                                 + 1.3743 w_k1_MSFT_s112 + 1.4294 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.903857 w_k1_ADM_s112
                                 + 0.82718 w_k1_AXP_s112
                                 + 0.824717 w_k1_DIS_s112
                                 + 0.775756 w_k1_FDX_s112
                                 + 0.990089 w_k1_IBM_s112
                                 + 0.767601 w_k1_INTC_s112
                                 + 0.909163 w_k1_KO_s112
                                 + 0.837304 w_k1_MAT_s112
                                 + 0.804022 w_k1_MSFT_s112
                                 + 0.922184 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.997494 w_k1_ADM_s112
                                 + 0.737113 w_k1_AXP_s112
                                 + 0.778101 w_k1_DIS_s112
                                 + 0.980003 w_k1_FDX_s112
                                 + 0.651975 w_k1_IBM_s112
                                 + 0.799406 w_k1_INTC_s112
                                 + 0.723004 w_k1_KO_s112
                                 + 0.549001 w_k1_MAT_s112
                                 + 0.876988 w_k1_MSFT_s112
                                 + 1.00765 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.05781 w_k1_ADM_s112
                                 + 1.29537 w_k1_AXP_s112
                                 + 1.01787 w_k1_DIS_s112
                                 + 1.03138 w_k1_FDX_s112
                                 + 1.55043 w_k1_IBM_s112
                                 + 0.938312 w_k1_INTC_s112
                                 + 1.24537 w_k1_KO_s112 + 1.5084 w_k1_MAT_s112
                                 + 0.83628 w_k1_MSFT_s112
                                 + 1.13468 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.25135 w_k1_ADM_s112
                                 + 1.18256 w_k1_AXP_s112
                                 + 1.21899 w_k1_DIS_s112
                                 + 1.16961 w_k1_FDX_s112
                                 + 0.941327 w_k1_IBM_s112
                                 + 1.08267 w_k1_INTC_s112 + 1.2076 w_k1_KO_s112
                                 + 0.962215 w_k1_MAT_s112
                                 + 1.06569 w_k1_MSFT_s112
                                 + 1.12576 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.759604 w_k1_ADM_s112
                                 + 0.847333 w_k1_AXP_s112
                                 + 0.783831 w_k1_DIS_s112
                                 + 0.865772 w_k1_FDX_s112
                                 + 0.7663 w_k1_IBM_s112
                                 + 0.883208 w_k1_INTC_s112
                                 + 0.787052 w_k1_KO_s112
                                 + 0.708677 w_k1_MAT_s112
                                 + 0.864277 w_k1_MSFT_s112
                                 + 0.974105 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.24979 w_k1_ADM_s112
                                 + 1.15307 w_k1_AXP_s112
                                 + 0.952064 w_k1_DIS_s112
                                 + 0.993464 w_k1_FDX_s112 + 1.037 w_k1_IBM_s112
                                 + 0.793108 w_k1_INTC_s112
                                 + 1.04881 w_k1_KO_s112 + 1.0171 w_k1_MAT_s112
                                 + 0.893256 w_k1_MSFT_s112
                                 + 0.982443 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.11535 w_k1_ADM_s112
                                 + 1.16432 w_k1_AXP_s112
                                 + 1.18535 w_k1_DIS_s112
                                 + 0.956928 w_k1_FDX_s112
                                 + 1.38457 w_k1_IBM_s112
                                 + 0.957576 w_k1_INTC_s112
                                 + 1.3138 w_k1_KO_s112 + 1.28725 w_k1_MAT_s112
                                 + 0.95984 w_k1_MSFT_s112
                                 + 1.15014 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.855193 w_k1_ADM_s113
                                 + 0.721393 w_k1_AXP_s113
                                 + 0.649031 w_k1_DIS_s113
                                 + 0.761649 w_k1_FDX_s113
                                 + 0.860727 w_k1_IBM_s113
                                 + 0.916715 w_k1_INTC_s113
                                 + 0.795193 w_k1_KO_s113
                                 + 0.775724 w_k1_MAT_s113
                                 + 0.862087 w_k1_MSFT_s113
                                 + 0.831264 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.07776 w_k1_ADM_s113
                                 + 1.21205 w_k1_AXP_s113
                                 + 0.961379 w_k1_DIS_s113
                                 + 0.972048 w_k1_FDX_s113
                                 + 0.986883 w_k1_IBM_s113
                                 + 1.07251 w_k1_INTC_s113
                                 + 0.832723 w_k1_KO_s113
                                 + 0.938023 w_k1_MAT_s113
                                 + 1.01354 w_k1_MSFT_s113
                                 + 0.966855 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.1175 w_k1_ADM_s113
                                 + 0.978738 w_k1_AXP_s113
                                 + 1.46728 w_k1_DIS_s113
                                 + 1.06258 w_k1_FDX_s113 + 1.631 w_k1_IBM_s113
                                 + 1.22154 w_k1_INTC_s113
                                 + 1.46608 w_k1_KO_s113 + 1.18088 w_k1_MAT_s113
                                 + 1.10856 w_k1_MSFT_s113 + 1.0728 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.788603 w_k1_ADM_s113
                                 + 0.763945 w_k1_AXP_s113
                                 + 1.10361 w_k1_DIS_s113
                                 + 0.993677 w_k1_FDX_s113
                                 + 0.91983 w_k1_IBM_s113
                                 + 1.18731 w_k1_INTC_s113
                                 + 1.00346 w_k1_KO_s113
                                 + 0.775903 w_k1_MAT_s113
                                 + 1.13889 w_k1_MSFT_s113
                                 + 0.941441 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.19267 w_k1_ADM_s113
                                 + 1.73906 w_k1_AXP_s113
                                 + 1.70826 w_k1_DIS_s113
                                 + 1.48327 w_k1_FDX_s113
                                 + 1.70059 w_k1_IBM_s113
                                 + 1.6537 w_k1_INTC_s113 + 1.52877 w_k1_KO_s113
                                 + 1.87044 w_k1_MAT_s113
                                 + 1.49421 w_k1_MSFT_s113
                                 + 1.25166 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.764439 w_k1_ADM_s113
                                 + 0.796935 w_k1_AXP_s113
                                 + 0.878694 w_k1_DIS_s113
                                 + 0.917726 w_k1_FDX_s113
                                 + 0.788905 w_k1_IBM_s113
                                 + 1.21395 w_k1_INTC_s113
                                 + 0.848411 w_k1_KO_s113
                                 + 0.692902 w_k1_MAT_s113
                                 + 0.921791 w_k1_MSFT_s113
                                 + 0.840456 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.19595 w_k1_ADM_s113
                                 + 1.36109 w_k1_AXP_s113
                                 + 1.52621 w_k1_DIS_s113
                                 + 1.24244 w_k1_FDX_s113
                                 + 1.14673 w_k1_IBM_s113
                                 + 1.54861 w_k1_INTC_s113
                                 + 1.06673 w_k1_KO_s113 + 1.1181 w_k1_MAT_s113
                                 + 1.2218 w_k1_MSFT_s113 + 1.1008 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.04183 w_k1_ADM_s113
                                 + 0.971283 w_k1_AXP_s113
                                 + 1.0998 w_k1_DIS_s113 + 1.09891 w_k1_FDX_s113
                                 + 1.17071 w_k1_IBM_s113
                                 + 1.32579 w_k1_INTC_s113
                                 + 1.08611 w_k1_KO_s113
                                 + 0.927633 w_k1_MAT_s113
                                 + 0.992362 w_k1_MSFT_s113
                                 + 1.00351 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.09873 w_k1_ADM_s114
                                 + 0.9696 w_k1_AXP_s114
                                 + 0.929546 w_k1_DIS_s114
                                 + 1.05364 w_k1_FDX_s114
                                 + 0.761906 w_k1_IBM_s114
                                 + 0.935401 w_k1_INTC_s114
                                 + 0.84302 w_k1_KO_s114
                                 + 0.637609 w_k1_MAT_s114
                                 + 1.01182 w_k1_MSFT_s114
                                 + 0.998903 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.04984 w_k1_ADM_s114
                                 + 1.23016 w_k1_AXP_s114
                                 + 1.15299 w_k1_DIS_s114
                                 + 1.03941 w_k1_FDX_s114
                                 + 1.18811 w_k1_IBM_s114
                                 + 0.797833 w_k1_INTC_s114
                                 + 1.08854 w_k1_KO_s114 + 1.27261 w_k1_MAT_s114
                                 + 0.98687 w_k1_MSFT_s114
                                 + 1.06502 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.24052 w_k1_ADM_s114
                                 + 1.59498 w_k1_AXP_s114
                                 + 1.47678 w_k1_DIS_s114
                                 + 1.47337 w_k1_FDX_s114
                                 + 0.884639 w_k1_IBM_s114
                                 + 1.27924 w_k1_INTC_s114
                                 + 1.02105 w_k1_KO_s114
                                 + 0.943475 w_k1_MAT_s114
                                 + 1.14317 w_k1_MSFT_s114
                                 + 1.01774 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.18656 w_k1_ADM_s114
                                 + 0.919632 w_k1_AXP_s114
                                 + 1.4264 w_k1_DIS_s114 + 1.18589 w_k1_FDX_s114
                                 + 1.01114 w_k1_IBM_s114
                                 + 1.08029 w_k1_INTC_s114
                                 + 1.08884 w_k1_KO_s114
                                 + 0.961604 w_k1_MAT_s114
                                 + 1.23136 w_k1_MSFT_s114
                                 + 1.15552 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.880271 w_k1_ADM_s114
                                 + 0.849132 w_k1_AXP_s114
                                 + 0.959487 w_k1_DIS_s114
                                 + 0.792067 w_k1_FDX_s114
                                 + 1.31722 w_k1_IBM_s114
                                 + 0.816586 w_k1_INTC_s114
                                 + 0.973116 w_k1_KO_s114
                                 + 0.992886 w_k1_MAT_s114
                                 + 0.866611 w_k1_MSFT_s114
                                 + 0.903251 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.36808 w_k1_ADM_s114
                                 + 1.45551 w_k1_AXP_s114
                                 + 1.80983 w_k1_DIS_s114
                                 + 1.43761 w_k1_FDX_s114
                                 + 1.43803 w_k1_IBM_s114
                                 + 1.23959 w_k1_INTC_s114
                                 + 1.40886 w_k1_KO_s114 + 1.40331 w_k1_MAT_s114
                                 + 1.38277 w_k1_MSFT_s114
                                 + 1.08678 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.783323 w_k1_ADM_s114
                                 + 0.769927 w_k1_AXP_s114
                                 + 0.695232 w_k1_DIS_s114
                                 + 0.761458 w_k1_FDX_s114
                                 + 0.830309 w_k1_IBM_s114
                                 + 0.744322 w_k1_INTC_s114
                                 + 0.787285 w_k1_KO_s114
                                 + 0.707883 w_k1_MAT_s114
                                 + 0.78263 w_k1_MSFT_s114
                                 + 0.765494 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.648918 w_k1_ADM_s114
                                 + 0.58592 w_k1_AXP_s114
                                 + 0.604334 w_k1_DIS_s114
                                 + 0.621371 w_k1_FDX_s114
                                 + 1.31312 w_k1_IBM_s114
                                 + 0.66344 w_k1_INTC_s114
                                 + 0.908381 w_k1_KO_s114
                                 + 1.10308 w_k1_MAT_s114
                                 + 0.743074 w_k1_MSFT_s114
                                 + 0.76527 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.02785 w_k1_ADM_s115
                                 + 1.33015 w_k1_AXP_s115
                                 + 1.44067 w_k1_DIS_s115
                                 + 1.17141 w_k1_FDX_s115
                                 + 1.23593 w_k1_IBM_s115
                                 + 1.36549 w_k1_INTC_s115
                                 + 1.32037 w_k1_KO_s115 + 0.88693 w_k1_MAT_s115
                                 + 1.20758 w_k1_MSFT_s115
                                 + 1.26086 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.988 w_k1_ADM_s115
                                 + 1.52263 w_k1_AXP_s115 + 1.6585 w_k1_DIS_s115
                                 + 1.40088 w_k1_FDX_s115
                                 + 0.991772 w_k1_IBM_s115
                                 + 1.39097 w_k1_INTC_s115
                                 + 1.36276 w_k1_KO_s115 + 1.43371 w_k1_MAT_s115
                                 + 1.27033 w_k1_MSFT_s115
                                 + 1.15822 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.706796 w_k1_ADM_s115
                                 + 0.790899 w_k1_AXP_s115
                                 + 1.18078 w_k1_DIS_s115
                                 + 1.03917 w_k1_FDX_s115
                                 + 0.628146 w_k1_IBM_s115
                                 + 1.12405 w_k1_INTC_s115
                                 + 0.962879 w_k1_KO_s115
                                 + 0.580349 w_k1_MAT_s115
                                 + 1.16296 w_k1_MSFT_s115
                                 + 1.03401 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.817268 w_k1_ADM_s115
                                 + 0.920988 w_k1_AXP_s115
                                 + 0.992805 w_k1_DIS_s115
                                 + 0.856873 w_k1_FDX_s115
                                 + 1.11205 w_k1_IBM_s115
                                 + 1.21619 w_k1_INTC_s115 + 1.1903 w_k1_KO_s115
                                 + 1.10121 w_k1_MAT_s115
                                 + 0.90057 w_k1_MSFT_s115
                                 + 1.10288 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.27191 w_k1_ADM_s115
                                 + 2.03909 w_k1_AXP_s115
                                 + 1.58298 w_k1_DIS_s115
                                 + 1.32652 w_k1_FDX_s115
                                 + 1.65946 w_k1_IBM_s115
                                 + 1.58225 w_k1_INTC_s115
                                 + 1.51276 w_k1_KO_s115 + 1.7002 w_k1_MAT_s115
                                 + 1.38218 w_k1_MSFT_s115
                                 + 1.31291 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.648851 w_k1_ADM_s115
                                 + 1.01903 w_k1_AXP_s115
                                 + 1.01461 w_k1_DIS_s115
                                 + 0.820592 w_k1_FDX_s115
                                 + 1.49083 w_k1_IBM_s115
                                 + 1.19147 w_k1_INTC_s115
                                 + 1.10571 w_k1_KO_s115 + 1.54415 w_k1_MAT_s115
                                 + 0.87245 w_k1_MSFT_s115
                                 + 1.01651 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.969027 w_k1_ADM_s115
                                 + 1.16236 w_k1_AXP_s115
                                 + 1.04739 w_k1_DIS_s115
                                 + 1.05944 w_k1_FDX_s115
                                 + 0.848355 w_k1_IBM_s115
                                 + 1.024 w_k1_INTC_s115 + 0.925642 w_k1_KO_s115
                                 + 0.837659 w_k1_MAT_s115
                                 + 0.962186 w_k1_MSFT_s115
                                 + 0.975489 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.718058 w_k1_ADM_s115
                                 + 0.642269 w_k1_AXP_s115
                                 + 0.681001 w_k1_DIS_s115
                                 + 0.71589 w_k1_FDX_s115
                                 + 0.771157 w_k1_IBM_s115
                                 + 0.855078 w_k1_INTC_s115
                                 + 0.833068 w_k1_KO_s115
                                 + 0.591265 w_k1_MAT_s115
                                 + 0.748703 w_k1_MSFT_s115
                                 + 0.866711 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.14575 w_k1_ADM_s116
                                 + 0.840534 w_k1_AXP_s116
                                 + 1.07548 w_k1_DIS_s116
                                 + 0.936111 w_k1_FDX_s116
                                 + 1.10942 w_k1_IBM_s116
                                 + 0.975223 w_k1_INTC_s116
                                 + 1.0578 w_k1_KO_s116 + 0.81367 w_k1_MAT_s116
                                 + 0.978666 w_k1_MSFT_s116
                                 + 0.867388 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.35394 w_k1_ADM_s116
                                 + 1.10316 w_k1_AXP_s116
                                 + 0.79867 w_k1_DIS_s116
                                 + 1.09229 w_k1_FDX_s116
                                 + 0.804859 w_k1_IBM_s116
                                 + 1.33029 w_k1_INTC_s116
                                 + 0.816011 w_k1_KO_s116
                                 + 0.579351 w_k1_MAT_s116
                                 + 1.03158 w_k1_MSFT_s116
                                 + 0.928993 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.1191 w_k1_ADM_s116
                                 + 0.885825 w_k1_AXP_s116
                                 + 1.07738 w_k1_DIS_s116
                                 + 1.06155 w_k1_FDX_s116
                                 + 0.861099 w_k1_IBM_s116
                                 + 0.960755 w_k1_INTC_s116
                                 + 0.927915 w_k1_KO_s116
                                 + 0.564508 w_k1_MAT_s116
                                 + 1.18102 w_k1_MSFT_s116
                                 + 0.93096 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.944555 w_k1_ADM_s116
                                 + 0.631783 w_k1_AXP_s116
                                 + 0.682625 w_k1_DIS_s116
                                 + 0.750437 w_k1_FDX_s116
                                 + 0.850537 w_k1_IBM_s116
                                 + 0.800138 w_k1_INTC_s116
                                 + 0.871422 w_k1_KO_s116
                                 + 0.55473 w_k1_MAT_s116
                                 + 0.828223 w_k1_MSFT_s116
                                 + 0.814255 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.19745 w_k1_ADM_s116
                                 + 1.21063 w_k1_AXP_s116
                                 + 1.10054 w_k1_DIS_s116
                                 + 1.02639 w_k1_FDX_s116 + 1.5709 w_k1_IBM_s116
                                 + 1.11078 w_k1_INTC_s116
                                 + 1.32669 w_k1_KO_s116 + 1.37992 w_k1_MAT_s116
                                 + 1.0219 w_k1_MSFT_s116 + 1.13032 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.54725 w_k1_ADM_s116
                                 + 1.41815 w_k1_AXP_s116 + 1.9246 w_k1_DIS_s116
                                 + 1.51644 w_k1_FDX_s116
                                 + 1.20471 w_k1_IBM_s116
                                 + 1.40429 w_k1_INTC_s116
                                 + 1.48991 w_k1_KO_s116 + 0.98152 w_k1_MAT_s116
                                 + 1.54251 w_k1_MSFT_s116
                                 + 1.28942 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.18183 w_k1_ADM_s116
                                 + 1.02972 w_k1_AXP_s116
                                 + 1.37033 w_k1_DIS_s116
                                 + 1.25332 w_k1_FDX_s116
                                 + 1.17762 w_k1_IBM_s116
                                 + 1.34393 w_k1_INTC_s116
                                 + 1.32933 w_k1_KO_s116
                                 + 0.801669 w_k1_MAT_s116
                                 + 1.33743 w_k1_MSFT_s116
                                 + 1.14718 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.16414 w_k1_ADM_s116
                                 + 1.37413 w_k1_AXP_s116
                                 + 1.25668 w_k1_DIS_s116
                                 + 1.30044 w_k1_FDX_s116
                                 + 0.89777 w_k1_IBM_s116
                                 + 1.10993 w_k1_INTC_s116
                                 + 1.10527 w_k1_KO_s116
                                 + 0.993826 w_k1_MAT_s116
                                 + 1.16381 w_k1_MSFT_s116
                                 + 1.12444 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.714501 w_k1_ADM_s117
                                 + 0.789432 w_k1_AXP_s117
                                 + 0.567156 w_k1_DIS_s117
                                 + 0.734428 w_k1_FDX_s117
                                 + 0.781099 w_k1_IBM_s117
                                 + 0.913314 w_k1_INTC_s117
                                 + 0.722348 w_k1_KO_s117
                                 + 0.858612 w_k1_MAT_s117
                                 + 0.815679 w_k1_MSFT_s117
                                 + 0.834258 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.75501 w_k1_ADM_s117
                                 + 1.00887 w_k1_AXP_s117
                                 + 0.881524 w_k1_DIS_s117
                                 + 0.7565 w_k1_FDX_s117 + 1.55596 w_k1_IBM_s117
                                 + 0.982689 w_k1_INTC_s117
                                 + 0.958468 w_k1_KO_s117
                                 + 1.18437 w_k1_MAT_s117
                                 + 0.785966 w_k1_MSFT_s117
                                 + 0.882043 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.17319 w_k1_ADM_s117
                                 + 1.10155 w_k1_AXP_s117
                                 + 1.18003 w_k1_DIS_s117 + 1.1718 w_k1_FDX_s117
                                 + 0.817094 w_k1_IBM_s117
                                 + 1.2398 w_k1_INTC_s117 + 1.00746 w_k1_KO_s117
                                 + 0.916778 w_k1_MAT_s117
                                 + 1.15467 w_k1_MSFT_s117
                                 + 0.889992 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.16757 w_k1_ADM_s117
                                 + 1.40001 w_k1_AXP_s117
                                 + 1.37506 w_k1_DIS_s117
                                 + 1.32163 w_k1_FDX_s117
                                 + 1.16752 w_k1_IBM_s117
                                 + 1.5177 w_k1_INTC_s117 + 1.24679 w_k1_KO_s117
                                 + 1.44076 w_k1_MAT_s117
                                 + 1.37275 w_k1_MSFT_s117
                                 + 1.12862 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.954273 w_k1_ADM_s117
                                 + 0.962863 w_k1_AXP_s117
                                 + 1.01187 w_k1_DIS_s117
                                 + 0.989851 w_k1_FDX_s117
                                 + 1.21327 w_k1_IBM_s117
                                 + 1.31563 w_k1_INTC_s117
                                 + 1.05943 w_k1_KO_s117 + 1.23017 w_k1_MAT_s117
                                 + 1.08334 w_k1_MSFT_s117
                                 + 0.997744 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.936567 w_k1_ADM_s117
                                 + 1.01121 w_k1_AXP_s117
                                 + 0.783346 w_k1_DIS_s117
                                 + 1.04727 w_k1_FDX_s117
                                 + 0.647969 w_k1_IBM_s117
                                 + 1.14297 w_k1_INTC_s117
                                 + 0.691008 w_k1_KO_s117
                                 + 0.68498 w_k1_MAT_s117
                                 + 0.904918 w_k1_MSFT_s117
                                 + 0.927932 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.704261 w_k1_ADM_s117
                                 + 0.605954 w_k1_AXP_s117
                                 + 0.623586 w_k1_DIS_s117
                                 + 0.745511 w_k1_FDX_s117
                                 + 0.854107 w_k1_IBM_s117
                                 + 0.960472 w_k1_INTC_s117
                                 + 0.844387 w_k1_KO_s117
                                 + 0.724638 w_k1_MAT_s117
                                 + 0.82587 w_k1_MSFT_s117
                                 + 0.856407 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.867046 w_k1_ADM_s117
                                 + 0.787431 w_k1_AXP_s117
                                 + 0.831865 w_k1_DIS_s117
                                 + 0.930907 w_k1_FDX_s117
                                 + 0.975529 w_k1_IBM_s117
                                 + 1.10505 w_k1_INTC_s117
                                 + 0.858136 w_k1_KO_s117
                                 + 0.643956 w_k1_MAT_s117
                                 + 0.978297 w_k1_MSFT_s117
                                 + 0.85873 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0246176947931998 w_k1_ADM_s000
                                 + 0.0441261784412384 w_k1_ADM_s001
                                 + 0.0177108373576804 w_k1_ADM_s002
                                 + 0.0149726147708704 w_k1_ADM_s003
                                 + 0.0152841661409994 w_k1_ADM_s004
                                 + 0.02353324989582 w_k1_ADM_s005
                                 + 0.0163433070326215 w_k1_ADM_s006
                                 + 0.0441965195856669 w_k1_ADM_s007
                                 + 0.0201472562107311 w_k1_ADM_s010
                                 + 0.0192672325140527 w_k1_ADM_s011
                                 + 0.019970959408413 w_k1_ADM_s012
                                 + 0.0209867918505328 w_k1_ADM_s013
                                 + 0.0251284266989467 w_k1_ADM_s014
                                 + 0.00996658206934096 w_k1_ADM_s015
                                 + 0.0208001595158506 w_k1_ADM_s016
                                 + 0.0125930839843245 w_k1_ADM_s017
                                 + 0.0278663990427733 w_k1_ADM_s100
                                 + 0.0253993873888709 w_k1_ADM_s101
                                 + 0.0155723872734791 w_k1_ADM_s102
                                 + 0.0278940097233986 w_k1_ADM_s103
                                 + 0.0244981178513556 w_k1_ADM_s104
                                 + 0.0408351738177506 w_k1_ADM_s105
                                 + 0.0244415791466979 w_k1_ADM_s106
                                 + 0.0724811847428843 w_k1_ADM_s107
                                 + 0.0634291088863538 w_k1_ADM_s110
                                 + 0.0298844575905015 w_k1_ADM_s111
                                 + 0.0485414119955956 w_k1_ADM_s112
                                 + 0.0337726383246767 w_k1_ADM_s113
                                 + 0.111426810040039 w_k1_ADM_s114
                                 + 0.0370399376709688 w_k1_ADM_s115
                                 + 0.0263188981681753 w_k1_ADM_s116
                                 + 0.0560559581220743 w_k1_ADM_s117
                                 + 0.0225597734204083 w_k1_AXP_s000
                                 + 0.0430357123524497 w_k1_AXP_s001
                                 + 0.0196801850746941 w_k1_AXP_s002
                                 + 0.0147391987857779 w_k1_AXP_s003
                                 + 0.0193257134851396 w_k1_AXP_s004
                                 + 0.0278804602259619 w_k1_AXP_s005
                                 + 0.0179833137072036 w_k1_AXP_s006
                                 + 0.0472157956069757 w_k1_AXP_s007
                                 + 0.0211702482933155 w_k1_AXP_s010
                                 + 0.0152973752543747 w_k1_AXP_s011
                                 + 0.019645793086989 w_k1_AXP_s012
                                 + 0.0209462293903436 w_k1_AXP_s013
                                 + 0.0227145417642432 w_k1_AXP_s014
                                 + 0.0103773535340677 w_k1_AXP_s015
                                 + 0.020638650727733 w_k1_AXP_s016
                                 + 0.0105674031550184 w_k1_AXP_s017
                                 + 0.0254165654824697 w_k1_AXP_s100
                                 + 0.0239290127436379 w_k1_AXP_s101
                                 + 0.0146006620158422 w_k1_AXP_s102
                                 + 0.0272579548737561 w_k1_AXP_s103
                                 + 0.0260183786082472 w_k1_AXP_s104
                                 + 0.0425603178208491 w_k1_AXP_s105
                                 + 0.0238910361249387 w_k1_AXP_s106
                                 + 0.064198408420865 w_k1_AXP_s107
                                 + 0.0582284775957735 w_k1_AXP_s110
                                 + 0.0302330003470348 w_k1_AXP_s111
                                 + 0.049801389922891 w_k1_AXP_s112
                                 + 0.0354186189338588 w_k1_AXP_s113
                                 + 0.111747962943221 w_k1_AXP_s114
                                 + 0.0454314346153107 w_k1_AXP_s115
                                 + 0.0220893387130954 w_k1_AXP_s116
                                 + 0.0591213560542181 w_k1_AXP_s117
                                 + 0.0242001422130533 w_k1_DIS_s000
                                 + 0.0436527869876937 w_k1_DIS_s001
                                 + 0.0180975174793225 w_k1_DIS_s002
                                 + 0.0147397138634116 w_k1_DIS_s003
                                 + 0.0197761974649408 w_k1_DIS_s004
                                 + 0.0299206263710378 w_k1_DIS_s005
                                 + 0.0177724724655355 w_k1_DIS_s006
                                 + 0.0471864390705846 w_k1_DIS_s007
                                 + 0.0209882826864009 w_k1_DIS_s010
                                 + 0.0173706285623284 w_k1_DIS_s011
                                 + 0.0201815583753739 w_k1_DIS_s012
                                 + 0.0219726428769104 w_k1_DIS_s013
                                 + 0.022846038588538 w_k1_DIS_s014
                                 + 0.0102162721360348 w_k1_DIS_s015
                                 + 0.0178785264980687 w_k1_DIS_s016
                                 + 0.0105019885131247 w_k1_DIS_s017
                                 + 0.0272685775801121 w_k1_DIS_s100
                                 + 0.02663240264738 w_k1_DIS_s101
                                 + 0.0146684932350535 w_k1_DIS_s102
                                 + 0.0303787121273648 w_k1_DIS_s103
                                 + 0.0285044692080489 w_k1_DIS_s104
                                 + 0.0400095814783217 w_k1_DIS_s105
                                 + 0.0244017301537936 w_k1_DIS_s106
                                 + 0.0655766597333937 w_k1_DIS_s107
                                 + 0.0626773834881586 w_k1_DIS_s110
                                 + 0.030362292104844 w_k1_DIS_s111
                                 + 0.0472952069851983 w_k1_DIS_s112
                                 + 0.0373139595871702 w_k1_DIS_s113
                                 + 0.119916114969831 w_k1_DIS_s114
                                 + 0.045826050801182 w_k1_DIS_s115
                                 + 0.0231325979780272 w_k1_DIS_s116
                                 + 0.0578045796168449 w_k1_DIS_s117
                                 + 0.0223330100310244 w_k1_FDX_s000
                                 + 0.0438317152523547 w_k1_FDX_s001
                                 + 0.0182929982460925 w_k1_FDX_s002
                                 + 0.0149729524608481 w_k1_FDX_s003
                                 + 0.018846617318401 w_k1_FDX_s004
                                 + 0.0265405495545629 w_k1_FDX_s005
                                 + 0.0180215347344987 w_k1_FDX_s006
                                 + 0.0461916328422388 w_k1_FDX_s007
                                 + 0.0218587628733254 w_k1_FDX_s010
                                 + 0.0160658200454319 w_k1_FDX_s011
                                 + 0.0188608349933596 w_k1_FDX_s012
                                 + 0.0228898936603397 w_k1_FDX_s013
                                 + 0.0232159549633176 w_k1_FDX_s014
                                 + 0.00927391839750922 w_k1_FDX_s015
                                 + 0.0187951182296432 w_k1_FDX_s016
                                 + 0.0101484360009855 w_k1_FDX_s017
                                 + 0.0247064565791419 w_k1_FDX_s100
                                 + 0.024617006350356 w_k1_FDX_s101
                                 + 0.0147274697040182 w_k1_FDX_s102
                                 + 0.0298988972927028 w_k1_FDX_s103
                                 + 0.0289456201258339 w_k1_FDX_s104
                                 + 0.0406093326970651 w_k1_FDX_s105
                                 + 0.0251426885492307 w_k1_FDX_s106
                                 + 0.0694957223905888 w_k1_FDX_s107
                                 + 0.0591611999255885 w_k1_FDX_s110
                                 + 0.0301663463706528 w_k1_FDX_s111
                                 + 0.0461395505238765 w_k1_FDX_s112
                                 + 0.0353429336488286 w_k1_FDX_s113
                                 + 0.110916468532919 w_k1_FDX_s114
                                 + 0.041633898851692 w_k1_FDX_s115
                                 + 0.0234196981011231 w_k1_FDX_s116
                                 + 0.0600187347603172 w_k1_FDX_s117
                                 + 0.0223700079302459 w_k1_IBM_s000
                                 + 0.0429294505952576 w_k1_IBM_s001
                                 + 0.0190652906808429 w_k1_IBM_s002
                                 + 0.0146967739928752 w_k1_IBM_s003
                                 + 0.0199833062591131 w_k1_IBM_s004
                                 + 0.0284517237410296 w_k1_IBM_s005
                                 + 0.0190699185000688 w_k1_IBM_s006
                                 + 0.0507132860659972 w_k1_IBM_s007
                                 + 0.0232592309760815 w_k1_IBM_s010
                                 + 0.0159084342806239 w_k1_IBM_s011
                                 + 0.0192691258742745 w_k1_IBM_s012
                                 + 0.0233530444677216 w_k1_IBM_s013
                                 + 0.023169746466997 w_k1_IBM_s014
                                 + 0.00937818312017805 w_k1_IBM_s015
                                 + 0.0184797076924507 w_k1_IBM_s016
                                 + 0.010239585545756 w_k1_IBM_s017
                                 + 0.0240790536110474 w_k1_IBM_s100
                                 + 0.0248776717889975 w_k1_IBM_s101
                                 + 0.0152065001875078 w_k1_IBM_s102
                                 + 0.030588141271846 w_k1_IBM_s103
                                 + 0.0289956723298208 w_k1_IBM_s104
                                 + 0.0414268157156703 w_k1_IBM_s105
                                 + 0.0255577607624217 w_k1_IBM_s106
                                 + 0.0723024499900924 w_k1_IBM_s107
                                 + 0.0626410976954495 w_k1_IBM_s110
                                 + 0.0299361868420128 w_k1_IBM_s111
                                 + 0.0465651283549959 w_k1_IBM_s112
                                 + 0.036124901539242 w_k1_IBM_s113
                                 + 0.11581111205634 w_k1_IBM_s114
                                 + 0.0433535783476144 w_k1_IBM_s115
                                 + 0.0224152891490125 w_k1_IBM_s116
                                 + 0.0600156102683799 w_k1_IBM_s117
                                 + 0.0255535205188117 w_k1_INTC_s000
                                 + 0.0438366469057941 w_k1_INTC_s001
                                 + 0.0183560367373037 w_k1_INTC_s002
                                 + 0.0128322476636823 w_k1_INTC_s003
                                 + 0.0207246029061492 w_k1_INTC_s004
                                 + 0.0235833411296656 w_k1_INTC_s005
                                 + 0.0185994895117115 w_k1_INTC_s006
                                 + 0.0486252786686664 w_k1_INTC_s007
                                 + 0.0262018959934039 w_k1_INTC_s010
                                 + 0.0134538167764177 w_k1_INTC_s011
                                 + 0.0178330002157715 w_k1_INTC_s012
                                 + 0.0197148551391692 w_k1_INTC_s013
                                 + 0.0218178367294447 w_k1_INTC_s014
                                 + 0.00895005998452788 w_k1_INTC_s015
                                 + 0.0162156879784178 w_k1_INTC_s016
                                 + 0.00986616343911655 w_k1_INTC_s017
                                 + 0.0264063455953631 w_k1_INTC_s100
                                 + 0.0256399570751834 w_k1_INTC_s101
                                 + 0.0134929002844533 w_k1_INTC_s102
                                 + 0.029515756440501 w_k1_INTC_s103
                                 + 0.0302854260579976 w_k1_INTC_s104
                                 + 0.03822798788455 w_k1_INTC_s105
                                 + 0.022442739673937 w_k1_INTC_s106
                                 + 0.0745883583274364 w_k1_INTC_s107
                                 + 0.0641036109049876 w_k1_INTC_s110
                                 + 0.0292671315859055 w_k1_INTC_s111
                                 + 0.042480040271943 w_k1_INTC_s112
                                 + 0.0424759518682524 w_k1_INTC_s113
                                 + 0.0990281861803626 w_k1_INTC_s114
                                 + 0.048495202231633 w_k1_INTC_s115
                                 + 0.024282355948569 w_k1_INTC_s116
                                 + 0.0717561380774184 w_k1_INTC_s117
                                 + 0.0227341305996049 w_k1_KO_s000
                                 + 0.0451745076134986 w_k1_KO_s001
                                 + 0.0172573692948017 w_k1_KO_s002
                                 + 0.0135422091918181 w_k1_KO_s003
                                 + 0.0195914567224183 w_k1_KO_s004
                                 + 0.0245953631548279 w_k1_KO_s005
                                 + 0.0179528809369129 w_k1_KO_s006
                                 + 0.0454889896502976 w_k1_KO_s007
                                 + 0.0229006304433355 w_k1_KO_s010
                                 + 0.0164367117826574 w_k1_KO_s011
                                 + 0.0195550586040983 w_k1_KO_s012
                                 + 0.0211077128310116 w_k1_KO_s013
                                 + 0.0217142047310309 w_k1_KO_s014
                                 + 0.00984789298962906 w_k1_KO_s015
                                 + 0.0157949119082775 w_k1_KO_s016
                                 + 0.0113890310275977 w_k1_KO_s017
                                 + 0.0261273204619441 w_k1_KO_s100
                                 + 0.0239416572341691 w_k1_KO_s101
                                 + 0.0143181103228478 w_k1_KO_s102
                                 + 0.0313473356608933 w_k1_KO_s103
                                 + 0.0322830948728312 w_k1_KO_s104
                                 + 0.0419547231266292 w_k1_KO_s105
                                 + 0.0255072543937604 w_k1_KO_s106
                                 + 0.0750787889255452 w_k1_KO_s107
                                 + 0.0606437399008101 w_k1_KO_s110
                                 + 0.0297940340067581 w_k1_KO_s111
                                 + 0.0486838821644374 w_k1_KO_s112
                                 + 0.0340918060308023 w_k1_KO_s113
                                 + 0.108785916872429 w_k1_KO_s114
                                 + 0.0460039961590598 w_k1_KO_s115
                                 + 0.0230380642564488 w_k1_KO_s116
                                 + 0.0576057137101443 w_k1_KO_s117
                                 + 0.020550027964295 w_k1_MAT_s000
                                 + 0.0372911456401209 w_k1_MAT_s001
                                 + 0.0211576199625289 w_k1_MAT_s002
                                 + 0.0161962756858221 w_k1_MAT_s003
                                 + 0.0196321319433474 w_k1_MAT_s004
                                 + 0.0282792292383624 w_k1_MAT_s005
                                 + 0.0170998545557094 w_k1_MAT_s006
                                 + 0.0482962277787973 w_k1_MAT_s007
                                 + 0.02505882734496 w_k1_MAT_s010
                                 + 0.0158737549100753 w_k1_MAT_s011
                                 + 0.0181382865206226 w_k1_MAT_s012
                                 + 0.024309066766657 w_k1_MAT_s013
                                 + 0.0240981883470289 w_k1_MAT_s014
                                 + 0.00890130208542689 w_k1_MAT_s015
                                 + 0.0166127478749074 w_k1_MAT_s016
                                 + 0.0116762000281114 w_k1_MAT_s017
                                 + 0.0250654279392787 w_k1_MAT_s100
                                 + 0.0231128750274897 w_k1_MAT_s101
                                 + 0.0153852788973668 w_k1_MAT_s102
                                 + 0.0289680341830992 w_k1_MAT_s103
                                 + 0.0306039407351283 w_k1_MAT_s104
                                 + 0.037296587221847 w_k1_MAT_s105
                                 + 0.0325111697597879 w_k1_MAT_s106
                                 + 0.0857170405853011 w_k1_MAT_s107
                                 + 0.0660944107096564 w_k1_MAT_s110
                                 + 0.0344630757319463 w_k1_MAT_s111
                                 + 0.0458697782062724 w_k1_MAT_s112
                                 + 0.0325137552047903 w_k1_MAT_s113
                                 + 0.106996584906557 w_k1_MAT_s114
                                 + 0.0419854180187232 w_k1_MAT_s115
                                 + 0.0169545663747322 w_k1_MAT_s116
                                 + 0.0606454028448924 w_k1_MAT_s117
                                 + 0.0226369844748804 w_k1_MSFT_s000
                                 + 0.0422134738909164 w_k1_MSFT_s001
                                 + 0.0189799466581784 w_k1_MSFT_s002
                                 + 0.0136533127349476 w_k1_MSFT_s003
                                 + 0.0190537477193054 w_k1_MSFT_s004
                                 + 0.0250216409166758 w_k1_MSFT_s005
                                 + 0.0183657578265386 w_k1_MSFT_s006
                                 + 0.0471219510493972 w_k1_MSFT_s007
                                 + 0.0233076463004889 w_k1_MSFT_s010
                                 + 0.0154976619832661 w_k1_MSFT_s011
                                 + 0.0189802778050705 w_k1_MSFT_s012
                                 + 0.0222368884249529 w_k1_MSFT_s013
                                 + 0.0235638214277861 w_k1_MSFT_s014
                                 + 0.00939169187672569 w_k1_MSFT_s015
                                 + 0.0177691248864096 w_k1_MSFT_s016
                                 + 0.00990645880796308 w_k1_MSFT_s017
                                 + 0.0260035343395628 w_k1_MSFT_s100
                                 + 0.0240881064932149 w_k1_MSFT_s101
                                 + 0.0142607083962643 w_k1_MSFT_s102
                                 + 0.0280874931290835 w_k1_MSFT_s103
                                 + 0.0276794742680598 w_k1_MSFT_s104
                                 + 0.0397545094731033 w_k1_MSFT_s105
                                 + 0.0245458753678947 w_k1_MSFT_s106
                                 + 0.0684690080023206 w_k1_MSFT_s107
                                 + 0.0628911357142532 w_k1_MSFT_s110
                                 + 0.0304218373749458 w_k1_MSFT_s111
                                 + 0.0437946271791562 w_k1_MSFT_s112
                                 + 0.0361453720669922 w_k1_MSFT_s113
                                 + 0.109207950243261 w_k1_MSFT_s114
                                 + 0.0418059206329662 w_k1_MSFT_s115
                                 + 0.0240263288185123 w_k1_MSFT_s116
                                 + 0.0622061976628849 w_k1_MSFT_s117
                                 + 0.0220200574132838 w_k1_PG_s000
                                 + 0.044014122930618 w_k1_PG_s001
                                 + 0.0191277909621034 w_k1_PG_s002
                                 + 0.0152976091257194 w_k1_PG_s003
                                 + 0.0191562239389712 w_k1_PG_s004
                                 + 0.0248855268884589 w_k1_PG_s005
                                 + 0.0172873489173554 w_k1_PG_s006
                                 + 0.0460275226285829 w_k1_PG_s007
                                 + 0.021875435890815 w_k1_PG_s010
                                 + 0.0149250088801024 w_k1_PG_s011
                                 + 0.0189531668669167 w_k1_PG_s012
                                 + 0.0228107240332318 w_k1_PG_s013
                                 + 0.0229451015107103 w_k1_PG_s014
                                 + 0.0100150967416838 w_k1_PG_s015
                                 + 0.0194773211604539 w_k1_PG_s016
                                 + 0.0105038856825996 w_k1_PG_s017
                                 + 0.0250181825823163 w_k1_PG_s100
                                 + 0.0236964730724204 w_k1_PG_s101
                                 + 0.0154160132787564 w_k1_PG_s102
                                 + 0.028993524632512 w_k1_PG_s103
                                 + 0.0304951604290791 w_k1_PG_s104
                                 + 0.040209315854776 w_k1_PG_s105
                                 + 0.0232098996103347 w_k1_PG_s106
                                 + 0.0629684731202067 w_k1_PG_s107
                                 + 0.0605407701981029 w_k1_PG_s110
                                 + 0.0266160490677871 w_k1_PG_s111
                                 + 0.0490607921400138 w_k1_PG_s112
                                 + 0.0334029918246571 w_k1_PG_s113
                                 + 0.105710028923627 w_k1_PG_s114
                                 + 0.0444709763544353 w_k1_PG_s115
                                 + 0.0217786505767915 w_k1_PG_s116
                                 + 0.0561405845853786 w_k1_PG_s117 >= 1.14
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
