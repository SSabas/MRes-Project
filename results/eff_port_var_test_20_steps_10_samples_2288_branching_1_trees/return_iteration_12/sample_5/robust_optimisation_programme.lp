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
 w_k1_ADM_s0_return_constraint:  1.07606 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.977494 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.01886 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.835435 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.19145 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.849853 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.08964 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.61975 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.05814 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.22133 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.14073 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.845697 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.991284 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.35237 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.20949 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.16512 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.1041 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.813036 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.743162 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.0115 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.761475 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.94896 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.03456 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.14248 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.588103 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.17623 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.43232 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.778819 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.889609 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.752742 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.00433 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.89451 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.23615 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.688891 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.26143 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.730427 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.30549 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.31586 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.05637 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.01694 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.18378 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.842735 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.24077 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.787484 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.919012 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.54393 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.17052 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.986492 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.31268 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.646557 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.822416 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.38806 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.26333 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.0755 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.56586 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.6989 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.65877 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.2122 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.757099 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.988133 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.14885 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.33805 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.577533 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.16791 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.72294 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.905649 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.09912 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.732135 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.874978 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.912495 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.591 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.720286 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.49521 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.663312 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.25973 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.14352 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.08843 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.960279 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.33065 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.73821 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.31327 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.841092 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.821746 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.58042 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.972741 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.04494 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.26059 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.572235 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.713193 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.25166 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.45176 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.18776 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.79657 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.703012 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.655436 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.3244 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.857787 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.971853 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.997731 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.77965 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.54629 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.26164 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.64878 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.89079 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.04336 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.683506 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.77895 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.783758 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.37535 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.524357 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.20227 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.608063 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.23526 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.10935 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.06148 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.921937 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.22997 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.843868 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.15459 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.890609 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.896209 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.41047 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.946638 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.10866 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.0592 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.705244 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.879379 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.14039 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.3399 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.17595 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.30272 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.840756 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.696701 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.07264 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.8213 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.958927 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.15586 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.47688 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.613351 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.12661 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.48381 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.939075 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.844657 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.855336 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.962852 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.86211 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.25395 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.615793 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.06058 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.752091 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.3055 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.13428 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.930883 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.23255 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.07457 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.979958 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.29248 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.79087 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.92227 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.07972 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.23758 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.831716 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.895 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.956254 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.603685 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.29516 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.07559 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.816278 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.82237 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.561018 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.00881 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.32726 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.777138 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.01885 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.829239 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.893694 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.828553 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.25149 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.29893 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.935999 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.59974 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.601083 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.804133 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.15406 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.24388 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.17963 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.68013 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.651982 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.0452 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.887581 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11334 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.97792 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.04435 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.761869 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.1336 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.930663 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.042 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.14491 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.07 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.12623 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.02134 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.840657 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.817182 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.98545 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.27684 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.19866 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.48883 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.901582 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.718755 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.16416 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.988737 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.02372 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.11071 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.45576 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.720751 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.0815 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.2396 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.859692 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.870713 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.876983 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.982677 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.882433 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.1439 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.813306 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.10775 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.710164 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.22488 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.00198 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.00409 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.01263 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.0754 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.87115 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.27745 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.917749 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.92439 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.34795 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.15926 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.01981 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.62483 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.897653 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.761908 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.30486 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.30267 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.06094 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.56624 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.681385 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.839984 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.20847 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.848279 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.03637 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.899096 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.1482 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.776146 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.33469 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.43609 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.867158 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.16038 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.741418 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.889553 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.966529 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.14422 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.870665 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.38886 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.607785 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.955107 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.94887 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.885473 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.11487 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.20263 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.07941 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.36593 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.845791 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.860168 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.05183 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.21667 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.04909 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 2.04953 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.789486 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.655956 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.65421 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.949241 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.06815 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.74863 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.536348 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.966246 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.08632 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.672767 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.926986 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.05277 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.885844 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.780556 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.35286 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.58363 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.982768 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.8014 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.546267 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.677415 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.949234 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.23562 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.15208 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.8866 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.571256 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.91649 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.698879 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.04985 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.939209 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.14749 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.820765 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.24965 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.995069 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.966197 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.3568 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.928679 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.09497 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.10477 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.816002 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.942261 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.15736 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.38812 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.22908 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.32954 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.924293 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.743519 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.12937 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.93868 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.940145 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.02714 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.32322 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.660691 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.06158 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.25826 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.8318 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.87247 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.765714 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.853639 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.785447 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.07149 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.657272 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.919542 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.691432 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.11083 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.997511 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.03157 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.990919 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.11452 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.917627 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11922 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.914727 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.89602 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.26262 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.06045 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.03744 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.09958 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.795759 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.878307 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.09059 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.13042 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.05765 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.20026 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.772709 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.803514 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.07572 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.875348 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.938414 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.04022 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.29636 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.735806 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.09455 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.27296 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.906095 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.03893 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.87766 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.977045 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.910186 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.19758 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.839089 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.1183 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.831948 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.16403 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.06927 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.410000430297191 z_k1_s0000
                                 - 0.90600190846471 z_k1_s0001
                                 - 0.536999256074492 z_k1_s0002
                                 - 0.35200054667634 z_k1_s0003
                                 - 0.474000529692889 z_k1_s0004
                                 - 0.942003203113792 z_k1_s0005
                                 - 0.706001007977438 z_k1_s0006
                                 - 0.392000254873222 z_k1_s0007
                                 - 0.388999428439965 z_k1_s0010
                                 - 0.491999153196022 z_k1_s0011
                                 - 0.650997416105387 z_k1_s0012
                                 - 0.466999166935912 z_k1_s0013
                                 - 0.359999404626312 z_k1_s0014
                                 - 0.398999025424748 z_k1_s0015
                                 - 1.48499779253308 z_k1_s0016
                                 - 0.725999362482015 z_k1_s0017
                                 - 0.234000641314295 z_k1_s0020
                                 - 0.139000054989214 z_k1_s0021
                                 - 0.195000219761867 z_k1_s0022
                                 - 0.136000022562104 z_k1_s0023
                                 - 0.225000544032965 z_k1_s0024
                                 - 0.364999539964397 z_k1_s0025
                                 - 0.46799939462832 z_k1_s0026
                                 - 0.126000040338423 z_k1_s0027
                                 - 0.150000131691825 z_k1_s0030
                                 - 0.428000713211455 z_k1_s0031
                                 - 0.371999222211351 z_k1_s0032
                                 - 0.185000200766597 z_k1_s0033
                                 - 0.334001497319539 z_k1_s0034
                                 - 0.737999106387256 z_k1_s0035
                                 - 0.198000012777154 z_k1_s0036
                                 - 0.470999803800123 z_k1_s0037
                                 - 0.452001557481033 z_k1_s0040
                                 - 0.280000621472513 z_k1_s0041
                                 - 0.309000758080883 z_k1_s0042
                                 - 0.793003106634516 z_k1_s0043
                                 - 0.219000356992883 z_k1_s0044
                                 - 0.730002295311843 z_k1_s0045
                                 - 0.266000441178585 z_k1_s0046
                                 - 0.267000548799772 z_k1_s0047
                                 - 0.259000313628908 z_k1_s0050
                                 - 0.144000158224552 z_k1_s0051
                                 - 0.635999827491009 z_k1_s0052
                                 - 0.251000159616305 z_k1_s0053
                                 - 0.518001150058274 z_k1_s0054
                                 - 0.199000073435191 z_k1_s0055
                                 - 0.208000083324226 z_k1_s0056
                                 - 0.399001309747961 z_k1_s0057
                                 - 0.42400016659965 z_k1_s0060
                                 - 0.418000140593051 z_k1_s0061
                                 - 0.871000223991259 z_k1_s0062
                                 - 1.16200211201136 z_k1_s0063
                                 - 1.15799749887347 z_k1_s0064
                                 - 0.73300087923951 z_k1_s0065
                                 - 0.301999846698798 z_k1_s0066
                                 - 1.1990024812854 z_k1_s0067
                                 - 0.527998827162792 z_k1_s0070
                                 - 0.300999421536929 z_k1_s0071
                                 - 0.525997375449385 z_k1_s0072
                                 - 0.19699959707771 z_k1_s0073
                                 - 0.169999525304895 z_k1_s0074
                                 - 0.246999583090645 z_k1_s0075
                                 - 0.55399939347629 z_k1_s0076
                                 - 0.527998827162792 z_k1_s0077
                                 - 0.625000700988644 z_k1_s0100
                                 - 0.345999408388253 z_k1_s0101
                                 - 0.375000420593187 z_k1_s0102
                                 - 0.72900054131342 z_k1_s0103
                                 - 0.279999811643394 z_k1_s0104
                                 - 0.216999854023631 z_k1_s0105
                                 - 0.264999732019727 z_k1_s0106
                                 - 0.302999661599945 z_k1_s0107
                                 - 0.380000914377206 z_k1_s0110
                                 - 0.757001358324127 z_k1_s0111
                                 - 0.70899924745115 z_k1_s0112
                                 - 1.3760027409015 z_k1_s0113
                                 - 0.348000951398318 z_k1_s0114
                                 - 0.650999179151624 z_k1_s0115
                                 - 0.538001137786341 z_k1_s0116
                                 - 0.657000120012197 z_k1_s0117
                                 - 0.951000207402783 z_k1_s0120
                                 - 0.770001092928848 z_k1_s0121
                                 - 0.748998253348358 z_k1_s0122
                                 - 0.484002045555572 z_k1_s0123
                                 - 0.458000273384869 z_k1_s0124
                                 - 0.761998426183554 z_k1_s0125
                                 - 0.334000307341828 z_k1_s0126
                                 - 0.246999955367999 z_k1_s0127
                                 - 0.266999808409927 z_k1_s0130
                                 - 0.817999466981651 z_k1_s0131
                                 - 1.26499781095514 z_k1_s0132
                                 - 0.670998682280763 z_k1_s0133
                                 - 0.714999284626109 z_k1_s0134
                                 - 0.588999469452757 z_k1_s0135
                                 - 1.01199584796736 z_k1_s0136
                                 - 0.664996690418078 z_k1_s0137
                                 - 1.00999886675708 z_k1_s0140
                                 - 0.509001566701436 z_k1_s0141
                                 - 0.404001000702224 z_k1_s0142
                                 - 0.46599953472156 z_k1_s0143
                                 - 0.360999695722043 z_k1_s0144
                                 - 0.340999934103001 z_k1_s0145
                                 - 0.253999880560623 z_k1_s0146
                                 - 0.289999814974747 z_k1_s0147
                                 - 0.391000810488736 z_k1_s0150
                                 - 0.251999730575722 z_k1_s0151
                                 - 0.308999869809357 z_k1_s0152
                                 - 0.721999565627792 z_k1_s0153
                                 - 0.175000005224654 z_k1_s0154
                                 - 0.531001210308145 z_k1_s0155
                                 - 0.420001199458227 z_k1_s0156
                                 - 0.496999276310604 z_k1_s0157
                                 - 0.480999300154162 z_k1_s0160
                                 - 0.214999960367992 z_k1_s0161
                                 - 0.177999863617718 z_k1_s0162
                                 - 0.21999991392075 z_k1_s0163
                                 - 0.27599949159161 z_k1_s0164
                                 - 0.435000608388515 z_k1_s0165
                                 - 0.236999951760067 z_k1_s0166
                                 - 0.405000397672117 z_k1_s0167
                                 - 0.497000189055461 z_k1_s0170
                                 - 0.529000374294505 z_k1_s0171
                                 - 1.2649994442907 z_k1_s0172
                                 - 0.742998090954377 z_k1_s0173
                                 - 1.56099893325108 z_k1_s0174
                                 - 1.16399969381776 z_k1_s0175
                                 - 1.03399986815947 z_k1_s0176
                                 - 0.622000078457685 z_k1_s0177
                                 - 0.0979996107380463 z_k1_s1000
                                 - 0.307999387231065 z_k1_s1001
                                 - 0.14999922972638 z_k1_s1002
                                 - 0.106999496843939 z_k1_s1003
                                 - 0.307999387231065 z_k1_s1004
                                 - 0.24199939132499 z_k1_s1005
                                 - 0.40999857136234 z_k1_s1006
                                 - 0.157999267008761 z_k1_s1007
                                 - 0.226999948053838 z_k1_s1010
                                 - 0.0819998743032541 z_k1_s1011
                                 - 0.228000066067174 z_k1_s1012
                                 - 0.193999554610284 z_k1_s1013
                                 - 0.133999475803193 z_k1_s1014
                                 - 0.0959998926915752 z_k1_s1015
                                 - 0.0729999791820754 z_k1_s1016
                                 - 0.132999357789857 z_k1_s1017
                                 - 0.0849999059011511 z_k1_s1020
                                 - 0.12400019511182 z_k1_s1021
                                 - 0.246000133915726 z_k1_s1022
                                 - 0.0949999121424397 z_k1_s1023
                                 - 0.0589999092937744 z_k1_s1024
                                 - 0.119999682495991 z_k1_s1025
                                 - 0.116999788533459 z_k1_s1026
                                 - 0.13500013330733 z_k1_s1027
                                 - 0.447999053523371 z_k1_s1030
                                 - 0.274998580755012 z_k1_s1031
                                 - 0.281999582181497 z_k1_s1032
                                 - 0.357999365422266 z_k1_s1033
                                 - 0.128999483011829 z_k1_s1034
                                 - 0.339998588604991 z_k1_s1035
                                 - 0.144999404252107 z_k1_s1036
                                 - 0.120999627291321 z_k1_s1037
                                 - 0.168999486821637 z_k1_s1040
                                 - 0.344999882434542 z_k1_s1041
                                 - 0.222999493498343 z_k1_s1042
                                 - 0.0980000334799453 z_k1_s1043
                                 - 0.0789999563603763 z_k1_s1044
                                 - 0.186000663886323 z_k1_s1045
                                 - 0.216999332534293 z_k1_s1046
                                 - 0.125000036818299 z_k1_s1047
                                 - 0.533998925181023 z_k1_s1050
                                 - 0.756000333444419 z_k1_s1051
                                 - 0.528997252925379 z_k1_s1052
                                 - 0.453999362648606 z_k1_s1053
                                 - 0.221999465866403 z_k1_s1054
                                 - 0.641000719504446 z_k1_s1055
                                 - 0.688997349188708 z_k1_s1056
                                 - 1.03100004364772 z_k1_s1057
                                 - 0.119999789870887 z_k1_s1060
                                 - 0.350999371264878 z_k1_s1061
                                 - 0.226998516230958 z_k1_s1062
                                 - 0.217999834579907 z_k1_s1063
                                 - 0.191998836155474 z_k1_s1064
                                 - 0.13099966245182 z_k1_s1065
                                 - 0.495999658145023 z_k1_s1066
                                 - 0.145999659698122 z_k1_s1067
                                 - 0.230999876903094 z_k1_s1070
                                 - 0.187999468994797 z_k1_s1071
                                 - 0.256999472615195 z_k1_s1072
                                 - 0.0870000586517973 z_k1_s1073
                                 - 0.110999924693697 z_k1_s1074
                                 - 0.194000213005144 z_k1_s1075
                                 - 0.154000021602045 z_k1_s1076
                                 - 0.33300008508104 z_k1_s1077
                                 - 0.35399819123451 z_k1_s1100
                                 - 0.421998637312313 z_k1_s1101
                                 - 0.269000234633511 z_k1_s1102
                                 - 0.218999723035391 z_k1_s1103
                                 - 0.145999902056803 z_k1_s1104
                                 - 0.666999479505498 z_k1_s1105
                                 - 0.367000571510785 z_k1_s1106
                                 - 0.156999775316734 z_k1_s1107
                                 - 0.580999371008169 z_k1_s1110
                                 - 0.301000674920078 z_k1_s1111
                                 - 0.23799922842494 z_k1_s1112
                                 - 0.264000823169147 z_k1_s1113
                                 - 0.142000092409098 z_k1_s1114
                                 - 0.117999915530066 z_k1_s1115
                                 - 0.246000409884822 z_k1_s1116
                                 - 0.354999669772643 z_k1_s1117
                                 - 0.648001908441141 z_k1_s1120
                                 - 0.807001297964222 z_k1_s1121
                                 - 0.332000546280841 z_k1_s1122
                                 - 0.367000433609858 z_k1_s1123
                                 - 0.255000535256565 z_k1_s1124
                                 - 0.681999851071741 z_k1_s1125
                                 - 0.381000474841611 z_k1_s1126
                                 - 0.84300140398873 z_k1_s1127
                                 - 0.257000067798048 z_k1_s1130
                                 - 0.16900012580506 z_k1_s1131
                                 - 0.0580000837591921 z_k1_s1132
                                 - 0.171000056713893 z_k1_s1133
                                 - 0.0770001231941173 z_k1_s1134
                                 - 0.135000306353454 z_k1_s1135
                                 - 0.0670000710493742 z_k1_s1136
                                 - 0.0600001140681696 z_k1_s1137
                                 - 0.438999831811895 z_k1_s1140
                                 - 0.144000170675217 z_k1_s1141
                                 - 0.145000157559058 z_k1_s1142
                                 - 0.186999606680345 z_k1_s1143
                                 - 0.0950000778662515 z_k1_s1144
                                 - 0.144000170675217 z_k1_s1145
                                 - 0.227000553035546 z_k1_s1146
                                 - 0.0900001434470507 z_k1_s1147
                                 - 0.174999877066909 z_k1_s1150
                                 - 0.401998982273584 z_k1_s1151
                                 - 0.348000087728017 z_k1_s1152
                                 - 0.401998982273584 z_k1_s1153
                                 - 0.607000063818822 z_k1_s1154
                                 - 0.501999234539938 z_k1_s1155
                                 - 0.189999632707256 z_k1_s1156
                                 - 0.19599976072308 z_k1_s1157
                                 - 0.186000129253949 z_k1_s1160
                                 - 0.213000062919833 z_k1_s1161
                                 - 0.0979999986058345 z_k1_s1162
                                 - 0.117999998321311 z_k1_s1163
                                 - 0.208999535376666 z_k1_s1164
                                 - 0.0710000649399503 z_k1_s1165
                                 - 0.348999533385001 z_k1_s1166
                                 - 0.0750000648830455 z_k1_s1167
                                 - 0.541002487080917 z_k1_s1170
                                 - 0.252001006276456 z_k1_s1171
                                 - 0.285001566356033 z_k1_s1172
                                 - 0.253000918263298 z_k1_s1173
                                 - 0.672003749408133 z_k1_s1174
                                 - 0.29400130723973 z_k1_s1175
                                 - 0.194000781018408 z_k1_s1176
                                 - 0.174000675774144 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.987721 w_k1_ADM_s000
                                 + 0.858589 w_k1_AXP_s000
                                 + 0.825673 w_k1_DIS_s000
                                 + 0.853882 w_k1_FDX_s000
                                 + 1.13699 w_k1_IBM_s000
                                 + 0.729933 w_k1_INTC_s000
                                 + 1.00269 w_k1_KO_s000 + 1.0953 w_k1_MAT_s000
                                 + 0.909681 w_k1_MSFT_s000
                                 + 1.02666 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.14805 w_k1_ADM_s000
                                 + 1.28817 w_k1_AXP_s000
                                 + 1.27533 w_k1_DIS_s000
                                 + 1.16504 w_k1_FDX_s000
                                 + 1.14575 w_k1_IBM_s000
                                 + 0.948727 w_k1_INTC_s000
                                 + 1.27134 w_k1_KO_s000 + 1.2139 w_k1_MAT_s000
                                 + 1.09346 w_k1_MSFT_s000 + 1.1295 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.04051 w_k1_ADM_s000
                                 + 0.791523 w_k1_AXP_s000
                                 + 0.805034 w_k1_DIS_s000
                                 + 0.964255 w_k1_FDX_s000
                                 + 0.656319 w_k1_IBM_s000
                                 + 0.960677 w_k1_INTC_s000
                                 + 0.917592 w_k1_KO_s000 + 0.4649 w_k1_MAT_s000
                                 + 0.856759 w_k1_MSFT_s000
                                 + 0.943594 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.18057 w_k1_ADM_s000
                                 + 1.42055 w_k1_AXP_s000
                                 + 1.59381 w_k1_DIS_s000
                                 + 1.43915 w_k1_FDX_s000
                                 + 1.06292 w_k1_IBM_s000
                                 + 1.25125 w_k1_INTC_s000
                                 + 1.14249 w_k1_KO_s000 + 1.15163 w_k1_MAT_s000
                                 + 1.21186 w_k1_MSFT_s000 + 1.0825 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.875199 w_k1_ADM_s000
                                 + 1.12327 w_k1_AXP_s000
                                 + 0.936813 w_k1_DIS_s000
                                 + 0.993292 w_k1_FDX_s000
                                 + 0.893988 w_k1_IBM_s000
                                 + 1.07412 w_k1_INTC_s000
                                 + 0.858011 w_k1_KO_s000 + 1.1906 w_k1_MAT_s000
                                 + 0.960952 w_k1_MSFT_s000
                                 + 1.05193 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.943419 w_k1_ADM_s000
                                 + 1.03798 w_k1_AXP_s000
                                 + 1.04956 w_k1_DIS_s000 + 1.0231 w_k1_FDX_s000
                                 + 1.51562 w_k1_IBM_s000
                                 + 1.04537 w_k1_INTC_s000
                                 + 1.24938 w_k1_KO_s000 + 1.52951 w_k1_MAT_s000
                                 + 1.03005 w_k1_MSFT_s000
                                 + 1.08974 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.875697 w_k1_ADM_s000
                                 + 0.833334 w_k1_AXP_s000
                                 + 0.82264 w_k1_DIS_s000
                                 + 0.899655 w_k1_FDX_s000
                                 + 0.906934 w_k1_IBM_s000
                                 + 0.864107 w_k1_INTC_s000
                                 + 0.937087 w_k1_KO_s000
                                 + 0.795145 w_k1_MAT_s000
                                 + 0.849666 w_k1_MSFT_s000
                                 + 0.909235 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.773167 w_k1_ADM_s000
                                 + 0.619623 w_k1_AXP_s000
                                 + 0.663368 w_k1_DIS_s000
                                 + 0.692885 w_k1_FDX_s000
                                 + 0.850014 w_k1_IBM_s000
                                 + 0.671464 w_k1_INTC_s000
                                 + 0.83136 w_k1_KO_s000
                                 + 0.793775 w_k1_MAT_s000
                                 + 0.711968 w_k1_MSFT_s000
                                 + 0.792148 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.862441 w_k1_ADM_s001
                                 + 0.647444 w_k1_AXP_s001
                                 + 0.718145 w_k1_DIS_s001
                                 + 0.808679 w_k1_FDX_s001
                                 + 0.590827 w_k1_IBM_s001
                                 + 1.06644 w_k1_INTC_s001
                                 + 0.705363 w_k1_KO_s001
                                 + 0.519787 w_k1_MAT_s001
                                 + 0.830781 w_k1_MSFT_s001
                                 + 0.725338 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.08431 w_k1_ADM_s001
                                 + 1.74203 w_k1_AXP_s001
                                 + 1.90407 w_k1_DIS_s001
                                 + 1.44914 w_k1_FDX_s001
                                 + 1.58554 w_k1_IBM_s001
                                 + 1.63874 w_k1_INTC_s001
                                 + 1.60842 w_k1_KO_s001 + 2.23729 w_k1_MAT_s001
                                 + 1.33447 w_k1_MSFT_s001
                                 + 1.16428 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.814189 w_k1_ADM_s001
                                 + 0.991513 w_k1_AXP_s001
                                 + 1.08336 w_k1_DIS_s001
                                 + 1.14102 w_k1_FDX_s001
                                 + 0.714495 w_k1_IBM_s001
                                 + 1.17505 w_k1_INTC_s001
                                 + 0.853642 w_k1_KO_s001
                                 + 0.72305 w_k1_MAT_s001
                                 + 1.07908 w_k1_MSFT_s001
                                 + 0.957632 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.795863 w_k1_ADM_s001
                                 + 0.725827 w_k1_AXP_s001
                                 + 0.666363 w_k1_DIS_s001
                                 + 0.757551 w_k1_FDX_s001
                                 + 0.783574 w_k1_IBM_s001
                                 + 0.872197 w_k1_INTC_s001
                                 + 0.764923 w_k1_KO_s001
                                 + 0.885929 w_k1_MAT_s001
                                 + 0.823331 w_k1_MSFT_s001
                                 + 0.85402 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.857802 w_k1_ADM_s001
                                 + 0.718129 w_k1_AXP_s001
                                 + 0.696973 w_k1_DIS_s001
                                 + 0.773161 w_k1_FDX_s001
                                 + 1.13723 w_k1_IBM_s001
                                 + 0.876236 w_k1_INTC_s001
                                 + 0.714482 w_k1_KO_s001
                                 + 1.03397 w_k1_MAT_s001
                                 + 0.704055 w_k1_MSFT_s001
                                 + 0.810469 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.876281 w_k1_ADM_s001
                                 + 1.05631 w_k1_AXP_s001
                                 + 0.685421 w_k1_DIS_s001
                                 + 0.969949 w_k1_FDX_s001
                                 + 0.874705 w_k1_IBM_s001
                                 + 1.19822 w_k1_INTC_s001
                                 + 0.754022 w_k1_KO_s001
                                 + 0.96455 w_k1_MAT_s001
                                 + 0.884212 w_k1_MSFT_s001
                                 + 0.850024 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.806505 w_k1_ADM_s001
                                 + 1.19988 w_k1_AXP_s001
                                 + 1.08318 w_k1_DIS_s001
                                 + 1.02419 w_k1_FDX_s001
                                 + 1.20934 w_k1_IBM_s001
                                 + 1.25364 w_k1_INTC_s001
                                 + 1.11226 w_k1_KO_s001 + 1.6158 w_k1_MAT_s001
                                 + 1.00663 w_k1_MSFT_s001
                                 + 1.14867 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.850694 w_k1_ADM_s001
                                 + 0.754097 w_k1_AXP_s001
                                 + 1.20618 w_k1_DIS_s001
                                 + 0.917256 w_k1_FDX_s001
                                 + 1.28938 w_k1_IBM_s001
                                 + 0.963955 w_k1_INTC_s001
                                 + 1.08685 w_k1_KO_s001 + 1.57671 w_k1_MAT_s001
                                 + 1.09915 w_k1_MSFT_s001
                                 + 0.935708 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.05674 w_k1_ADM_s002
                                 + 1.14572 w_k1_AXP_s002
                                 + 1.26649 w_k1_DIS_s002 + 1.062 w_k1_FDX_s002
                                 + 1.3886 w_k1_IBM_s002
                                 + 1.25685 w_k1_INTC_s002
                                 + 1.41093 w_k1_KO_s002 + 1.64388 w_k1_MAT_s002
                                 + 1.26725 w_k1_MSFT_s002
                                 + 1.14979 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.886444 w_k1_ADM_s002
                                 + 0.528024 w_k1_AXP_s002
                                 + 0.733168 w_k1_DIS_s002
                                 + 0.778755 w_k1_FDX_s002
                                 + 0.856144 w_k1_IBM_s002
                                 + 0.842579 w_k1_INTC_s002
                                 + 0.859847 w_k1_KO_s002
                                 + 0.747295 w_k1_MAT_s002
                                 + 0.975635 w_k1_MSFT_s002
                                 + 0.855121 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.02845 w_k1_ADM_s002
                                 + 0.734455 w_k1_AXP_s002
                                 + 0.941201 w_k1_DIS_s002
                                 + 0.888146 w_k1_FDX_s002
                                 + 1.18688 w_k1_IBM_s002
                                 + 0.935834 w_k1_INTC_s002
                                 + 1.07866 w_k1_KO_s002 + 1.10992 w_k1_MAT_s002
                                 + 1.00546 w_k1_MSFT_s002
                                 + 0.934362 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.57279 w_k1_ADM_s002
                                 + 1.24384 w_k1_AXP_s002
                                 + 1.64313 w_k1_DIS_s002
                                 + 1.42397 w_k1_FDX_s002
                                 + 1.08591 w_k1_IBM_s002
                                 + 1.06283 w_k1_INTC_s002
                                 + 1.44886 w_k1_KO_s002 + 1.4324 w_k1_MAT_s002
                                 + 1.48276 w_k1_MSFT_s002
                                 + 1.11132 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.32759 w_k1_ADM_s002
                                 + 1.13669 w_k1_AXP_s002
                                 + 1.05251 w_k1_DIS_s002
                                 + 1.15585 w_k1_FDX_s002
                                 + 1.07862 w_k1_IBM_s002
                                 + 1.15688 w_k1_INTC_s002
                                 + 1.12306 w_k1_KO_s002 + 1.04548 w_k1_MAT_s002
                                 + 1.20994 w_k1_MSFT_s002
                                 + 1.09029 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.969557 w_k1_ADM_s002
                                 + 0.79047 w_k1_AXP_s002
                                 + 0.697968 w_k1_DIS_s002
                                 + 0.862155 w_k1_FDX_s002
                                 + 0.871725 w_k1_IBM_s002
                                 + 0.753337 w_k1_INTC_s002
                                 + 0.900401 w_k1_KO_s002
                                 + 0.974507 w_k1_MAT_s002
                                 + 0.85918 w_k1_MSFT_s002
                                 + 0.838234 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.24883 w_k1_ADM_s002
                                 + 0.874656 w_k1_AXP_s002
                                 + 1.10195 w_k1_DIS_s002
                                 + 1.03928 w_k1_FDX_s002
                                 + 0.946658 w_k1_IBM_s002
                                 + 1.02042 w_k1_INTC_s002
                                 + 1.10278 w_k1_KO_s002
                                 + 0.878799 w_k1_MAT_s002
                                 + 1.18679 w_k1_MSFT_s002
                                 + 0.894263 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.851203 w_k1_ADM_s002
                                 + 0.524246 w_k1_AXP_s002
                                 + 0.819387 w_k1_DIS_s002
                                 + 0.840145 w_k1_FDX_s002
                                 + 0.629635 w_k1_IBM_s002
                                 + 0.77766 w_k1_INTC_s002
                                 + 0.836101 w_k1_KO_s002
                                 + 0.534267 w_k1_MAT_s002
                                 + 0.974486 w_k1_MSFT_s002
                                 + 0.838183 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.827064 w_k1_ADM_s003
                                 + 0.778105 w_k1_AXP_s003
                                 + 0.652582 w_k1_DIS_s003
                                 + 0.846225 w_k1_FDX_s003
                                 + 0.640959 w_k1_IBM_s003
                                 + 0.906649 w_k1_INTC_s003
                                 + 0.680512 w_k1_KO_s003
                                 + 0.393941 w_k1_MAT_s003
                                 + 0.782328 w_k1_MSFT_s003
                                 + 0.770884 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.929056 w_k1_ADM_s003
                                 + 1.07033 w_k1_AXP_s003
                                 + 1.34334 w_k1_DIS_s003
                                 + 1.12909 w_k1_FDX_s003
                                 + 0.951871 w_k1_IBM_s003
                                 + 1.18024 w_k1_INTC_s003
                                 + 1.26024 w_k1_KO_s003 + 1.00088 w_k1_MAT_s003
                                 + 1.25068 w_k1_MSFT_s003
                                 + 1.06821 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.746928 w_k1_ADM_s003
                                 + 0.829973 w_k1_AXP_s003
                                 + 0.705381 w_k1_DIS_s003
                                 + 0.759681 w_k1_FDX_s003
                                 + 1.00593 w_k1_IBM_s003
                                 + 0.815453 w_k1_INTC_s003
                                 + 0.826256 w_k1_KO_s003
                                 + 0.856359 w_k1_MAT_s003
                                 + 0.844987 w_k1_MSFT_s003
                                 + 0.903618 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.984122 w_k1_ADM_s003
                                 + 0.91661 w_k1_AXP_s003
                                 + 0.766931 w_k1_DIS_s003
                                 + 0.937918 w_k1_FDX_s003
                                 + 0.8269 w_k1_IBM_s003
                                 + 0.669388 w_k1_INTC_s003
                                 + 0.694702 w_k1_KO_s003
                                 + 0.688885 w_k1_MAT_s003
                                 + 0.833353 w_k1_MSFT_s003
                                 + 0.868549 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.763294 w_k1_ADM_s003
                                 + 1.1102 w_k1_AXP_s003 + 0.90493 w_k1_DIS_s003
                                 + 0.863379 w_k1_FDX_s003
                                 + 1.34725 w_k1_IBM_s003
                                 + 1.0076 w_k1_INTC_s003 + 1.16832 w_k1_KO_s003
                                 + 1.24329 w_k1_MAT_s003
                                 + 0.956404 w_k1_MSFT_s003
                                 + 0.974347 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.08632 w_k1_ADM_s003
                                 + 1.17588 w_k1_AXP_s003 + 1.2424 w_k1_DIS_s003
                                 + 1.04729 w_k1_FDX_s003
                                 + 1.39153 w_k1_IBM_s003
                                 + 1.02761 w_k1_INTC_s003
                                 + 1.26945 w_k1_KO_s003 + 1.03408 w_k1_MAT_s003
                                 + 1.05269 w_k1_MSFT_s003
                                 + 1.03423 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.73372 w_k1_ADM_s003
                                 + 0.909237 w_k1_AXP_s003
                                 + 0.974424 w_k1_DIS_s003
                                 + 0.904391 w_k1_FDX_s003
                                 + 0.809484 w_k1_IBM_s003
                                 + 1.02372 w_k1_INTC_s003
                                 + 0.923286 w_k1_KO_s003
                                 + 0.504194 w_k1_MAT_s003
                                 + 0.980319 w_k1_MSFT_s003
                                 + 0.915179 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.992707 w_k1_ADM_s003
                                 + 1.42718 w_k1_AXP_s003
                                 + 1.04903 w_k1_DIS_s003 + 1.2315 w_k1_FDX_s003
                                 + 0.909615 w_k1_IBM_s003
                                 + 1.32434 w_k1_INTC_s003
                                 + 0.970249 w_k1_KO_s003
                                 + 0.923407 w_k1_MAT_s003
                                 + 1.16327 w_k1_MSFT_s003
                                 + 1.04787 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.756528 w_k1_ADM_s004
                                 + 0.608878 w_k1_AXP_s004 + 0.562 w_k1_DIS_s004
                                 + 0.719142 w_k1_FDX_s004
                                 + 0.702802 w_k1_IBM_s004
                                 + 0.775773 w_k1_INTC_s004
                                 + 0.586763 w_k1_KO_s004
                                 + 0.689429 w_k1_MAT_s004
                                 + 0.70954 w_k1_MSFT_s004
                                 + 0.778846 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.27782 w_k1_ADM_s004
                                 + 1.24938 w_k1_AXP_s004 + 1.5031 w_k1_DIS_s004
                                 + 1.28452 w_k1_FDX_s004
                                 + 0.826214 w_k1_IBM_s004
                                 + 1.39297 w_k1_INTC_s004 + 1.1484 w_k1_KO_s004
                                 + 1.04202 w_k1_MAT_s004
                                 + 1.27972 w_k1_MSFT_s004
                                 + 1.37174 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.854872 w_k1_ADM_s004
                                 + 1.23252 w_k1_AXP_s004
                                 + 1.12586 w_k1_DIS_s004
                                 + 1.05433 w_k1_FDX_s004
                                 + 1.00581 w_k1_IBM_s004
                                 + 1.24805 w_k1_INTC_s004
                                 + 0.919657 w_k1_KO_s004 + 1.1018 w_k1_MAT_s004
                                 + 1.09692 w_k1_MSFT_s004
                                 + 1.12953 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.31186 w_k1_ADM_s004
                                 + 1.25971 w_k1_AXP_s004 + 1.4512 w_k1_DIS_s004
                                 + 1.14162 w_k1_FDX_s004
                                 + 1.45085 w_k1_IBM_s004
                                 + 1.34263 w_k1_INTC_s004 + 1.2822 w_k1_KO_s004
                                 + 1.11252 w_k1_MAT_s004
                                 + 1.08748 w_k1_MSFT_s004
                                 + 1.22552 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.896915 w_k1_ADM_s004
                                 + 0.834112 w_k1_AXP_s004
                                 + 0.854831 w_k1_DIS_s004
                                 + 0.813051 w_k1_FDX_s004
                                 + 1.10623 w_k1_IBM_s004
                                 + 1.15802 w_k1_INTC_s004
                                 + 0.980798 w_k1_KO_s004
                                 + 0.892111 w_k1_MAT_s004
                                 + 0.958618 w_k1_MSFT_s004
                                 + 0.935034 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.00883 w_k1_ADM_s004
                                 + 0.847404 w_k1_AXP_s004
                                 + 0.820193 w_k1_DIS_s004
                                 + 0.874125 w_k1_FDX_s004
                                 + 0.833828 w_k1_IBM_s004
                                 + 1.06041 w_k1_INTC_s004
                                 + 0.814232 w_k1_KO_s004
                                 + 0.717172 w_k1_MAT_s004
                                 + 0.829135 w_k1_MSFT_s004
                                 + 0.957921 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.03655 w_k1_ADM_s004
                                 + 0.979108 w_k1_AXP_s004
                                 + 0.761981 w_k1_DIS_s004
                                 + 0.969454 w_k1_FDX_s004
                                 + 0.669108 w_k1_IBM_s004
                                 + 0.881317 w_k1_INTC_s004
                                 + 0.753467 w_k1_KO_s004
                                 + 0.672822 w_k1_MAT_s004
                                 + 0.977469 w_k1_MSFT_s004
                                 + 0.987855 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.32151 w_k1_ADM_s004
                                 + 1.01906 w_k1_AXP_s004
                                 + 1.41908 w_k1_DIS_s004
                                 + 1.05476 w_k1_FDX_s004
                                 + 0.972177 w_k1_IBM_s004
                                 + 1.22511 w_k1_INTC_s004
                                 + 1.09397 w_k1_KO_s004
                                 + 0.821518 w_k1_MAT_s004
                                 + 1.11065 w_k1_MSFT_s004
                                 + 1.05291 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.03476 w_k1_ADM_s005
                                 + 0.904585 w_k1_AXP_s005
                                 + 1.00349 w_k1_DIS_s005
                                 + 0.843297 w_k1_FDX_s005
                                 + 1.24297 w_k1_IBM_s005
                                 + 0.886647 w_k1_INTC_s005
                                 + 0.989041 w_k1_KO_s005
                                 + 1.05356 w_k1_MAT_s005
                                 + 0.914753 w_k1_MSFT_s005
                                 + 1.02518 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.764341 w_k1_ADM_s005
                                 + 0.755389 w_k1_AXP_s005
                                 + 0.479061 w_k1_DIS_s005
                                 + 0.645627 w_k1_FDX_s005
                                 + 0.728441 w_k1_IBM_s005
                                 + 0.713302 w_k1_INTC_s005
                                 + 0.565161 w_k1_KO_s005
                                 + 0.676008 w_k1_MAT_s005
                                 + 0.540927 w_k1_MSFT_s005
                                 + 0.680374 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.7669 w_k1_ADM_s005
                                 + 0.869379 w_k1_AXP_s005
                                 + 0.846214 w_k1_DIS_s005
                                 + 0.878889 w_k1_FDX_s005
                                 + 0.731799 w_k1_IBM_s005
                                 + 0.932608 w_k1_INTC_s005
                                 + 0.838382 w_k1_KO_s005
                                 + 0.723264 w_k1_MAT_s005
                                 + 0.850005 w_k1_MSFT_s005
                                 + 0.911977 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.15051 w_k1_ADM_s005
                                 + 1.51074 w_k1_AXP_s005 + 1.3215 w_k1_DIS_s005
                                 + 1.00718 w_k1_FDX_s005 + 1.7009 w_k1_IBM_s005
                                 + 1.07806 w_k1_INTC_s005
                                 + 1.16917 w_k1_KO_s005 + 2.25928 w_k1_MAT_s005
                                 + 1.02442 w_k1_MSFT_s005
                                 + 1.14227 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.21104 w_k1_ADM_s005
                                 + 1.13097 w_k1_AXP_s005
                                 + 1.40003 w_k1_DIS_s005
                                 + 1.05628 w_k1_FDX_s005
                                 + 0.979808 w_k1_IBM_s005
                                 + 1.04503 w_k1_INTC_s005
                                 + 1.13676 w_k1_KO_s005 + 1.09461 w_k1_MAT_s005
                                 + 1.11382 w_k1_MSFT_s005
                                 + 1.12039 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.05758 w_k1_ADM_s005
                                 + 0.964118 w_k1_AXP_s005
                                 + 0.966075 w_k1_DIS_s005
                                 + 0.98623 w_k1_FDX_s005
                                 + 0.562007 w_k1_IBM_s005
                                 + 0.881031 w_k1_INTC_s005
                                 + 0.698387 w_k1_KO_s005
                                 + 0.676114 w_k1_MAT_s005
                                 + 1.0003 w_k1_MSFT_s005
                                 + 0.870049 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.00627 w_k1_ADM_s005
                                 + 1.17661 w_k1_AXP_s005
                                 + 0.840282 w_k1_DIS_s005
                                 + 0.950808 w_k1_FDX_s005
                                 + 0.983389 w_k1_IBM_s005
                                 + 0.87493 w_k1_INTC_s005
                                 + 0.811858 w_k1_KO_s005
                                 + 0.930144 w_k1_MAT_s005
                                 + 0.715938 w_k1_MSFT_s005
                                 + 0.919901 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.31344 w_k1_ADM_s005
                                 + 1.50328 w_k1_AXP_s005
                                 + 1.43351 w_k1_DIS_s005
                                 + 1.17175 w_k1_FDX_s005
                                 + 1.28205 w_k1_IBM_s005
                                 + 1.36477 w_k1_INTC_s005
                                 + 1.10671 w_k1_KO_s005 + 1.1606 w_k1_MAT_s005
                                 + 1.15476 w_k1_MSFT_s005
                                 + 1.19582 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.27849 w_k1_ADM_s006
                                 + 1.51989 w_k1_AXP_s006
                                 + 1.31186 w_k1_DIS_s006
                                 + 1.26307 w_k1_FDX_s006
                                 + 1.11536 w_k1_IBM_s006
                                 + 1.37699 w_k1_INTC_s006
                                 + 1.27659 w_k1_KO_s006 + 1.00735 w_k1_MAT_s006
                                 + 1.0742 w_k1_MSFT_s006 + 1.17843 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.22629 w_k1_ADM_s006
                                 + 1.27493 w_k1_AXP_s006
                                 + 1.41545 w_k1_DIS_s006
                                 + 1.11459 w_k1_FDX_s006
                                 + 1.53494 w_k1_IBM_s006
                                 + 1.25051 w_k1_INTC_s006 + 1.3905 w_k1_KO_s006
                                 + 1.7963 w_k1_MAT_s006 + 1.1 w_k1_MSFT_s006
                                 + 1.24228 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.93677 w_k1_ADM_s006
                                 + 0.921717 w_k1_AXP_s006
                                 + 0.754681 w_k1_DIS_s006
                                 + 0.982849 w_k1_FDX_s006
                                 + 0.77909 w_k1_IBM_s006
                                 + 0.981171 w_k1_INTC_s006
                                 + 0.774794 w_k1_KO_s006
                                 + 0.765057 w_k1_MAT_s006
                                 + 0.883039 w_k1_MSFT_s006
                                 + 0.952203 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.795507 w_k1_ADM_s006
                                 + 0.661127 w_k1_AXP_s006
                                 + 0.598628 w_k1_DIS_s006
                                 + 0.728551 w_k1_FDX_s006
                                 + 0.855507 w_k1_IBM_s006
                                 + 0.761069 w_k1_INTC_s006
                                 + 0.750889 w_k1_KO_s006
                                 + 0.669696 w_k1_MAT_s006
                                 + 0.669977 w_k1_MSFT_s006
                                 + 0.73977 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.923219 w_k1_ADM_s006
                                 + 0.940377 w_k1_AXP_s006
                                 + 1.16205 w_k1_DIS_s006
                                 + 1.03353 w_k1_FDX_s006
                                 + 1.12164 w_k1_IBM_s006
                                 + 1.22203 w_k1_INTC_s006
                                 + 1.09855 w_k1_KO_s006
                                 + 0.993449 w_k1_MAT_s006
                                 + 1.10834 w_k1_MSFT_s006 + 1.0761 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.14534 w_k1_ADM_s006
                                 + 0.956902 w_k1_AXP_s006
                                 + 1.42516 w_k1_DIS_s006
                                 + 1.25565 w_k1_FDX_s006
                                 + 0.970191 w_k1_IBM_s006
                                 + 1.17781 w_k1_INTC_s006
                                 + 1.13267 w_k1_KO_s006
                                 + 0.555712 w_k1_MAT_s006
                                 + 1.12051 w_k1_MSFT_s006
                                 + 0.965372 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.36176 w_k1_ADM_s006
                                 + 1.46656 w_k1_AXP_s006
                                 + 1.64159 w_k1_DIS_s006
                                 + 1.51791 w_k1_FDX_s006
                                 + 1.18298 w_k1_IBM_s006
                                 + 1.51665 w_k1_INTC_s006
                                 + 1.14528 w_k1_KO_s006 + 1.21818 w_k1_MAT_s006
                                 + 1.37893 w_k1_MSFT_s006
                                 + 1.21221 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.932283 w_k1_ADM_s006
                                 + 0.904644 w_k1_AXP_s006
                                 + 0.826001 w_k1_DIS_s006
                                 + 0.856341 w_k1_FDX_s006
                                 + 1.04411 w_k1_IBM_s006
                                 + 0.942927 w_k1_INTC_s006
                                 + 0.915968 w_k1_KO_s006
                                 + 0.965921 w_k1_MAT_s006
                                 + 0.783257 w_k1_MSFT_s006
                                 + 0.811578 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.780492 w_k1_ADM_s007
                                 + 0.97424 w_k1_AXP_s007
                                 + 0.779908 w_k1_DIS_s007
                                 + 0.862721 w_k1_FDX_s007
                                 + 0.993978 w_k1_IBM_s007
                                 + 1.06761 w_k1_INTC_s007
                                 + 0.921647 w_k1_KO_s007
                                 + 1.06371 w_k1_MAT_s007
                                 + 0.852018 w_k1_MSFT_s007
                                 + 0.949388 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.968774 w_k1_ADM_s007
                                 + 0.999856 w_k1_AXP_s007
                                 + 1.19091 w_k1_DIS_s007
                                 + 1.01553 w_k1_FDX_s007
                                 + 1.28348 w_k1_IBM_s007
                                 + 1.26741 w_k1_INTC_s007 + 1.2858 w_k1_KO_s007
                                 + 1.01199 w_k1_MAT_s007
                                 + 1.11007 w_k1_MSFT_s007
                                 + 1.08272 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.728315 w_k1_ADM_s007
                                 + 0.762601 w_k1_AXP_s007
                                 + 0.723625 w_k1_DIS_s007
                                 + 0.849211 w_k1_FDX_s007
                                 + 0.74986 w_k1_IBM_s007
                                 + 0.986008 w_k1_INTC_s007
                                 + 0.753791 w_k1_KO_s007
                                 + 0.63951 w_k1_MAT_s007
                                 + 0.841514 w_k1_MSFT_s007
                                 + 0.900514 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.768773 w_k1_ADM_s007
                                 + 1.11739 w_k1_AXP_s007
                                 + 1.06698 w_k1_DIS_s007
                                 + 0.98415 w_k1_FDX_s007
                                 + 1.43781 w_k1_IBM_s007
                                 + 1.36193 w_k1_INTC_s007
                                 + 1.13465 w_k1_KO_s007 + 1.46063 w_k1_MAT_s007
                                 + 1.02379 w_k1_MSFT_s007
                                 + 1.13032 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.04957 w_k1_ADM_s007
                                 + 1.49794 w_k1_AXP_s007
                                 + 1.66833 w_k1_DIS_s007
                                 + 1.30647 w_k1_FDX_s007
                                 + 1.86523 w_k1_IBM_s007
                                 + 1.84998 w_k1_INTC_s007
                                 + 1.59682 w_k1_KO_s007 + 1.55187 w_k1_MAT_s007
                                 + 1.29861 w_k1_MSFT_s007
                                 + 1.43303 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.807906 w_k1_ADM_s007
                                 + 0.989582 w_k1_AXP_s007
                                 + 1.53882 w_k1_DIS_s007 + 1.2067 w_k1_FDX_s007
                                 + 1.34511 w_k1_IBM_s007
                                 + 1.43546 w_k1_INTC_s007 + 1.2163 w_k1_KO_s007
                                 + 1.12225 w_k1_MAT_s007
                                 + 1.21858 w_k1_MSFT_s007
                                 + 1.19813 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.884557 w_k1_ADM_s007
                                 + 0.847276 w_k1_AXP_s007
                                 + 1.07559 w_k1_DIS_s007
                                 + 1.00089 w_k1_FDX_s007
                                 + 0.839689 w_k1_IBM_s007
                                 + 1.20413 w_k1_INTC_s007
                                 + 0.900855 w_k1_KO_s007
                                 + 0.734782 w_k1_MAT_s007
                                 + 0.987276 w_k1_MSFT_s007
                                 + 1.08783 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.12358 w_k1_ADM_s007
                                 + 1.54804 w_k1_AXP_s007 + 1.3399 w_k1_DIS_s007
                                 + 1.30224 w_k1_FDX_s007 + 1.0522 w_k1_IBM_s007
                                 + 1.58314 w_k1_INTC_s007
                                 + 1.09265 w_k1_KO_s007 + 0.99481 w_k1_MAT_s007
                                 + 1.18164 w_k1_MSFT_s007
                                 + 1.13582 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.03324 w_k1_ADM_s010
                                 + 0.866249 w_k1_AXP_s010
                                 + 0.713382 w_k1_DIS_s010
                                 + 0.806905 w_k1_FDX_s010
                                 + 0.75706 w_k1_IBM_s010
                                 + 0.880837 w_k1_INTC_s010
                                 + 0.846663 w_k1_KO_s010
                                 + 0.725143 w_k1_MAT_s010
                                 + 0.770981 w_k1_MSFT_s010
                                 + 0.9388 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.921065 w_k1_ADM_s010
                                 + 1.14163 w_k1_AXP_s010
                                 + 0.829217 w_k1_DIS_s010
                                 + 0.816732 w_k1_FDX_s010
                                 + 1.23666 w_k1_IBM_s010
                                 + 0.916025 w_k1_INTC_s010
                                 + 0.907277 w_k1_KO_s010
                                 + 1.72515 w_k1_MAT_s010
                                 + 0.800093 w_k1_MSFT_s010
                                 + 0.879009 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.17813 w_k1_ADM_s010
                                 + 1.54727 w_k1_AXP_s010
                                 + 1.16325 w_k1_DIS_s010
                                 + 1.23949 w_k1_FDX_s010
                                 + 0.745692 w_k1_IBM_s010
                                 + 1.10155 w_k1_INTC_s010
                                 + 0.792711 w_k1_KO_s010
                                 + 1.16612 w_k1_MAT_s010
                                 + 0.999966 w_k1_MSFT_s010
                                 + 1.08404 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.988939 w_k1_ADM_s010
                                 + 0.88527 w_k1_AXP_s010
                                 + 1.04931 w_k1_DIS_s010
                                 + 0.929922 w_k1_FDX_s010
                                 + 0.923716 w_k1_IBM_s010
                                 + 0.875976 w_k1_INTC_s010
                                 + 1.03414 w_k1_KO_s010
                                 + 0.957623 w_k1_MAT_s010
                                 + 1.02906 w_k1_MSFT_s010 + 1.015 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.2337 w_k1_ADM_s010
                                 + 1.40482 w_k1_AXP_s010
                                 + 1.25226 w_k1_DIS_s010
                                 + 1.23917 w_k1_FDX_s010
                                 + 1.25687 w_k1_IBM_s010
                                 + 1.25398 w_k1_INTC_s010 + 1.1466 w_k1_KO_s010
                                 + 1.00177 w_k1_MAT_s010
                                 + 1.03665 w_k1_MSFT_s010
                                 + 1.24262 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.82596 w_k1_ADM_s010
                                 + 1.02248 w_k1_AXP_s010
                                 + 1.19106 w_k1_DIS_s010
                                 + 0.930041 w_k1_FDX_s010
                                 + 1.29413 w_k1_IBM_s010
                                 + 0.956254 w_k1_INTC_s010
                                 + 0.956996 w_k1_KO_s010
                                 + 1.20484 w_k1_MAT_s010
                                 + 0.928646 w_k1_MSFT_s010
                                 + 1.02561 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.42025 w_k1_ADM_s010
                                 + 1.5387 w_k1_AXP_s010 + 1.93939 w_k1_DIS_s010
                                 + 1.35889 w_k1_FDX_s010
                                 + 1.40001 w_k1_IBM_s010
                                 + 1.15844 w_k1_INTC_s010
                                 + 1.42264 w_k1_KO_s010 + 1.75905 w_k1_MAT_s010
                                 + 1.28097 w_k1_MSFT_s010
                                 + 1.22696 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.23605 w_k1_ADM_s010
                                 + 1.44852 w_k1_AXP_s010 + 1.077 w_k1_DIS_s010
                                 + 1.09887 w_k1_FDX_s010
                                 + 1.10817 w_k1_IBM_s010
                                 + 1.35197 w_k1_INTC_s010
                                 + 1.05887 w_k1_KO_s010 + 1.10083 w_k1_MAT_s010
                                 + 0.945165 w_k1_MSFT_s010
                                 + 1.09459 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.07063 w_k1_ADM_s011
                                 + 1.29433 w_k1_AXP_s011
                                 + 1.12592 w_k1_DIS_s011
                                 + 1.04646 w_k1_FDX_s011
                                 + 1.27163 w_k1_IBM_s011
                                 + 0.938293 w_k1_INTC_s011
                                 + 0.992656 w_k1_KO_s011
                                 + 0.813275 w_k1_MAT_s011
                                 + 0.905703 w_k1_MSFT_s011
                                 + 0.953654 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.846485 w_k1_ADM_s011
                                 + 0.742692 w_k1_AXP_s011
                                 + 0.612372 w_k1_DIS_s011
                                 + 0.750214 w_k1_FDX_s011
                                 + 0.774799 w_k1_IBM_s011
                                 + 0.891405 w_k1_INTC_s011
                                 + 0.604321 w_k1_KO_s011
                                 + 0.454631 w_k1_MAT_s011
                                 + 0.725908 w_k1_MSFT_s011
                                 + 0.752435 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.11381 w_k1_ADM_s011
                                 + 0.968118 w_k1_AXP_s011
                                 + 0.902044 w_k1_DIS_s011
                                 + 1.00555 w_k1_FDX_s011
                                 + 0.663928 w_k1_IBM_s011
                                 + 0.969606 w_k1_INTC_s011
                                 + 0.806669 w_k1_KO_s011
                                 + 0.433638 w_k1_MAT_s011
                                 + 0.869245 w_k1_MSFT_s011
                                 + 0.878696 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.10235 w_k1_ADM_s011
                                 + 0.952554 w_k1_AXP_s011
                                 + 1.03978 w_k1_DIS_s011
                                 + 1.05962 w_k1_FDX_s011
                                 + 0.942299 w_k1_IBM_s011
                                 + 0.889038 w_k1_INTC_s011
                                 + 0.867134 w_k1_KO_s011
                                 + 0.65734 w_k1_MAT_s011
                                 + 1.01634 w_k1_MSFT_s011
                                 + 0.933062 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.699952 w_k1_ADM_s011
                                 + 0.781321 w_k1_AXP_s011
                                 + 0.850407 w_k1_DIS_s011
                                 + 0.882881 w_k1_FDX_s011
                                 + 0.766963 w_k1_IBM_s011
                                 + 0.794433 w_k1_INTC_s011
                                 + 0.754301 w_k1_KO_s011
                                 + 0.550163 w_k1_MAT_s011
                                 + 0.874256 w_k1_MSFT_s011
                                 + 0.891658 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.856583 w_k1_ADM_s011
                                 + 1.12984 w_k1_AXP_s011
                                 + 0.988941 w_k1_DIS_s011
                                 + 0.870973 w_k1_FDX_s011
                                 + 1.35499 w_k1_IBM_s011
                                 + 0.944813 w_k1_INTC_s011
                                 + 1.04273 w_k1_KO_s011
                                 + 0.942067 w_k1_MAT_s011
                                 + 0.813675 w_k1_MSFT_s011
                                 + 0.87444 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.01482 w_k1_ADM_s011
                                 + 1.61011 w_k1_AXP_s011
                                 + 1.53369 w_k1_DIS_s011
                                 + 1.37541 w_k1_FDX_s011
                                 + 1.39391 w_k1_IBM_s011
                                 + 1.22976 w_k1_INTC_s011
                                 + 1.25634 w_k1_KO_s011 + 1.28386 w_k1_MAT_s011
                                 + 1.2261 w_k1_MSFT_s011 + 1.26993 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.00683 w_k1_ADM_s011
                                 + 1.2126 w_k1_AXP_s011 + 1.45395 w_k1_DIS_s011
                                 + 1.19105 w_k1_FDX_s011
                                 + 1.10455 w_k1_IBM_s011
                                 + 1.28629 w_k1_INTC_s011
                                 + 1.19038 w_k1_KO_s011
                                 + 0.586069 w_k1_MAT_s011
                                 + 1.20798 w_k1_MSFT_s011
                                 + 1.04916 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.99431 w_k1_ADM_s012
                                 + 0.830782 w_k1_AXP_s012
                                 + 0.992091 w_k1_DIS_s012
                                 + 1.02338 w_k1_FDX_s012
                                 + 0.700151 w_k1_IBM_s012
                                 + 0.918145 w_k1_INTC_s012
                                 + 0.894828 w_k1_KO_s012 + 0.5346 w_k1_MAT_s012
                                 + 1.01407 w_k1_MSFT_s012
                                 + 0.911604 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.835367 w_k1_ADM_s012
                                 + 0.682978 w_k1_AXP_s012
                                 + 0.597317 w_k1_DIS_s012
                                 + 0.723936 w_k1_FDX_s012
                                 + 0.773431 w_k1_IBM_s012
                                 + 0.841904 w_k1_INTC_s012
                                 + 0.819029 w_k1_KO_s012
                                 + 0.525852 w_k1_MAT_s012
                                 + 0.768885 w_k1_MSFT_s012
                                 + 0.814255 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.20228 w_k1_ADM_s012
                                 + 0.870367 w_k1_AXP_s012
                                 + 1.12508 w_k1_DIS_s012
                                 + 1.02418 w_k1_FDX_s012
                                 + 1.05054 w_k1_IBM_s012
                                 + 0.884509 w_k1_INTC_s012
                                 + 1.15068 w_k1_KO_s012
                                 + 0.984504 w_k1_MAT_s012
                                 + 1.08486 w_k1_MSFT_s012
                                 + 1.21266 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.63923 w_k1_ADM_s012
                                 + 1.41411 w_k1_AXP_s012
                                 + 1.59795 w_k1_DIS_s012
                                 + 1.44306 w_k1_FDX_s012
                                 + 0.973072 w_k1_IBM_s012
                                 + 1.04591 w_k1_INTC_s012 + 1.2723 w_k1_KO_s012
                                 + 1.46394 w_k1_MAT_s012
                                 + 1.4235 w_k1_MSFT_s012 + 1.20909 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.00811 w_k1_ADM_s012
                                 + 0.823803 w_k1_AXP_s012
                                 + 0.764699 w_k1_DIS_s012
                                 + 0.801158 w_k1_FDX_s012
                                 + 1.1837 w_k1_IBM_s012
                                 + 0.798615 w_k1_INTC_s012
                                 + 1.03135 w_k1_KO_s012 + 1.11141 w_k1_MAT_s012
                                 + 0.717799 w_k1_MSFT_s012
                                 + 0.969882 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.13234 w_k1_ADM_s012
                                 + 1.3422 w_k1_AXP_s012 + 1.27244 w_k1_DIS_s012
                                 + 1.07804 w_k1_FDX_s012
                                 + 1.63367 w_k1_IBM_s012
                                 + 0.967852 w_k1_INTC_s012
                                 + 1.30924 w_k1_KO_s012 + 1.29063 w_k1_MAT_s012
                                 + 1.0977 w_k1_MSFT_s012 + 1.16508 w_k1_PG_s012
                                 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.864981 w_k1_ADM_s012
                                 + 0.642916 w_k1_AXP_s012
                                 + 0.78612 w_k1_DIS_s012
                                 + 0.801893 w_k1_FDX_s012
                                 + 0.952387 w_k1_IBM_s012
                                 + 0.923953 w_k1_INTC_s012
                                 + 1.00042 w_k1_KO_s012
                                 + 0.819842 w_k1_MAT_s012
                                 + 1.0049 w_k1_MSFT_s012
                                 + 0.910359 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.19273 w_k1_ADM_s012
                                 + 1.22072 w_k1_AXP_s012
                                 + 0.955812 w_k1_DIS_s012
                                 + 1.04559 w_k1_FDX_s012
                                 + 0.928284 w_k1_IBM_s012
                                 + 0.768466 w_k1_INTC_s012
                                 + 1.0876 w_k1_KO_s012 + 1.43173 w_k1_MAT_s012
                                 + 0.887072 w_k1_MSFT_s012
                                 + 1.04608 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.02019 w_k1_ADM_s013
                                 + 0.661183 w_k1_AXP_s013
                                 + 1.22577 w_k1_DIS_s013
                                 + 0.88683 w_k1_FDX_s013
                                 + 1.13332 w_k1_IBM_s013
                                 + 0.87183 w_k1_INTC_s013
                                 + 1.10428 w_k1_KO_s013
                                 + 0.568342 w_k1_MAT_s013
                                 + 1.0143 w_k1_MSFT_s013 + 1.04449 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.859111 w_k1_ADM_s013
                                 + 0.726453 w_k1_AXP_s013
                                 + 0.642451 w_k1_DIS_s013
                                 + 0.715271 w_k1_FDX_s013
                                 + 0.6952 w_k1_IBM_s013
                                 + 0.743779 w_k1_INTC_s013
                                 + 0.672295 w_k1_KO_s013
                                 + 0.953212 w_k1_MAT_s013
                                 + 0.676393 w_k1_MSFT_s013
                                 + 0.903759 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.943172 w_k1_ADM_s013
                                 + 0.927646 w_k1_AXP_s013
                                 + 0.935912 w_k1_DIS_s013
                                 + 0.975721 w_k1_FDX_s013
                                 + 0.779355 w_k1_IBM_s013
                                 + 1.18796 w_k1_INTC_s013
                                 + 0.932995 w_k1_KO_s013
                                 + 0.709322 w_k1_MAT_s013
                                 + 1.03025 w_k1_MSFT_s013
                                 + 1.02498 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.16527 w_k1_ADM_s013
                                 + 1.28322 w_k1_AXP_s013
                                 + 1.09851 w_k1_DIS_s013
                                 + 1.01974 w_k1_FDX_s013
                                 + 1.34404 w_k1_IBM_s013
                                 + 0.999646 w_k1_INTC_s013
                                 + 1.28313 w_k1_KO_s013 + 1.58821 w_k1_MAT_s013
                                 + 0.922379 w_k1_MSFT_s013
                                 + 1.19984 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.01305 w_k1_ADM_s013
                                 + 0.984235 w_k1_AXP_s013
                                 + 1.40408 w_k1_DIS_s013 + 1.185 w_k1_FDX_s013
                                 + 0.890129 w_k1_IBM_s013
                                 + 1.14848 w_k1_INTC_s013
                                 + 1.12274 w_k1_KO_s013
                                 + 0.906068 w_k1_MAT_s013
                                 + 1.1115 w_k1_MSFT_s013 + 1.06715 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.824911 w_k1_ADM_s013
                                 + 0.585851 w_k1_AXP_s013
                                 + 0.703557 w_k1_DIS_s013
                                 + 0.685629 w_k1_FDX_s013
                                 + 1.00536 w_k1_IBM_s013
                                 + 0.813183 w_k1_INTC_s013
                                 + 0.908884 w_k1_KO_s013
                                 + 0.764671 w_k1_MAT_s013
                                 + 0.796665 w_k1_MSFT_s013
                                 + 0.886603 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.35157 w_k1_ADM_s013
                                 + 1.43598 w_k1_AXP_s013
                                 + 1.61638 w_k1_DIS_s013
                                 + 1.32947 w_k1_FDX_s013
                                 + 1.14356 w_k1_IBM_s013
                                 + 1.28464 w_k1_INTC_s013
                                 + 1.29276 w_k1_KO_s013 + 1.2338 w_k1_MAT_s013
                                 + 1.23541 w_k1_MSFT_s013
                                 + 1.24728 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.884278 w_k1_ADM_s013
                                 + 0.849932 w_k1_AXP_s013
                                 + 0.76249 w_k1_DIS_s013
                                 + 0.87597 w_k1_FDX_s013
                                 + 1.07566 w_k1_IBM_s013 + 0.953 w_k1_INTC_s013
                                 + 0.967326 w_k1_KO_s013
                                 + 0.824394 w_k1_MAT_s013
                                 + 0.871583 w_k1_MSFT_s013
                                 + 0.919701 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.97562 w_k1_ADM_s014
                                 + 0.833748 w_k1_AXP_s014
                                 + 0.688219 w_k1_DIS_s014
                                 + 0.849004 w_k1_FDX_s014
                                 + 0.704436 w_k1_IBM_s014
                                 + 0.993242 w_k1_INTC_s014
                                 + 0.649215 w_k1_KO_s014
                                 + 0.555746 w_k1_MAT_s014
                                 + 0.849287 w_k1_MSFT_s014
                                 + 0.850538 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.15058 w_k1_ADM_s014
                                 + 1.27537 w_k1_AXP_s014
                                 + 1.40812 w_k1_DIS_s014
                                 + 1.23339 w_k1_FDX_s014
                                 + 1.26636 w_k1_IBM_s014
                                 + 1.53672 w_k1_INTC_s014
                                 + 1.34658 w_k1_KO_s014 + 1.2575 w_k1_MAT_s014
                                 + 1.30059 w_k1_MSFT_s014 + 1.1812 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.10588 w_k1_ADM_s014
                                 + 1.38009 w_k1_AXP_s014 + 1.3868 w_k1_DIS_s014
                                 + 1.19247 w_k1_FDX_s014
                                 + 1.01937 w_k1_IBM_s014
                                 + 1.30862 w_k1_INTC_s014 + 1.0692 w_k1_KO_s014
                                 + 1.04903 w_k1_MAT_s014
                                 + 1.13685 w_k1_MSFT_s014
                                 + 1.07767 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.23535 w_k1_ADM_s014
                                 + 1.6401 w_k1_AXP_s014 + 1.53929 w_k1_DIS_s014
                                 + 1.2496 w_k1_FDX_s014 + 1.7897 w_k1_IBM_s014
                                 + 1.28366 w_k1_INTC_s014
                                 + 1.63739 w_k1_KO_s014 + 1.6832 w_k1_MAT_s014
                                 + 1.17674 w_k1_MSFT_s014
                                 + 1.22999 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.14696 w_k1_ADM_s014
                                 + 1.20772 w_k1_AXP_s014
                                 + 1.02421 w_k1_DIS_s014
                                 + 1.18393 w_k1_FDX_s014
                                 + 0.809488 w_k1_IBM_s014
                                 + 1.34165 w_k1_INTC_s014
                                 + 0.936923 w_k1_KO_s014
                                 + 0.685722 w_k1_MAT_s014
                                 + 1.11877 w_k1_MSFT_s014
                                 + 1.00221 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.07014 w_k1_ADM_s014
                                 + 0.857995 w_k1_AXP_s014
                                 + 0.797999 w_k1_DIS_s014
                                 + 0.853719 w_k1_FDX_s014
                                 + 1.00708 w_k1_IBM_s014
                                 + 0.923673 w_k1_INTC_s014
                                 + 0.957562 w_k1_KO_s014
                                 + 0.690826 w_k1_MAT_s014
                                 + 0.938062 w_k1_MSFT_s014
                                 + 0.958988 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.05564 w_k1_ADM_s014
                                 + 0.881174 w_k1_AXP_s014
                                 + 1.17889 w_k1_DIS_s014
                                 + 0.926584 w_k1_FDX_s014
                                 + 1.0157 w_k1_IBM_s014
                                 + 1.10584 w_k1_INTC_s014
                                 + 1.05427 w_k1_KO_s014
                                 + 0.765095 w_k1_MAT_s014
                                 + 1.05531 w_k1_MSFT_s014 + 1.0354 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.940065 w_k1_ADM_s014
                                 + 1.04572 w_k1_AXP_s014
                                 + 0.879782 w_k1_DIS_s014
                                 + 0.806129 w_k1_FDX_s014
                                 + 1.33737 w_k1_IBM_s014
                                 + 1.02438 w_k1_INTC_s014
                                 + 1.06659 w_k1_KO_s014 + 1.07405 w_k1_MAT_s014
                                 + 0.844044 w_k1_MSFT_s014
                                 + 1.02837 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.07539 w_k1_ADM_s015
                                 + 0.683312 w_k1_AXP_s015
                                 + 0.795057 w_k1_DIS_s015
                                 + 0.848974 w_k1_FDX_s015
                                 + 0.844644 w_k1_IBM_s015
                                 + 0.862274 w_k1_INTC_s015
                                 + 0.89125 w_k1_KO_s015
                                 + 0.731785 w_k1_MAT_s015
                                 + 0.857187 w_k1_MSFT_s015
                                 + 0.832965 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.2745 w_k1_ADM_s015
                                 + 1.06857 w_k1_AXP_s015
                                 + 1.46596 w_k1_DIS_s015
                                 + 1.20418 w_k1_FDX_s015
                                 + 1.01852 w_k1_IBM_s015
                                 + 1.07441 w_k1_INTC_s015
                                 + 1.48208 w_k1_KO_s015 + 1.2568 w_k1_MAT_s015
                                 + 1.23809 w_k1_MSFT_s015
                                 + 1.09206 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.27081 w_k1_ADM_s015
                                 + 1.38762 w_k1_AXP_s015
                                 + 1.27768 w_k1_DIS_s015
                                 + 1.22307 w_k1_FDX_s015
                                 + 0.973894 w_k1_IBM_s015
                                 + 1.05125 w_k1_INTC_s015 + 1.2535 w_k1_KO_s015
                                 + 1.47771 w_k1_MAT_s015
                                 + 1.13294 w_k1_MSFT_s015
                                 + 1.16272 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.25029 w_k1_ADM_s015
                                 + 1.16439 w_k1_AXP_s015
                                 + 0.865957 w_k1_DIS_s015
                                 + 0.968867 w_k1_FDX_s015
                                 + 0.948839 w_k1_IBM_s015
                                 + 0.939703 w_k1_INTC_s015
                                 + 1.06133 w_k1_KO_s015 + 1.25071 w_k1_MAT_s015
                                 + 0.917526 w_k1_MSFT_s015
                                 + 1.04358 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.33855 w_k1_ADM_s015
                                 + 1.96531 w_k1_AXP_s015
                                 + 1.70179 w_k1_DIS_s015
                                 + 1.44448 w_k1_FDX_s015
                                 + 1.54155 w_k1_IBM_s015
                                 + 1.43256 w_k1_INTC_s015 + 1.3929 w_k1_KO_s015
                                 + 2.01349 w_k1_MAT_s015
                                 + 1.1905 w_k1_MSFT_s015 + 1.20619 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.12567 w_k1_ADM_s015
                                 + 1.0414 w_k1_AXP_s015 + 1.19569 w_k1_DIS_s015
                                 + 1.06215 w_k1_FDX_s015
                                 + 1.21656 w_k1_IBM_s015
                                 + 0.867966 w_k1_INTC_s015
                                 + 1.13028 w_k1_KO_s015 + 1.23449 w_k1_MAT_s015
                                 + 0.927676 w_k1_MSFT_s015
                                 + 1.0735 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.769022 w_k1_ADM_s015
                                 + 0.769944 w_k1_AXP_s015
                                 + 0.71629 w_k1_DIS_s015
                                 + 0.888527 w_k1_FDX_s015
                                 + 0.681214 w_k1_IBM_s015
                                 + 0.825307 w_k1_INTC_s015
                                 + 0.840281 w_k1_KO_s015
                                 + 0.999895 w_k1_MAT_s015
                                 + 0.847383 w_k1_MSFT_s015
                                 + 0.84823 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.06021 w_k1_ADM_s015
                                 + 0.789866 w_k1_AXP_s015
                                 + 0.912772 w_k1_DIS_s015
                                 + 0.866955 w_k1_FDX_s015
                                 + 1.36771 w_k1_IBM_s015
                                 + 0.795982 w_k1_INTC_s015
                                 + 1.05517 w_k1_KO_s015 + 1.6449 w_k1_MAT_s015
                                 + 0.944234 w_k1_MSFT_s015
                                 + 1.00734 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.09869 w_k1_ADM_s016
                                 + 0.928275 w_k1_AXP_s016
                                 + 0.8815 w_k1_DIS_s016
                                 + 0.894048 w_k1_FDX_s016
                                 + 1.1074 w_k1_IBM_s016
                                 + 0.891026 w_k1_INTC_s016
                                 + 1.03331 w_k1_KO_s016 + 1.23023 w_k1_MAT_s016
                                 + 0.818893 w_k1_MSFT_s016
                                 + 0.900942 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.25587 w_k1_ADM_s016
                                 + 1.31852 w_k1_AXP_s016
                                 + 1.60587 w_k1_DIS_s016
                                 + 1.42424 w_k1_FDX_s016
                                 + 1.14409 w_k1_IBM_s016
                                 + 1.39299 w_k1_INTC_s016
                                 + 1.31639 w_k1_KO_s016 + 1.16485 w_k1_MAT_s016
                                 + 1.42423 w_k1_MSFT_s016
                                 + 1.20242 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.14225 w_k1_ADM_s016
                                 + 1.28519 w_k1_AXP_s016
                                 + 0.940849 w_k1_DIS_s016
                                 + 1.10848 w_k1_FDX_s016
                                 + 1.37112 w_k1_IBM_s016
                                 + 0.978514 w_k1_INTC_s016
                                 + 1.1397 w_k1_KO_s016 + 1.90391 w_k1_MAT_s016
                                 + 1.11034 w_k1_MSFT_s016
                                 + 1.06239 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.2658 w_k1_ADM_s016
                                 + 1.1101 w_k1_AXP_s016 + 1.30659 w_k1_DIS_s016
                                 + 1.06636 w_k1_FDX_s016
                                 + 1.47709 w_k1_IBM_s016
                                 + 1.02524 w_k1_INTC_s016 + 1.3275 w_k1_KO_s016
                                 + 1.20178 w_k1_MAT_s016
                                 + 1.09621 w_k1_MSFT_s016
                                 + 1.15973 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.19494 w_k1_ADM_s016
                                 + 1.01563 w_k1_AXP_s016
                                 + 1.17323 w_k1_DIS_s016 + 1.1476 w_k1_FDX_s016
                                 + 0.984099 w_k1_IBM_s016
                                 + 1.30565 w_k1_INTC_s016
                                 + 1.21338 w_k1_KO_s016 + 1.21983 w_k1_MAT_s016
                                 + 1.27443 w_k1_MSFT_s016
                                 + 1.10297 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.826664 w_k1_ADM_s016
                                 + 0.685567 w_k1_AXP_s016
                                 + 0.739202 w_k1_DIS_s016
                                 + 0.848632 w_k1_FDX_s016
                                 + 0.739244 w_k1_IBM_s016
                                 + 0.850469 w_k1_INTC_s016
                                 + 0.846395 w_k1_KO_s016
                                 + 0.754382 w_k1_MAT_s016
                                 + 0.918334 w_k1_MSFT_s016
                                 + 0.831776 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.1597 w_k1_ADM_s016
                                 + 1.03302 w_k1_AXP_s016
                                 + 1.15208 w_k1_DIS_s016
                                 + 1.07453 w_k1_FDX_s016
                                 + 0.963297 w_k1_IBM_s016
                                 + 0.967877 w_k1_INTC_s016
                                 + 1.04451 w_k1_KO_s016 + 1.32413 w_k1_MAT_s016
                                 + 1.07626 w_k1_MSFT_s016
                                 + 1.04461 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.07902 w_k1_ADM_s016
                                 + 0.978202 w_k1_AXP_s016
                                 + 0.820765 w_k1_DIS_s016
                                 + 1.06326 w_k1_FDX_s016
                                 + 0.832041 w_k1_IBM_s016
                                 + 1.05751 w_k1_INTC_s016
                                 + 0.827361 w_k1_KO_s016
                                 + 0.879084 w_k1_MAT_s016
                                 + 1.00219 w_k1_MSFT_s016
                                 + 0.971227 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.820236 w_k1_ADM_s017
                                 + 0.657601 w_k1_AXP_s017
                                 + 0.603339 w_k1_DIS_s017
                                 + 0.777071 w_k1_FDX_s017
                                 + 0.674466 w_k1_IBM_s017
                                 + 0.947507 w_k1_INTC_s017
                                 + 0.654984 w_k1_KO_s017
                                 + 0.583883 w_k1_MAT_s017
                                 + 0.93539 w_k1_MSFT_s017
                                 + 0.882375 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.14415 w_k1_ADM_s017
                                 + 1.19516 w_k1_AXP_s017
                                 + 0.780869 w_k1_DIS_s017
                                 + 1.06099 w_k1_FDX_s017
                                 + 0.950057 w_k1_IBM_s017
                                 + 0.920493 w_k1_INTC_s017
                                 + 0.72656 w_k1_KO_s017 + 1.09789 w_k1_MAT_s017
                                 + 0.877849 w_k1_MSFT_s017
                                 + 0.993775 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.874119 w_k1_ADM_s017
                                 + 0.825895 w_k1_AXP_s017
                                 + 1.04238 w_k1_DIS_s017
                                 + 0.870186 w_k1_FDX_s017
                                 + 1.09792 w_k1_IBM_s017
                                 + 1.00995 w_k1_INTC_s017
                                 + 1.07767 w_k1_KO_s017 + 1.06701 w_k1_MAT_s017
                                 + 1.09172 w_k1_MSFT_s017 + 1.0042 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.85737 w_k1_ADM_s017
                                 + 0.802816 w_k1_AXP_s017
                                 + 0.683053 w_k1_DIS_s017
                                 + 0.772513 w_k1_FDX_s017
                                 + 0.884919 w_k1_IBM_s017
                                 + 0.865828 w_k1_INTC_s017
                                 + 0.727024 w_k1_KO_s017
                                 + 0.759369 w_k1_MAT_s017
                                 + 0.925755 w_k1_MSFT_s017
                                 + 0.885581 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.07534 w_k1_ADM_s017
                                 + 0.930129 w_k1_AXP_s017
                                 + 1.2352 w_k1_DIS_s017 + 1.16693 w_k1_FDX_s017
                                 + 0.882687 w_k1_IBM_s017
                                 + 1.23233 w_k1_INTC_s017 + 1.1036 w_k1_KO_s017
                                 + 0.770747 w_k1_MAT_s017
                                 + 1.13581 w_k1_MSFT_s017
                                 + 1.11472 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.39527 w_k1_ADM_s017
                                 + 1.16437 w_k1_AXP_s017
                                 + 1.09992 w_k1_DIS_s017
                                 + 1.07003 w_k1_FDX_s017
                                 + 1.30826 w_k1_IBM_s017
                                 + 1.09691 w_k1_INTC_s017
                                 + 1.09566 w_k1_KO_s017 + 1.25944 w_k1_MAT_s017
                                 + 1.01149 w_k1_MSFT_s017
                                 + 1.12133 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.09909 w_k1_ADM_s017
                                 + 0.898422 w_k1_AXP_s017
                                 + 0.735809 w_k1_DIS_s017
                                 + 0.963732 w_k1_FDX_s017
                                 + 0.641243 w_k1_IBM_s017
                                 + 0.904265 w_k1_INTC_s017
                                 + 0.70395 w_k1_KO_s017
                                 + 0.702618 w_k1_MAT_s017
                                 + 0.957362 w_k1_MSFT_s017
                                 + 0.978587 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.20863 w_k1_ADM_s017
                                 + 1.43388 w_k1_AXP_s017
                                 + 1.44788 w_k1_DIS_s017 + 1.1783 w_k1_FDX_s017
                                 + 1.59665 w_k1_IBM_s017
                                 + 1.10289 w_k1_INTC_s017
                                 + 1.21547 w_k1_KO_s017 + 1.6031 w_k1_MAT_s017
                                 + 1.37396 w_k1_MSFT_s017
                                 + 1.14067 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.03206 w_k1_ADM_s100
                                 + 1.00455 w_k1_AXP_s100
                                 + 0.840695 w_k1_DIS_s100
                                 + 0.79811 w_k1_FDX_s100
                                 + 1.09418 w_k1_IBM_s100
                                 + 0.704951 w_k1_INTC_s100
                                 + 0.920619 w_k1_KO_s100
                                 + 1.06513 w_k1_MAT_s100
                                 + 0.683865 w_k1_MSFT_s100
                                 + 0.945784 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.939303 w_k1_ADM_s100
                                 + 1.39952 w_k1_AXP_s100
                                 + 1.02221 w_k1_DIS_s100
                                 + 1.06596 w_k1_FDX_s100
                                 + 1.02153 w_k1_IBM_s100
                                 + 0.938124 w_k1_INTC_s100
                                 + 0.950416 w_k1_KO_s100
                                 + 1.27177 w_k1_MAT_s100
                                 + 0.957461 w_k1_MSFT_s100
                                 + 1.04439 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.893091 w_k1_ADM_s100
                                 + 0.950993 w_k1_AXP_s100
                                 + 0.888516 w_k1_DIS_s100
                                 + 0.962498 w_k1_FDX_s100
                                 + 0.588615 w_k1_IBM_s100
                                 + 0.932043 w_k1_INTC_s100
                                 + 0.781839 w_k1_KO_s100
                                 + 0.555779 w_k1_MAT_s100
                                 + 0.843167 w_k1_MSFT_s100
                                 + 0.754126 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.05318 w_k1_ADM_s100
                                 + 1.47765 w_k1_AXP_s100
                                 + 1.94182 w_k1_DIS_s100
                                 + 1.20608 w_k1_FDX_s100
                                 + 1.56466 w_k1_IBM_s100
                                 + 0.987749 w_k1_INTC_s100
                                 + 1.67338 w_k1_KO_s100 + 1.8008 w_k1_MAT_s100
                                 + 1.07977 w_k1_MSFT_s100
                                 + 1.17872 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.09177 w_k1_ADM_s100
                                 + 1.22907 w_k1_AXP_s100
                                 + 1.45027 w_k1_DIS_s100
                                 + 1.21911 w_k1_FDX_s100
                                 + 0.954992 w_k1_IBM_s100
                                 + 0.932899 w_k1_INTC_s100
                                 + 1.14028 w_k1_KO_s100 + 1.04709 w_k1_MAT_s100
                                 + 1.12005 w_k1_MSFT_s100
                                 + 1.11428 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.864281 w_k1_ADM_s100
                                 + 0.929206 w_k1_AXP_s100
                                 + 1.24698 w_k1_DIS_s100
                                 + 0.931733 w_k1_FDX_s100
                                 + 1.02422 w_k1_IBM_s100
                                 + 0.943163 w_k1_INTC_s100
                                 + 1.15027 w_k1_KO_s100
                                 + 0.793169 w_k1_MAT_s100
                                 + 1.03309 w_k1_MSFT_s100
                                 + 0.983685 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.907317 w_k1_ADM_s100
                                 + 0.842737 w_k1_AXP_s100
                                 + 0.862241 w_k1_DIS_s100
                                 + 0.880864 w_k1_FDX_s100
                                 + 0.775522 w_k1_IBM_s100
                                 + 0.886322 w_k1_INTC_s100
                                 + 0.769947 w_k1_KO_s100
                                 + 0.629755 w_k1_MAT_s100
                                 + 0.8168 w_k1_MSFT_s100
                                 + 0.942542 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.00129 w_k1_ADM_s100
                                 + 1.22475 w_k1_AXP_s100
                                 + 1.07761 w_k1_DIS_s100
                                 + 0.934957 w_k1_FDX_s100
                                 + 1.38147 w_k1_IBM_s100
                                 + 1.00145 w_k1_INTC_s100
                                 + 1.05592 w_k1_KO_s100 + 1.26872 w_k1_MAT_s100
                                 + 0.916648 w_k1_MSFT_s100
                                 + 1.01698 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.29679 w_k1_ADM_s101
                                 + 1.07954 w_k1_AXP_s101
                                 + 0.94962 w_k1_DIS_s101
                                 + 1.13831 w_k1_FDX_s101
                                 + 0.889098 w_k1_IBM_s101
                                 + 0.978353 w_k1_INTC_s101
                                 + 1.03458 w_k1_KO_s101
                                 + 0.835555 w_k1_MAT_s101
                                 + 1.0091 w_k1_MSFT_s101
                                 + 0.957058 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.14623 w_k1_ADM_s101
                                 + 0.937385 w_k1_AXP_s101
                                 + 1.47838 w_k1_DIS_s101
                                 + 1.09267 w_k1_FDX_s101
                                 + 1.45418 w_k1_IBM_s101
                                 + 1.06648 w_k1_INTC_s101
                                 + 1.68353 w_k1_KO_s101 + 1.68338 w_k1_MAT_s101
                                 + 1.22084 w_k1_MSFT_s101
                                 + 1.18251 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.27369 w_k1_ADM_s101
                                 + 1.03325 w_k1_AXP_s101
                                 + 1.01188 w_k1_DIS_s101
                                 + 0.951183 w_k1_FDX_s101
                                 + 1.34715 w_k1_IBM_s101
                                 + 1.01689 w_k1_INTC_s101
                                 + 1.15218 w_k1_KO_s101 + 1.07634 w_k1_MAT_s101
                                 + 0.921989 w_k1_MSFT_s101
                                 + 0.87191 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.821446 w_k1_ADM_s101
                                 + 0.865029 w_k1_AXP_s101
                                 + 0.775899 w_k1_DIS_s101
                                 + 0.898246 w_k1_FDX_s101
                                 + 0.77142 w_k1_IBM_s101
                                 + 0.717191 w_k1_INTC_s101
                                 + 0.859072 w_k1_KO_s101
                                 + 1.27898 w_k1_MAT_s101
                                 + 0.858084 w_k1_MSFT_s101
                                 + 0.87975 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.42036 w_k1_ADM_s101
                                 + 1.3409 w_k1_AXP_s101 + 1.68809 w_k1_DIS_s101
                                 + 1.33437 w_k1_FDX_s101
                                 + 1.25391 w_k1_IBM_s101
                                 + 1.06379 w_k1_INTC_s101 + 1.5324 w_k1_KO_s101
                                 + 1.40957 w_k1_MAT_s101
                                 + 1.34456 w_k1_MSFT_s101
                                 + 1.28662 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.05886 w_k1_ADM_s101
                                 + 0.816832 w_k1_AXP_s101
                                 + 1.27426 w_k1_DIS_s101
                                 + 1.09324 w_k1_FDX_s101
                                 + 0.965767 w_k1_IBM_s101
                                 + 0.87657 w_k1_INTC_s101 + 1.193 w_k1_KO_s101
                                 + 1.11159 w_k1_MAT_s101
                                 + 1.12695 w_k1_MSFT_s101
                                 + 1.01137 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.902677 w_k1_ADM_s101
                                 + 0.615191 w_k1_AXP_s101
                                 + 0.67619 w_k1_DIS_s101
                                 + 0.864874 w_k1_FDX_s101
                                 + 0.570382 w_k1_IBM_s101
                                 + 0.785391 w_k1_INTC_s101
                                 + 0.741527 w_k1_KO_s101
                                 + 0.521102 w_k1_MAT_s101
                                 + 0.782717 w_k1_MSFT_s101
                                 + 0.73255 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.998156 w_k1_ADM_s101
                                 + 0.716678 w_k1_AXP_s101
                                 + 0.708161 w_k1_DIS_s101
                                 + 0.751582 w_k1_FDX_s101
                                 + 0.96489 w_k1_IBM_s101
                                 + 0.892989 w_k1_INTC_s101
                                 + 0.884312 w_k1_KO_s101
                                 + 0.901988 w_k1_MAT_s101
                                 + 0.806099 w_k1_MSFT_s101
                                 + 0.773241 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.2888 w_k1_ADM_s102
                                 + 1.54063 w_k1_AXP_s102
                                 + 1.43906 w_k1_DIS_s102
                                 + 1.39704 w_k1_FDX_s102
                                 + 1.06761 w_k1_IBM_s102
                                 + 1.02866 w_k1_INTC_s102
                                 + 1.02486 w_k1_KO_s102 + 1.19487 w_k1_MAT_s102
                                 + 1.05558 w_k1_MSFT_s102
                                 + 1.30456 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.810093 w_k1_ADM_s102
                                 + 0.839963 w_k1_AXP_s102
                                 + 0.589438 w_k1_DIS_s102
                                 + 0.763503 w_k1_FDX_s102
                                 + 0.755177 w_k1_IBM_s102
                                 + 0.764692 w_k1_INTC_s102
                                 + 0.703325 w_k1_KO_s102
                                 + 0.662241 w_k1_MAT_s102
                                 + 0.699807 w_k1_MSFT_s102
                                 + 0.862973 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.13451 w_k1_ADM_s102
                                 + 1.21311 w_k1_AXP_s102
                                 + 0.89959 w_k1_DIS_s102
                                 + 1.03342 w_k1_FDX_s102
                                 + 1.01589 w_k1_IBM_s102
                                 + 1.17214 w_k1_INTC_s102
                                 + 1.03757 w_k1_KO_s102
                                 + 0.997667 w_k1_MAT_s102
                                 + 0.959614 w_k1_MSFT_s102
                                 + 1.07658 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.834062 w_k1_ADM_s102
                                 + 0.739597 w_k1_AXP_s102
                                 + 0.773017 w_k1_DIS_s102
                                 + 0.871548 w_k1_FDX_s102
                                 + 0.904477 w_k1_IBM_s102
                                 + 0.854696 w_k1_INTC_s102
                                 + 0.824413 w_k1_KO_s102
                                 + 0.775893 w_k1_MAT_s102
                                 + 0.767369 w_k1_MSFT_s102
                                 + 1.02666 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.831892 w_k1_ADM_s102
                                 + 0.700446 w_k1_AXP_s102
                                 + 0.714478 w_k1_DIS_s102
                                 + 0.889033 w_k1_FDX_s102
                                 + 0.589736 w_k1_IBM_s102
                                 + 0.727566 w_k1_INTC_s102
                                 + 0.680321 w_k1_KO_s102
                                 + 0.629614 w_k1_MAT_s102
                                 + 0.741789 w_k1_MSFT_s102
                                 + 0.830608 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.45663 w_k1_ADM_s102
                                 + 1.34178 w_k1_AXP_s102
                                 + 1.57378 w_k1_DIS_s102
                                 + 1.17399 w_k1_FDX_s102
                                 + 1.58147 w_k1_IBM_s102
                                 + 1.0363 w_k1_INTC_s102 + 1.28848 w_k1_KO_s102
                                 + 1.18603 w_k1_MAT_s102
                                 + 1.19514 w_k1_MSFT_s102
                                 + 1.41011 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.32889 w_k1_ADM_s102
                                 + 0.928415 w_k1_AXP_s102
                                 + 1.01423 w_k1_DIS_s102
                                 + 1.15529 w_k1_FDX_s102
                                 + 0.736019 w_k1_IBM_s102
                                 + 0.825787 w_k1_INTC_s102
                                 + 0.958395 w_k1_KO_s102
                                 + 0.812883 w_k1_MAT_s102
                                 + 1.12932 w_k1_MSFT_s102
                                 + 1.20496 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.20526 w_k1_ADM_s102
                                 + 1.21109 w_k1_AXP_s102
                                 + 0.933089 w_k1_DIS_s102
                                 + 0.930194 w_k1_FDX_s102
                                 + 1.25964 w_k1_IBM_s102
                                 + 0.94702 w_k1_INTC_s102
                                 + 1.14889 w_k1_KO_s102 + 1.06701 w_k1_MAT_s102
                                 + 0.90568 w_k1_MSFT_s102
                                 + 1.16306 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.981065 w_k1_ADM_s103
                                 + 1.19007 w_k1_AXP_s103 + 1.0622 w_k1_DIS_s103
                                 + 1.04566 w_k1_FDX_s103
                                 + 1.29306 w_k1_IBM_s103
                                 + 0.917017 w_k1_INTC_s103
                                 + 1.18943 w_k1_KO_s103 + 1.48195 w_k1_MAT_s103
                                 + 1.01761 w_k1_MSFT_s103
                                 + 1.10883 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.815276 w_k1_ADM_s103
                                 + 0.742802 w_k1_AXP_s103
                                 + 0.887726 w_k1_DIS_s103
                                 + 0.878895 w_k1_FDX_s103
                                 + 1.01909 w_k1_IBM_s103
                                 + 0.837924 w_k1_INTC_s103
                                 + 0.874396 w_k1_KO_s103
                                 + 0.707927 w_k1_MAT_s103
                                 + 0.84393 w_k1_MSFT_s103
                                 + 1.01495 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.938303 w_k1_ADM_s103
                                 + 0.827295 w_k1_AXP_s103
                                 + 0.844937 w_k1_DIS_s103
                                 + 0.927844 w_k1_FDX_s103
                                 + 0.772681 w_k1_IBM_s103
                                 + 0.945585 w_k1_INTC_s103
                                 + 0.70464 w_k1_KO_s103
                                 + 0.704453 w_k1_MAT_s103
                                 + 0.913031 w_k1_MSFT_s103
                                 + 0.903302 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.745431 w_k1_ADM_s103
                                 + 0.744572 w_k1_AXP_s103
                                 + 0.610582 w_k1_DIS_s103
                                 + 0.746807 w_k1_FDX_s103
                                 + 0.882268 w_k1_IBM_s103
                                 + 0.777393 w_k1_INTC_s103
                                 + 0.725391 w_k1_KO_s103
                                 + 0.70814 w_k1_MAT_s103
                                 + 0.74912 w_k1_MSFT_s103
                                 + 0.818137 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.946782 w_k1_ADM_s103
                                 + 1.14126 w_k1_AXP_s103
                                 + 1.01781 w_k1_DIS_s103
                                 + 1.09124 w_k1_FDX_s103
                                 + 0.913995 w_k1_IBM_s103
                                 + 1.05801 w_k1_INTC_s103
                                 + 0.869669 w_k1_KO_s103
                                 + 0.95968 w_k1_MAT_s103
                                 + 1.00984 w_k1_MSFT_s103
                                 + 1.05496 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.864989 w_k1_ADM_s103
                                 + 1.08963 w_k1_AXP_s103
                                 + 1.01742 w_k1_DIS_s103
                                 + 1.04062 w_k1_FDX_s103
                                 + 1.05704 w_k1_IBM_s103
                                 + 1.09032 w_k1_INTC_s103
                                 + 1.00481 w_k1_KO_s103 + 1.05364 w_k1_MAT_s103
                                 + 0.982138 w_k1_MSFT_s103
                                 + 1.07232 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.990818 w_k1_ADM_s103
                                 + 1.54838 w_k1_AXP_s103 + 1.635 w_k1_DIS_s103
                                 + 1.25877 w_k1_FDX_s103
                                 + 1.71448 w_k1_IBM_s103
                                 + 1.25922 w_k1_INTC_s103 + 1.3308 w_k1_KO_s103
                                 + 1.91076 w_k1_MAT_s103
                                 + 1.17728 w_k1_MSFT_s103
                                 + 1.34678 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.919864 w_k1_ADM_s103
                                 + 1.24912 w_k1_AXP_s103
                                 + 1.38796 w_k1_DIS_s103
                                 + 1.28503 w_k1_FDX_s103
                                 + 1.03253 w_k1_IBM_s103
                                 + 1.25615 w_k1_INTC_s103
                                 + 1.20698 w_k1_KO_s103 + 1.34132 w_k1_MAT_s103
                                 + 1.26775 w_k1_MSFT_s103
                                 + 1.23933 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.01891 w_k1_ADM_s104
                                 + 1.24986 w_k1_AXP_s104
                                 + 1.64757 w_k1_DIS_s104
                                 + 1.18427 w_k1_FDX_s104
                                 + 1.88137 w_k1_IBM_s104
                                 + 1.00036 w_k1_INTC_s104
                                 + 1.47711 w_k1_KO_s104 + 2.12614 w_k1_MAT_s104
                                 + 1.13941 w_k1_MSFT_s104 + 1.214 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.756715 w_k1_ADM_s104
                                 + 0.773101 w_k1_AXP_s104
                                 + 0.895555 w_k1_DIS_s104
                                 + 0.902694 w_k1_FDX_s104
                                 + 0.832508 w_k1_IBM_s104
                                 + 0.703549 w_k1_INTC_s104
                                 + 0.781489 w_k1_KO_s104
                                 + 0.803667 w_k1_MAT_s104
                                 + 0.755577 w_k1_MSFT_s104
                                 + 0.976609 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.805556 w_k1_ADM_s104
                                 + 0.782324 w_k1_AXP_s104
                                 + 0.758777 w_k1_DIS_s104
                                 + 0.903913 w_k1_FDX_s104
                                 + 0.559957 w_k1_IBM_s104
                                 + 0.961243 w_k1_INTC_s104
                                 + 0.7264 w_k1_KO_s104 + 0.663254 w_k1_MAT_s104
                                 + 0.937209 w_k1_MSFT_s104
                                 + 0.882805 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.26814 w_k1_ADM_s104
                                 + 0.895467 w_k1_AXP_s104
                                 + 1.2575 w_k1_DIS_s104 + 1.13006 w_k1_FDX_s104
                                 + 1.12345 w_k1_IBM_s104
                                 + 1.09351 w_k1_INTC_s104
                                 + 0.989115 w_k1_KO_s104
                                 + 1.06696 w_k1_MAT_s104
                                 + 1.03766 w_k1_MSFT_s104
                                 + 1.11039 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.02717 w_k1_ADM_s104
                                 + 1.28574 w_k1_AXP_s104
                                 + 1.37548 w_k1_DIS_s104
                                 + 1.07733 w_k1_FDX_s104
                                 + 1.34869 w_k1_IBM_s104
                                 + 1.14444 w_k1_INTC_s104
                                 + 1.33533 w_k1_KO_s104 + 1.56843 w_k1_MAT_s104
                                 + 1.02378 w_k1_MSFT_s104
                                 + 1.26315 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.772412 w_k1_ADM_s104
                                 + 0.806256 w_k1_AXP_s104
                                 + 0.993309 w_k1_DIS_s104
                                 + 0.835261 w_k1_FDX_s104
                                 + 1.26307 w_k1_IBM_s104
                                 + 0.929453 w_k1_INTC_s104
                                 + 1.08984 w_k1_KO_s104 + 1.45896 w_k1_MAT_s104
                                 + 1.00347 w_k1_MSFT_s104
                                 + 1.06719 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.798848 w_k1_ADM_s104
                                 + 1.29071 w_k1_AXP_s104
                                 + 1.18146 w_k1_DIS_s104
                                 + 1.16375 w_k1_FDX_s104
                                 + 0.958412 w_k1_IBM_s104
                                 + 1.21998 w_k1_INTC_s104
                                 + 1.07211 w_k1_KO_s104 + 1.29267 w_k1_MAT_s104
                                 + 1.11982 w_k1_MSFT_s104
                                 + 1.04986 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.00985 w_k1_ADM_s104
                                 + 1.19946 w_k1_AXP_s104
                                 + 1.95817 w_k1_DIS_s104
                                 + 1.40872 w_k1_FDX_s104
                                 + 1.03248 w_k1_IBM_s104
                                 + 1.66025 w_k1_INTC_s104 + 1.4092 w_k1_KO_s104
                                 + 1.01055 w_k1_MAT_s104
                                 + 1.62511 w_k1_MSFT_s104
                                 + 1.15539 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.45743 w_k1_ADM_s105
                                 + 1.5274 w_k1_AXP_s105 + 1.65976 w_k1_DIS_s105
                                 + 1.26096 w_k1_FDX_s105
                                 + 1.35084 w_k1_IBM_s105
                                 + 1.57818 w_k1_INTC_s105
                                 + 1.50922 w_k1_KO_s105 + 1.28409 w_k1_MAT_s105
                                 + 1.35675 w_k1_MSFT_s105
                                 + 1.24842 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.16339 w_k1_ADM_s105
                                 + 0.821133 w_k1_AXP_s105
                                 + 0.848791 w_k1_DIS_s105
                                 + 0.949809 w_k1_FDX_s105
                                 + 0.657721 w_k1_IBM_s105
                                 + 0.855434 w_k1_INTC_s105
                                 + 0.871168 w_k1_KO_s105
                                 + 0.849073 w_k1_MAT_s105
                                 + 0.870654 w_k1_MSFT_s105
                                 + 0.96216 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.933352 w_k1_ADM_s105
                                 + 0.640054 w_k1_AXP_s105
                                 + 0.569314 w_k1_DIS_s105
                                 + 0.64979 w_k1_FDX_s105
                                 + 0.863333 w_k1_IBM_s105
                                 + 0.945068 w_k1_INTC_s105
                                 + 0.911222 w_k1_KO_s105
                                 + 0.646987 w_k1_MAT_s105
                                 + 0.740678 w_k1_MSFT_s105
                                 + 0.85823 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.06299 w_k1_ADM_s105
                                 + 1.18161 w_k1_AXP_s105
                                 + 0.964004 w_k1_DIS_s105
                                 + 1.0289 w_k1_FDX_s105
                                 + 0.856851 w_k1_IBM_s105
                                 + 0.980624 w_k1_INTC_s105
                                 + 0.943801 w_k1_KO_s105
                                 + 0.80906 w_k1_MAT_s105
                                 + 0.97424 w_k1_MSFT_s105
                                 + 0.999906 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.46767 w_k1_ADM_s105
                                 + 1.29967 w_k1_AXP_s105 + 1.1129 w_k1_DIS_s105
                                 + 1.07163 w_k1_FDX_s105
                                 + 1.11588 w_k1_IBM_s105
                                 + 0.923974 w_k1_INTC_s105
                                 + 1.13841 w_k1_KO_s105 + 1.55991 w_k1_MAT_s105
                                 + 1.01207 w_k1_MSFT_s105
                                 + 1.16731 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.879263 w_k1_ADM_s105
                                 + 0.907837 w_k1_AXP_s105
                                 + 0.983047 w_k1_DIS_s105
                                 + 0.863984 w_k1_FDX_s105
                                 + 1.43596 w_k1_IBM_s105
                                 + 1.0409 w_k1_INTC_s105 + 1.33662 w_k1_KO_s105
                                 + 1.63911 w_k1_MAT_s105
                                 + 1.09015 w_k1_MSFT_s105
                                 + 1.04112 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.37205 w_k1_ADM_s105
                                 + 1.13278 w_k1_AXP_s105
                                 + 1.33523 w_k1_DIS_s105
                                 + 1.21787 w_k1_FDX_s105
                                 + 0.89187 w_k1_IBM_s105
                                 + 1.2528 w_k1_INTC_s105 + 1.22173 w_k1_KO_s105
                                 + 0.82567 w_k1_MAT_s105
                                 + 1.23594 w_k1_MSFT_s105
                                 + 1.20632 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.16139 w_k1_ADM_s105
                                 + 0.841458 w_k1_AXP_s105
                                 + 0.968819 w_k1_DIS_s105
                                 + 0.8654 w_k1_FDX_s105 + 1.06149 w_k1_IBM_s105
                                 + 1.01772 w_k1_INTC_s105
                                 + 1.07192 w_k1_KO_s105
                                 + 0.846454 w_k1_MAT_s105
                                 + 0.980581 w_k1_MSFT_s105
                                 + 0.979406 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.11451 w_k1_ADM_s106
                                 + 1.08606 w_k1_AXP_s106
                                 + 1.05869 w_k1_DIS_s106
                                 + 1.11586 w_k1_FDX_s106
                                 + 1.25765 w_k1_IBM_s106
                                 + 1.03434 w_k1_INTC_s106
                                 + 1.16718 w_k1_KO_s106 + 1.3079 w_k1_MAT_s106
                                 + 1.04564 w_k1_MSFT_s106
                                 + 1.13233 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.02613 w_k1_ADM_s106
                                 + 0.763237 w_k1_AXP_s106
                                 + 0.731769 w_k1_DIS_s106
                                 + 0.95165 w_k1_FDX_s106
                                 + 1.01471 w_k1_IBM_s106
                                 + 1.03177 w_k1_INTC_s106
                                 + 1.00092 w_k1_KO_s106
                                 + 0.925572 w_k1_MAT_s106
                                 + 0.921354 w_k1_MSFT_s106
                                 + 0.920335 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.04319 w_k1_ADM_s106
                                 + 0.728437 w_k1_AXP_s106
                                 + 0.982096 w_k1_DIS_s106
                                 + 0.950747 w_k1_FDX_s106
                                 + 0.872329 w_k1_IBM_s106
                                 + 1.16365 w_k1_INTC_s106
                                 + 1.04813 w_k1_KO_s106
                                 + 0.575295 w_k1_MAT_s106
                                 + 0.966893 w_k1_MSFT_s106
                                 + 0.880163 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.26024 w_k1_ADM_s106
                                 + 1.18198 w_k1_AXP_s106
                                 + 1.20051 w_k1_DIS_s106
                                 + 1.32096 w_k1_FDX_s106
                                 + 1.09543 w_k1_IBM_s106
                                 + 0.946752 w_k1_INTC_s106
                                 + 1.12298 w_k1_KO_s106 + 1.07174 w_k1_MAT_s106
                                 + 1.10024 w_k1_MSFT_s106
                                 + 1.21433 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.38422 w_k1_ADM_s106
                                 + 1.59583 w_k1_AXP_s106
                                 + 2.08591 w_k1_DIS_s106
                                 + 1.65072 w_k1_FDX_s106
                                 + 1.36967 w_k1_IBM_s106
                                 + 1.63817 w_k1_INTC_s106
                                 + 1.47785 w_k1_KO_s106 + 1.43581 w_k1_MAT_s106
                                 + 1.48415 w_k1_MSFT_s106
                                 + 1.42936 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.10173 w_k1_ADM_s106
                                 + 1.14808 w_k1_AXP_s106 + 1.057 w_k1_DIS_s106
                                 + 1.0643 w_k1_FDX_s106 + 1.39321 w_k1_IBM_s106
                                 + 1.2013 w_k1_INTC_s106 + 1.34482 w_k1_KO_s106
                                 + 1.13637 w_k1_MAT_s106
                                 + 1.02879 w_k1_MSFT_s106
                                 + 1.13985 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.80645 w_k1_ADM_s106
                                 + 0.6504 w_k1_AXP_s106
                                 + 0.765493 w_k1_DIS_s106
                                 + 0.905681 w_k1_FDX_s106
                                 + 0.681997 w_k1_IBM_s106
                                 + 0.95086 w_k1_INTC_s106
                                 + 0.847753 w_k1_KO_s106
                                 + 0.577136 w_k1_MAT_s106
                                 + 0.845966 w_k1_MSFT_s106
                                 + 0.844274 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.979902 w_k1_ADM_s106
                                 + 1.28731 w_k1_AXP_s106
                                 + 1.04699 w_k1_DIS_s106
                                 + 1.09918 w_k1_FDX_s106
                                 + 1.51863 w_k1_IBM_s106
                                 + 1.36358 w_k1_INTC_s106
                                 + 1.50018 w_k1_KO_s106 + 1.93072 w_k1_MAT_s106
                                 + 1.17012 w_k1_MSFT_s106
                                 + 1.11063 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.46491 w_k1_ADM_s107
                                 + 1.41981 w_k1_AXP_s107
                                 + 1.43409 w_k1_DIS_s107
                                 + 1.14748 w_k1_FDX_s107 + 1.3827 w_k1_IBM_s107
                                 + 0.951344 w_k1_INTC_s107
                                 + 1.35163 w_k1_KO_s107 + 1.83316 w_k1_MAT_s107
                                 + 1.25366 w_k1_MSFT_s107
                                 + 1.17741 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.818029 w_k1_ADM_s107
                                 + 0.850321 w_k1_AXP_s107
                                 + 0.74127 w_k1_DIS_s107
                                 + 0.864536 w_k1_FDX_s107
                                 + 0.908708 w_k1_IBM_s107
                                 + 0.812387 w_k1_INTC_s107
                                 + 0.70948 w_k1_KO_s107 + 1.16387 w_k1_MAT_s107
                                 + 0.872219 w_k1_MSFT_s107
                                 + 0.885574 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.12499 w_k1_ADM_s107
                                 + 1.17931 w_k1_AXP_s107
                                 + 1.27445 w_k1_DIS_s107
                                 + 1.20509 w_k1_FDX_s107
                                 + 0.853299 w_k1_IBM_s107
                                 + 0.865385 w_k1_INTC_s107
                                 + 0.98463 w_k1_KO_s107 + 1.31376 w_k1_MAT_s107
                                 + 1.1106 w_k1_MSFT_s107
                                 + 0.983306 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.13666 w_k1_ADM_s107
                                 + 0.831395 w_k1_AXP_s107
                                 + 1.33985 w_k1_DIS_s107
                                 + 1.13262 w_k1_FDX_s107
                                 + 0.814849 w_k1_IBM_s107
                                 + 1.01104 w_k1_INTC_s107
                                 + 0.974723 w_k1_KO_s107
                                 + 0.916082 w_k1_MAT_s107
                                 + 1.07665 w_k1_MSFT_s107
                                 + 1.01658 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.921891 w_k1_ADM_s107
                                 + 0.881226 w_k1_AXP_s107
                                 + 0.825991 w_k1_DIS_s107
                                 + 0.888138 w_k1_FDX_s107
                                 + 0.685223 w_k1_IBM_s107
                                 + 0.962533 w_k1_INTC_s107
                                 + 0.827045 w_k1_KO_s107
                                 + 0.988822 w_k1_MAT_s107
                                 + 1.0718 w_k1_MSFT_s107
                                 + 0.889365 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.892618 w_k1_ADM_s107
                                 + 0.868879 w_k1_AXP_s107
                                 + 0.896559 w_k1_DIS_s107
                                 + 0.851311 w_k1_FDX_s107
                                 + 1.28495 w_k1_IBM_s107
                                 + 0.857996 w_k1_INTC_s107
                                 + 1.06658 w_k1_KO_s107 + 1.62421 w_k1_MAT_s107
                                 + 0.857047 w_k1_MSFT_s107
                                 + 0.806139 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.768274 w_k1_ADM_s107
                                 + 0.586058 w_k1_AXP_s107
                                 + 0.58293 w_k1_DIS_s107
                                 + 0.701898 w_k1_FDX_s107
                                 + 0.784404 w_k1_IBM_s107
                                 + 0.900013 w_k1_INTC_s107
                                 + 0.690084 w_k1_KO_s107
                                 + 0.670262 w_k1_MAT_s107
                                 + 0.928642 w_k1_MSFT_s107
                                 + 0.709542 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.03379 w_k1_ADM_s107
                                 + 0.940585 w_k1_AXP_s107
                                 + 1.17569 w_k1_DIS_s107
                                 + 1.09914 w_k1_FDX_s107
                                 + 1.29205 w_k1_IBM_s107
                                 + 1.06827 w_k1_INTC_s107
                                 + 1.13385 w_k1_KO_s107 + 1.40108 w_k1_MAT_s107
                                 + 1.13967 w_k1_MSFT_s107
                                 + 0.961757 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.04583 w_k1_ADM_s110
                                 + 0.970122 w_k1_AXP_s110
                                 + 0.910384 w_k1_DIS_s110
                                 + 1.02578 w_k1_FDX_s110
                                 + 0.775335 w_k1_IBM_s110
                                 + 1.15878 w_k1_INTC_s110
                                 + 0.840865 w_k1_KO_s110 + 0.7013 w_k1_MAT_s110
                                 + 1.04027 w_k1_MSFT_s110
                                 + 0.886077 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.782539 w_k1_ADM_s110
                                 + 0.900208 w_k1_AXP_s110
                                 + 1.31276 w_k1_DIS_s110
                                 + 1.04269 w_k1_FDX_s110
                                 + 1.02499 w_k1_IBM_s110
                                 + 0.951063 w_k1_INTC_s110
                                 + 1.0947 w_k1_KO_s110 + 1.31228 w_k1_MAT_s110
                                 + 1.19534 w_k1_MSFT_s110
                                 + 0.968551 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.929374 w_k1_ADM_s110
                                 + 0.709544 w_k1_AXP_s110
                                 + 0.720845 w_k1_DIS_s110
                                 + 0.80853 w_k1_FDX_s110
                                 + 0.738644 w_k1_IBM_s110
                                 + 0.727407 w_k1_INTC_s110
                                 + 0.731816 w_k1_KO_s110
                                 + 0.855596 w_k1_MAT_s110
                                 + 0.90393 w_k1_MSFT_s110
                                 + 0.813451 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.2135 w_k1_ADM_s110
                                 + 1.43483 w_k1_AXP_s110 + 1.4053 w_k1_DIS_s110
                                 + 1.27566 w_k1_FDX_s110
                                 + 0.791639 w_k1_IBM_s110
                                 + 0.95213 w_k1_INTC_s110
                                 + 0.889529 w_k1_KO_s110
                                 + 1.09247 w_k1_MAT_s110
                                 + 1.17513 w_k1_MSFT_s110
                                 + 1.03503 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.615452 w_k1_ADM_s110
                                 + 0.776852 w_k1_AXP_s110
                                 + 0.547122 w_k1_DIS_s110
                                 + 0.653263 w_k1_FDX_s110
                                 + 1.05377 w_k1_IBM_s110
                                 + 0.81477 w_k1_INTC_s110
                                 + 0.659578 w_k1_KO_s110
                                 + 1.16684 w_k1_MAT_s110
                                 + 0.778798 w_k1_MSFT_s110
                                 + 0.739631 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.741114 w_k1_ADM_s110
                                 + 0.857449 w_k1_AXP_s110
                                 + 0.936538 w_k1_DIS_s110
                                 + 0.8066 w_k1_FDX_s110 + 1.0924 w_k1_IBM_s110
                                 + 0.892541 w_k1_INTC_s110
                                 + 0.883068 w_k1_KO_s110
                                 + 1.01783 w_k1_MAT_s110
                                 + 0.85402 w_k1_MSFT_s110
                                 + 0.843297 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.18558 w_k1_ADM_s110
                                 + 1.51077 w_k1_AXP_s110
                                 + 1.33233 w_k1_DIS_s110
                                 + 1.09922 w_k1_FDX_s110
                                 + 1.27616 w_k1_IBM_s110
                                 + 0.907998 w_k1_INTC_s110
                                 + 1.09496 w_k1_KO_s110 + 1.79729 w_k1_MAT_s110
                                 + 1.19745 w_k1_MSFT_s110
                                 + 1.11526 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.04003 w_k1_ADM_s110
                                 + 1.192 w_k1_AXP_s110 + 2.01214 w_k1_DIS_s110
                                 + 1.41492 w_k1_FDX_s110
                                 + 1.05887 w_k1_IBM_s110
                                 + 1.25895 w_k1_INTC_s110
                                 + 1.16097 w_k1_KO_s110
                                 + 0.898367 w_k1_MAT_s110
                                 + 1.41354 w_k1_MSFT_s110
                                 + 1.18279 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.839801 w_k1_ADM_s111
                                 + 0.741018 w_k1_AXP_s111
                                 + 0.625225 w_k1_DIS_s111
                                 + 0.782465 w_k1_FDX_s111
                                 + 0.896408 w_k1_IBM_s111
                                 + 0.931963 w_k1_INTC_s111
                                 + 0.745239 w_k1_KO_s111
                                 + 0.831868 w_k1_MAT_s111
                                 + 0.831769 w_k1_MSFT_s111
                                 + 0.889034 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.878653 w_k1_ADM_s111
                                 + 0.968176 w_k1_AXP_s111
                                 + 1.3404 w_k1_DIS_s111
                                 + 0.987686 w_k1_FDX_s111
                                 + 1.44727 w_k1_IBM_s111
                                 + 1.01617 w_k1_INTC_s111
                                 + 1.25126 w_k1_KO_s111 + 1.39188 w_k1_MAT_s111
                                 + 1.14399 w_k1_MSFT_s111
                                 + 1.13964 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.23703 w_k1_ADM_s111
                                 + 1.19709 w_k1_AXP_s111 + 1.3046 w_k1_DIS_s111
                                 + 1.14483 w_k1_FDX_s111
                                 + 1.03516 w_k1_IBM_s111
                                 + 1.28705 w_k1_INTC_s111
                                 + 1.10733 w_k1_KO_s111 + 1.24735 w_k1_MAT_s111
                                 + 1.22984 w_k1_MSFT_s111
                                 + 1.11746 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.927406 w_k1_ADM_s111
                                 + 0.969996 w_k1_AXP_s111
                                 + 1.07773 w_k1_DIS_s111
                                 + 1.05775 w_k1_FDX_s111
                                 + 0.730892 w_k1_IBM_s111
                                 + 0.88782 w_k1_INTC_s111
                                 + 0.902487 w_k1_KO_s111
                                 + 0.980501 w_k1_MAT_s111
                                 + 0.9966 w_k1_MSFT_s111 + 1.02204 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.11062 w_k1_ADM_s111
                                 + 0.933357 w_k1_AXP_s111
                                 + 1.5406 w_k1_DIS_s111
                                 + 0.988778 w_k1_FDX_s111
                                 + 1.06011 w_k1_IBM_s111
                                 + 1.11941 w_k1_INTC_s111
                                 + 1.18982 w_k1_KO_s111 + 1.07318 w_k1_MAT_s111
                                 + 1.12297 w_k1_MSFT_s111
                                 + 1.14926 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.60318 w_k1_ADM_s111
                                 + 1.53102 w_k1_AXP_s111
                                 + 1.52042 w_k1_DIS_s111
                                 + 1.36552 w_k1_FDX_s111
                                 + 1.13666 w_k1_IBM_s111
                                 + 0.995954 w_k1_INTC_s111
                                 + 1.19961 w_k1_KO_s111 + 1.67824 w_k1_MAT_s111
                                 + 1.18395 w_k1_MSFT_s111
                                 + 1.11565 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.06626 w_k1_ADM_s111
                                 + 1.4301 w_k1_AXP_s111 + 1.00845 w_k1_DIS_s111
                                 + 1.07489 w_k1_FDX_s111
                                 + 1.35761 w_k1_IBM_s111
                                 + 1.09991 w_k1_INTC_s111
                                 + 1.16922 w_k1_KO_s111 + 2.16446 w_k1_MAT_s111
                                 + 1.05706 w_k1_MSFT_s111
                                 + 0.993303 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.22225 w_k1_ADM_s111
                                 + 1.11846 w_k1_AXP_s111
                                 + 0.874861 w_k1_DIS_s111
                                 + 0.957331 w_k1_FDX_s111
                                 + 0.861312 w_k1_IBM_s111
                                 + 0.900863 w_k1_INTC_s111
                                 + 0.771747 w_k1_KO_s111
                                 + 0.730827 w_k1_MAT_s111
                                 + 0.88515 w_k1_MSFT_s111
                                 + 0.945623 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.902779 w_k1_ADM_s112
                                 + 0.754959 w_k1_AXP_s112
                                 + 0.863945 w_k1_DIS_s112
                                 + 0.848895 w_k1_FDX_s112
                                 + 1.13419 w_k1_IBM_s112
                                 + 0.881247 w_k1_INTC_s112
                                 + 1.06668 w_k1_KO_s112
                                 + 0.917904 w_k1_MAT_s112
                                 + 0.879643 w_k1_MSFT_s112
                                 + 0.920141 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.764366 w_k1_ADM_s112
                                 + 0.893162 w_k1_AXP_s112
                                 + 1.02199 w_k1_DIS_s112
                                 + 0.981753 w_k1_FDX_s112
                                 + 0.934979 w_k1_IBM_s112
                                 + 1.04821 w_k1_INTC_s112
                                 + 1.07562 w_k1_KO_s112 + 1.14518 w_k1_MAT_s112
                                 + 1.11325 w_k1_MSFT_s112
                                 + 0.948755 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.05987 w_k1_ADM_s112
                                 + 0.667348 w_k1_AXP_s112
                                 + 0.864041 w_k1_DIS_s112
                                 + 0.884876 w_k1_FDX_s112
                                 + 0.772809 w_k1_IBM_s112
                                 + 0.897389 w_k1_INTC_s112
                                 + 0.860477 w_k1_KO_s112
                                 + 0.659065 w_k1_MAT_s112
                                 + 0.952579 w_k1_MSFT_s112
                                 + 0.806403 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.5539 w_k1_ADM_s112
                                 + 1.69996 w_k1_AXP_s112
                                 + 1.95832 w_k1_DIS_s112
                                 + 1.50572 w_k1_FDX_s112
                                 + 1.52409 w_k1_IBM_s112
                                 + 1.39725 w_k1_INTC_s112
                                 + 1.70139 w_k1_KO_s112 + 2.23206 w_k1_MAT_s112
                                 + 1.55547 w_k1_MSFT_s112
                                 + 1.32192 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.904996 w_k1_ADM_s112
                                 + 0.7965 w_k1_AXP_s112
                                 + 0.659848 w_k1_DIS_s112
                                 + 0.830208 w_k1_FDX_s112
                                 + 0.644329 w_k1_IBM_s112
                                 + 0.941348 w_k1_INTC_s112
                                 + 0.74091 w_k1_KO_s112
                                 + 0.542715 w_k1_MAT_s112
                                 + 0.907456 w_k1_MSFT_s112
                                 + 0.780247 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.34682 w_k1_ADM_s112
                                 + 1.01551 w_k1_AXP_s112
                                 + 1.22244 w_k1_DIS_s112
                                 + 1.23022 w_k1_FDX_s112
                                 + 0.702508 w_k1_IBM_s112
                                 + 0.985316 w_k1_INTC_s112
                                 + 1.02014 w_k1_KO_s112
                                 + 0.894672 w_k1_MAT_s112
                                 + 1.15032 w_k1_MSFT_s112
                                 + 1.05744 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.694431 w_k1_ADM_s112
                                 + 0.592787 w_k1_AXP_s112
                                 + 0.623682 w_k1_DIS_s112
                                 + 0.730638 w_k1_FDX_s112
                                 + 0.804831 w_k1_IBM_s112
                                 + 0.795184 w_k1_INTC_s112
                                 + 0.79617 w_k1_KO_s112 + 0.78745 w_k1_MAT_s112
                                 + 0.784456 w_k1_MSFT_s112
                                 + 0.758148 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.28427 w_k1_ADM_s112
                                 + 1.10663 w_k1_AXP_s112
                                 + 1.34531 w_k1_DIS_s112
                                 + 1.21091 w_k1_FDX_s112
                                 + 1.19988 w_k1_IBM_s112
                                 + 1.26264 w_k1_INTC_s112
                                 + 1.34083 w_k1_KO_s112 + 1.39267 w_k1_MAT_s112
                                 + 1.26961 w_k1_MSFT_s112
                                 + 1.05975 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.13297 w_k1_ADM_s113
                                 + 0.899047 w_k1_AXP_s113
                                 + 1.05028 w_k1_DIS_s113
                                 + 1.01746 w_k1_FDX_s113
                                 + 0.856512 w_k1_IBM_s113
                                 + 0.98312 w_k1_INTC_s113
                                 + 0.886514 w_k1_KO_s113
                                 + 0.749846 w_k1_MAT_s113
                                 + 1.03085 w_k1_MSFT_s113
                                 + 1.01977 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.769534 w_k1_ADM_s113
                                 + 0.830443 w_k1_AXP_s113
                                 + 0.915864 w_k1_DIS_s113
                                 + 0.848521 w_k1_FDX_s113
                                 + 0.950171 w_k1_IBM_s113
                                 + 0.754114 w_k1_INTC_s113
                                 + 0.821875 w_k1_KO_s113
                                 + 0.916971 w_k1_MAT_s113
                                 + 0.800563 w_k1_MSFT_s113
                                 + 0.933779 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.14547 w_k1_ADM_s113
                                 + 0.865269 w_k1_AXP_s113
                                 + 1.19764 w_k1_DIS_s113
                                 + 1.25758 w_k1_FDX_s113
                                 + 0.685087 w_k1_IBM_s113
                                 + 0.983111 w_k1_INTC_s113
                                 + 0.853763 w_k1_KO_s113
                                 + 0.573473 w_k1_MAT_s113
                                 + 1.14688 w_k1_MSFT_s113
                                 + 0.842898 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.909354 w_k1_ADM_s113
                                 + 0.998548 w_k1_AXP_s113
                                 + 1.04828 w_k1_DIS_s113
                                 + 1.03617 w_k1_FDX_s113
                                 + 1.30845 w_k1_IBM_s113
                                 + 0.909093 w_k1_INTC_s113
                                 + 0.928009 w_k1_KO_s113
                                 + 0.977365 w_k1_MAT_s113
                                 + 0.965594 w_k1_MSFT_s113
                                 + 0.965137 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.00699 w_k1_ADM_s113
                                 + 0.563849 w_k1_AXP_s113
                                 + 0.649518 w_k1_DIS_s113
                                 + 0.730656 w_k1_FDX_s113
                                 + 0.622452 w_k1_IBM_s113
                                 + 0.545766 w_k1_INTC_s113
                                 + 0.719801 w_k1_KO_s113
                                 + 0.611551 w_k1_MAT_s113
                                 + 0.706101 w_k1_MSFT_s113
                                 + 0.800431 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.10739 w_k1_ADM_s113
                                 + 1.24621 w_k1_AXP_s113
                                 + 1.56354 w_k1_DIS_s113
                                 + 1.23113 w_k1_FDX_s113
                                 + 1.17212 w_k1_IBM_s113
                                 + 1.26107 w_k1_INTC_s113 + 1.3008 w_k1_KO_s113
                                 + 1.13206 w_k1_MAT_s113
                                 + 1.27606 w_k1_MSFT_s113
                                 + 1.26702 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.28207 w_k1_ADM_s113
                                 + 1.60872 w_k1_AXP_s113
                                 + 1.32405 w_k1_DIS_s113
                                 + 1.21667 w_k1_FDX_s113
                                 + 1.66053 w_k1_IBM_s113
                                 + 0.920679 w_k1_INTC_s113
                                 + 1.29252 w_k1_KO_s113 + 1.52082 w_k1_MAT_s113
                                 + 1.12131 w_k1_MSFT_s113
                                 + 1.23634 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.924172 w_k1_ADM_s113
                                 + 0.78081 w_k1_AXP_s113
                                 + 1.06057 w_k1_DIS_s113
                                 + 1.02188 w_k1_FDX_s113
                                 + 0.640461 w_k1_IBM_s113
                                 + 0.808113 w_k1_INTC_s113
                                 + 0.771212 w_k1_KO_s113
                                 + 0.609704 w_k1_MAT_s113
                                 + 0.928545 w_k1_MSFT_s113
                                 + 0.8955 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.650177 w_k1_ADM_s114
                                 + 0.662825 w_k1_AXP_s114
                                 + 0.80517 w_k1_DIS_s114
                                 + 0.834342 w_k1_FDX_s114
                                 + 0.923999 w_k1_IBM_s114
                                 + 0.923337 w_k1_INTC_s114
                                 + 0.900417 w_k1_KO_s114
                                 + 0.800579 w_k1_MAT_s114
                                 + 0.939147 w_k1_MSFT_s114
                                 + 0.989571 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.34119 w_k1_ADM_s114
                                 + 1.3873 w_k1_AXP_s114 + 1.5966 w_k1_DIS_s114
                                 + 1.57881 w_k1_FDX_s114
                                 + 1.07168 w_k1_IBM_s114
                                 + 1.46557 w_k1_INTC_s114
                                 + 1.11493 w_k1_KO_s114
                                 + 0.954847 w_k1_MAT_s114
                                 + 1.50963 w_k1_MSFT_s114
                                 + 1.28507 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.994771 w_k1_ADM_s114
                                 + 1.25737 w_k1_AXP_s114
                                 + 1.43968 w_k1_DIS_s114
                                 + 1.13839 w_k1_FDX_s114
                                 + 1.41049 w_k1_IBM_s114
                                 + 1.24333 w_k1_INTC_s114
                                 + 1.20111 w_k1_KO_s114 + 1.18574 w_k1_MAT_s114
                                 + 1.19876 w_k1_MSFT_s114
                                 + 1.23345 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.12682 w_k1_ADM_s114
                                 + 1.12928 w_k1_AXP_s114
                                 + 1.09478 w_k1_DIS_s114
                                 + 1.15419 w_k1_FDX_s114
                                 + 1.02378 w_k1_IBM_s114
                                 + 0.942226 w_k1_INTC_s114
                                 + 1.02987 w_k1_KO_s114 + 1.04997 w_k1_MAT_s114
                                 + 1.12921 w_k1_MSFT_s114
                                 + 1.17904 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.850128 w_k1_ADM_s114
                                 + 1.0057 w_k1_AXP_s114 + 1.20972 w_k1_DIS_s114
                                 + 1.06522 w_k1_FDX_s114
                                 + 1.85629 w_k1_IBM_s114
                                 + 1.00702 w_k1_INTC_s114
                                 + 1.21705 w_k1_KO_s114 + 1.91562 w_k1_MAT_s114
                                 + 1.22645 w_k1_MSFT_s114
                                 + 1.36615 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.812558 w_k1_ADM_s114
                                 + 0.754033 w_k1_AXP_s114
                                 + 0.831046 w_k1_DIS_s114
                                 + 1.0027 w_k1_FDX_s114
                                 + 0.712145 w_k1_IBM_s114
                                 + 0.921807 w_k1_INTC_s114
                                 + 0.731567 w_k1_KO_s114
                                 + 0.637888 w_k1_MAT_s114
                                 + 1.06244 w_k1_MSFT_s114
                                 + 0.985241 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.869724 w_k1_ADM_s114
                                 + 0.801326 w_k1_AXP_s114
                                 + 1.0951 w_k1_DIS_s114
                                 + 0.927762 w_k1_FDX_s114
                                 + 1.26937 w_k1_IBM_s114
                                 + 0.871463 w_k1_INTC_s114
                                 + 1.01872 w_k1_KO_s114 + 1.08715 w_k1_MAT_s114
                                 + 1.02634 w_k1_MSFT_s114
                                 + 1.12195 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.641731 w_k1_ADM_s114
                                 + 0.605309 w_k1_AXP_s114
                                 + 0.626099 w_k1_DIS_s114
                                 + 0.766764 w_k1_FDX_s114
                                 + 0.675647 w_k1_IBM_s114
                                 + 0.677956 w_k1_INTC_s114
                                 + 0.668717 w_k1_KO_s114
                                 + 0.615426 w_k1_MAT_s114
                                 + 0.811099 w_k1_MSFT_s114
                                 + 0.754333 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.2889 w_k1_ADM_s115
                                 + 0.909487 w_k1_AXP_s115
                                 + 0.928833 w_k1_DIS_s115
                                 + 1.03898 w_k1_FDX_s115
                                 + 1.09625 w_k1_IBM_s115
                                 + 1.06443 w_k1_INTC_s115
                                 + 1.19615 w_k1_KO_s115 + 1.01438 w_k1_MAT_s115
                                 + 1.01403 w_k1_MSFT_s115
                                 + 1.01394 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.85666 w_k1_ADM_s115
                                 + 0.753725 w_k1_AXP_s115
                                 + 0.59236 w_k1_DIS_s115
                                 + 0.801037 w_k1_FDX_s115
                                 + 0.697978 w_k1_IBM_s115
                                 + 0.914996 w_k1_INTC_s115
                                 + 0.714863 w_k1_KO_s115
                                 + 0.700619 w_k1_MAT_s115
                                 + 0.861579 w_k1_MSFT_s115
                                 + 0.784194 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.44317 w_k1_ADM_s115
                                 + 1.44199 w_k1_AXP_s115
                                 + 1.02365 w_k1_DIS_s115
                                 + 1.09472 w_k1_FDX_s115
                                 + 1.15669 w_k1_IBM_s115
                                 + 1.02082 w_k1_INTC_s115
                                 + 1.22299 w_k1_KO_s115 + 1.36944 w_k1_MAT_s115
                                 + 0.925338 w_k1_MSFT_s115
                                 + 1.01018 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.23951 w_k1_ADM_s115
                                 + 1.42719 w_k1_AXP_s115 + 1.5602 w_k1_DIS_s115
                                 + 1.36609 w_k1_FDX_s115
                                 + 1.13373 w_k1_IBM_s115
                                 + 1.2915 w_k1_INTC_s115 + 1.40804 w_k1_KO_s115
                                 + 1.60995 w_k1_MAT_s115
                                 + 1.29498 w_k1_MSFT_s115
                                 + 1.14086 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.12417 w_k1_ADM_s115
                                 + 0.898804 w_k1_AXP_s115
                                 + 1.09087 w_k1_DIS_s115
                                 + 0.917676 w_k1_FDX_s115
                                 + 1.41046 w_k1_IBM_s115
                                 + 1.06661 w_k1_INTC_s115
                                 + 1.34984 w_k1_KO_s115 + 1.45942 w_k1_MAT_s115
                                 + 1.04592 w_k1_MSFT_s115
                                 + 0.959278 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.64833 w_k1_ADM_s115
                                 + 1.12718 w_k1_AXP_s115
                                 + 1.05286 w_k1_DIS_s115
                                 + 1.13385 w_k1_FDX_s115
                                 + 0.833147 w_k1_IBM_s115
                                 + 1.07789 w_k1_INTC_s115
                                 + 0.936637 w_k1_KO_s115
                                 + 0.729609 w_k1_MAT_s115
                                 + 1.09227 w_k1_MSFT_s115
                                 + 1.01144 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.905614 w_k1_ADM_s115
                                 + 0.707728 w_k1_AXP_s115
                                 + 0.738034 w_k1_DIS_s115
                                 + 0.802385 w_k1_FDX_s115
                                 + 0.872722 w_k1_IBM_s115
                                 + 0.967376 w_k1_INTC_s115
                                 + 1.08481 w_k1_KO_s115
                                 + 0.903157 w_k1_MAT_s115
                                 + 0.92846 w_k1_MSFT_s115
                                 + 0.881588 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.72136 w_k1_ADM_s115
                                 + 0.6552 w_k1_AXP_s115
                                 + 0.539586 w_k1_DIS_s115
                                 + 0.661052 w_k1_FDX_s115
                                 + 1.09079 w_k1_IBM_s115
                                 + 0.745992 w_k1_INTC_s115
                                 + 1.01331 w_k1_KO_s115 + 1.16843 w_k1_MAT_s115
                                 + 0.739446 w_k1_MSFT_s115
                                 + 0.772737 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.0783 w_k1_ADM_s116
                                 + 0.84178 w_k1_AXP_s116
                                 + 1.18494 w_k1_DIS_s116
                                 + 1.02129 w_k1_FDX_s116
                                 + 0.908728 w_k1_IBM_s116
                                 + 1.07157 w_k1_INTC_s116
                                 + 1.18527 w_k1_KO_s116
                                 + 0.769036 w_k1_MAT_s116
                                 + 1.03721 w_k1_MSFT_s116
                                 + 1.02396 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.03749 w_k1_ADM_s116
                                 + 1.1841 w_k1_AXP_s116 + 1.23938 w_k1_DIS_s116
                                 + 1.23922 w_k1_FDX_s116
                                 + 0.90762 w_k1_IBM_s116
                                 + 1.18576 w_k1_INTC_s116
                                 + 1.23398 w_k1_KO_s116
                                 + 0.952368 w_k1_MAT_s116
                                 + 1.1243 w_k1_MSFT_s116 + 0.99723 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.37574 w_k1_ADM_s116
                                 + 1.4676 w_k1_AXP_s116 + 1.94513 w_k1_DIS_s116
                                 + 1.57205 w_k1_FDX_s116
                                 + 1.01329 w_k1_IBM_s116
                                 + 1.23008 w_k1_INTC_s116 + 1.4587 w_k1_KO_s116
                                 + 1.36469 w_k1_MAT_s116
                                 + 1.50071 w_k1_MSFT_s116
                                 + 1.10118 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.01677 w_k1_ADM_s116
                                 + 1.17024 w_k1_AXP_s116
                                 + 1.17737 w_k1_DIS_s116
                                 + 1.04743 w_k1_FDX_s116
                                 + 1.30116 w_k1_IBM_s116
                                 + 0.964969 w_k1_INTC_s116
                                 + 1.21119 w_k1_KO_s116 + 1.6807 w_k1_MAT_s116
                                 + 0.987386 w_k1_MSFT_s116
                                 + 1.01228 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.804363 w_k1_ADM_s116
                                 + 0.722798 w_k1_AXP_s116
                                 + 0.755132 w_k1_DIS_s116
                                 + 0.800212 w_k1_FDX_s116
                                 + 0.654048 w_k1_IBM_s116
                                 + 0.905188 w_k1_INTC_s116
                                 + 0.811899 w_k1_KO_s116
                                 + 0.582222 w_k1_MAT_s116
                                 + 0.83071 w_k1_MSFT_s116
                                 + 0.801904 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.01699 w_k1_ADM_s116
                                 + 1.34936 w_k1_AXP_s116
                                 + 1.78877 w_k1_DIS_s116
                                 + 1.25459 w_k1_FDX_s116
                                 + 1.59729 w_k1_IBM_s116
                                 + 1.17631 w_k1_INTC_s116
                                 + 1.53331 w_k1_KO_s116 + 2.24397 w_k1_MAT_s116
                                 + 1.3009 w_k1_MSFT_s116 + 1.10486 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.751639 w_k1_ADM_s116
                                 + 0.724289 w_k1_AXP_s116
                                 + 0.740961 w_k1_DIS_s116
                                 + 0.789864 w_k1_FDX_s116
                                 + 1.00203 w_k1_IBM_s116
                                 + 0.75004 w_k1_INTC_s116
                                 + 0.933171 w_k1_KO_s116
                                 + 1.41011 w_k1_MAT_s116
                                 + 0.840982 w_k1_MSFT_s116
                                 + 0.92278 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.21288 w_k1_ADM_s116
                                 + 0.983138 w_k1_AXP_s116
                                 + 1.10808 w_k1_DIS_s116
                                 + 0.925119 w_k1_FDX_s116
                                 + 1.5892 w_k1_IBM_s116
                                 + 0.885224 w_k1_INTC_s116
                                 + 1.45744 w_k1_KO_s116 + 1.8032 w_k1_MAT_s116
                                 + 1.04364 w_k1_MSFT_s116
                                 + 1.05233 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.713817 w_k1_ADM_s117
                                 + 0.715772 w_k1_AXP_s117
                                 + 0.787293 w_k1_DIS_s117
                                 + 0.778287 w_k1_FDX_s117
                                 + 0.918928 w_k1_IBM_s117
                                 + 0.877849 w_k1_INTC_s117
                                 + 0.868631 w_k1_KO_s117
                                 + 0.935347 w_k1_MAT_s117
                                 + 0.861931 w_k1_MSFT_s117
                                 + 0.87604 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.09949 w_k1_ADM_s117
                                 + 1.34581 w_k1_AXP_s117
                                 + 1.51288 w_k1_DIS_s117
                                 + 1.41029 w_k1_FDX_s117
                                 + 0.88699 w_k1_IBM_s117
                                 + 1.72346 w_k1_INTC_s117
                                 + 1.21582 w_k1_KO_s117 + 1.32488 w_k1_MAT_s117
                                 + 1.43724 w_k1_MSFT_s117 + 1.148 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.822087 w_k1_ADM_s117
                                 + 1.03771 w_k1_AXP_s117
                                 + 1.23142 w_k1_DIS_s117 + 1.1169 w_k1_FDX_s117
                                 + 1.28691 w_k1_IBM_s117
                                 + 1.01694 w_k1_INTC_s117
                                 + 1.36149 w_k1_KO_s117 + 1.53833 w_k1_MAT_s117
                                 + 1.18094 w_k1_MSFT_s117
                                 + 1.06189 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.709736 w_k1_ADM_s117
                                 + 0.924034 w_k1_AXP_s117
                                 + 0.668468 w_k1_DIS_s117
                                 + 0.921345 w_k1_FDX_s117
                                 + 0.671569 w_k1_IBM_s117
                                 + 0.849921 w_k1_INTC_s117
                                 + 0.774224 w_k1_KO_s117
                                 + 1.17159 w_k1_MAT_s117
                                 + 0.879004 w_k1_MSFT_s117
                                 + 0.86217 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.998239 w_k1_ADM_s117
                                 + 0.998091 w_k1_AXP_s117
                                 + 0.934145 w_k1_DIS_s117
                                 + 1.07639 w_k1_FDX_s117
                                 + 0.863182 w_k1_IBM_s117
                                 + 1.23538 w_k1_INTC_s117
                                 + 0.850815 w_k1_KO_s117 + 1.0812 w_k1_MAT_s117
                                 + 1.07493 w_k1_MSFT_s117
                                 + 0.834114 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.78378 w_k1_ADM_s117
                                 + 1.01094 w_k1_AXP_s117
                                 + 0.841779 w_k1_DIS_s117
                                 + 0.819814 w_k1_FDX_s117
                                 + 1.24905 w_k1_IBM_s117
                                 + 0.873812 w_k1_INTC_s117
                                 + 1.10422 w_k1_KO_s117 + 1.88529 w_k1_MAT_s117
                                 + 0.943907 w_k1_MSFT_s117
                                 + 0.956949 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.2128 w_k1_ADM_s117
                                 + 1.24789 w_k1_AXP_s117
                                 + 1.39507 w_k1_DIS_s117
                                 + 1.26965 w_k1_FDX_s117
                                 + 1.13505 w_k1_IBM_s117
                                 + 1.0841 w_k1_INTC_s117 + 1.40925 w_k1_KO_s117
                                 + 1.64794 w_k1_MAT_s117
                                 + 1.22531 w_k1_MSFT_s117
                                 + 1.16332 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.884627 w_k1_ADM_s117
                                 + 1.25654 w_k1_AXP_s117
                                 + 1.15673 w_k1_DIS_s117
                                 + 0.996446 w_k1_FDX_s117
                                 + 1.6671 w_k1_IBM_s117 + 1.0011 w_k1_INTC_s117
                                 + 1.31511 w_k1_KO_s117 + 2.25127 w_k1_MAT_s117
                                 + 0.973958 w_k1_MSFT_s117
                                 + 1.0331 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0464428904256172 w_k1_ADM_s000
                                 + 0.0424437587170136 w_k1_ADM_s001
                                 + 0.0212924068149873 w_k1_ADM_s002
                                 + 0.0265110164190914 w_k1_ADM_s003
                                 + 0.0356564695325946 w_k1_ADM_s004
                                 + 0.0272574198536336 w_k1_ADM_s005
                                 + 0.0623265852079336 w_k1_ADM_s006
                                 + 0.0269951083462407 w_k1_ADM_s007
                                 + 0.0340275754939148 w_k1_ADM_s010
                                 + 0.0536283828428562 w_k1_ADM_s011
                                 + 0.0519078294250871 w_k1_ADM_s012
                                 + 0.0611617536275827 w_k1_ADM_s013
                                 + 0.0391319225934133 w_k1_ADM_s014
                                 + 0.037189331342195 w_k1_ADM_s015
                                 + 0.0268153190513963 w_k1_ADM_s016
                                 + 0.0794662286340167 w_k1_ADM_s017
                                 + 0.0171272064561125 w_k1_ADM_s100
                                 + 0.0132875099185431 w_k1_ADM_s101
                                 + 0.0111039179714478 w_k1_ADM_s102
                                 + 0.0186638146185898 w_k1_ADM_s103
                                 + 0.0126157449554146 w_k1_ADM_s104
                                 + 0.0566628883331491 w_k1_ADM_s105
                                 + 0.0195860581426772 w_k1_ADM_s106
                                 + 0.0161825865492122 w_k1_ADM_s107
                                 + 0.0239878190581758 w_k1_ADM_s110
                                 + 0.0233472912237738 w_k1_ADM_s111
                                 + 0.0462053171946418 w_k1_ADM_s112
                                 + 0.0101154720095795 w_k1_ADM_s113
                                 + 0.0128647045901309 w_k1_ADM_s114
                                 + 0.0339372320961258 w_k1_ADM_s115
                                 + 0.0126995633931516 w_k1_ADM_s116
                                 + 0.0236757183953795 w_k1_ADM_s117
                                 + 0.0478562605428571 w_k1_AXP_s000
                                 + 0.0510265140170894 w_k1_AXP_s001
                                 + 0.0167354628467886 w_k1_AXP_s002
                                 + 0.031439774894705 w_k1_AXP_s003
                                 + 0.0333865496862955 w_k1_AXP_s004
                                 + 0.0289743378982776 w_k1_AXP_s005
                                 + 0.0606633673342246 w_k1_AXP_s006
                                 + 0.0322243428416637 w_k1_AXP_s007
                                 + 0.0362388225042321 w_k1_AXP_s010
                                 + 0.0572152200603103 w_k1_AXP_s011
                                 + 0.0456861137011615 w_k1_AXP_s012
                                 + 0.0587248400345567 w_k1_AXP_s013
                                 + 0.0406873292948761 w_k1_AXP_s014
                                 + 0.0341877252606517 w_k1_AXP_s015
                                 + 0.0242250023716841 w_k1_AXP_s016
                                 + 0.0722841378242193 w_k1_AXP_s017
                                 + 0.0197270349695393 w_k1_AXP_s100
                                 + 0.0112363961047135 w_k1_AXP_s101
                                 + 0.0107825694795364 w_k1_AXP_s102
                                 + 0.0213062472134232 w_k1_AXP_s103
                                 + 0.0142171241225124 w_k1_AXP_s104
                                 + 0.0482991964268389 w_k1_AXP_s105
                                 + 0.0178858883631094 w_k1_AXP_s106
                                 + 0.0153309585391337 w_k1_AXP_s107
                                 + 0.0265533839587416 w_k1_AXP_s110
                                 + 0.0232499398792886 w_k1_AXP_s111
                                 + 0.0410987209103568 w_k1_AXP_s112
                                 + 0.00958625768141075 w_k1_AXP_s113
                                 + 0.0132474706523685 w_k1_AXP_s114
                                 + 0.0291200305595381 w_k1_AXP_s115
                                 + 0.0126407870991965 w_k1_AXP_s116
                                 + 0.0268457935192338 w_k1_AXP_s117
                                 + 0.0476086618398056 w_k1_DIS_s000
                                 + 0.0524121234475262 w_k1_DIS_s001
                                 + 0.019157980329205 w_k1_DIS_s002
                                 + 0.0298328838973769 w_k1_DIS_s003
                                 + 0.035411249990084 w_k1_DIS_s004
                                 + 0.0286299499308874 w_k1_DIS_s005
                                 + 0.0637725318141834 w_k1_DIS_s006
                                 + 0.0332811825394501 w_k1_DIS_s007
                                 + 0.033833005148456 w_k1_DIS_s010
                                 + 0.0568182022622609 w_k1_DIS_s011
                                 + 0.0483798698457801 w_k1_DIS_s012
                                 + 0.0633496439493732 w_k1_DIS_s013
                                 + 0.0388584434894167 w_k1_DIS_s014
                                 + 0.0338752876657102 w_k1_DIS_s015
                                 + 0.0247500080971775 w_k1_DIS_s016
                                 + 0.0740891709426354 w_k1_DIS_s017
                                 + 0.0201050815553733 w_k1_DIS_s100
                                 + 0.0121010224462268 w_k1_DIS_s101
                                 + 0.00965785271569125 w_k1_DIS_s102
                                 + 0.0205908283073001 w_k1_DIS_s103
                                 + 0.016744131299269 w_k1_DIS_s104
                                 + 0.0506283848043913 w_k1_DIS_s105
                                 + 0.0194004173821381 w_k1_DIS_s106
                                 + 0.0166162666469954 w_k1_DIS_s107
                                 + 0.0288734652166 w_k1_DIS_s110
                                 + 0.0231856129033971 w_k1_DIS_s111
                                 + 0.047638384865329 w_k1_DIS_s112
                                 + 0.0108685943225554 w_k1_DIS_s113
                                 + 0.0153638898879277 w_k1_DIS_s114
                                 + 0.0282078086941205 w_k1_DIS_s115
                                 + 0.0144046554491426 w_k1_DIS_s116
                                 + 0.0267440394027009 w_k1_DIS_s117
                                 + 0.0477135909995132 w_k1_FDX_s000
                                 + 0.0497632144504917 w_k1_FDX_s001
                                 + 0.0189059805814524 w_k1_FDX_s002
                                 + 0.0288667559697014 w_k1_FDX_s003
                                 + 0.0327148508440469 w_k1_FDX_s004
                                 + 0.0253187219591251 w_k1_FDX_s005
                                 + 0.0630646144714499 w_k1_FDX_s006
                                 + 0.0316397685175041 w_k1_FDX_s007
                                 + 0.0317147674396006 w_k1_FDX_s010
                                 + 0.0553328126537947 w_k1_FDX_s011
                                 + 0.0471070768461203 w_k1_FDX_s012
                                 + 0.0591944840263761 w_k1_FDX_s013
                                 + 0.037370085907954 w_k1_FDX_s014
                                 + 0.0333369793798796 w_k1_FDX_s015
                                 + 0.0253933313673233 w_k1_FDX_s016
                                 + 0.0741872156264148 w_k1_FDX_s017
                                 + 0.0178979286199959 w_k1_FDX_s100
                                 + 0.0118597573483172 w_k1_FDX_s101
                                 + 0.0100451752463685 w_k1_FDX_s102
                                 + 0.0207174539862699 w_k1_FDX_s103
                                 + 0.0149297955374108 w_k1_FDX_s104
                                 + 0.0472531656728824 w_k1_FDX_s105
                                 + 0.0203777536888872 w_k1_FDX_s106
                                 + 0.0157368933568933 w_k1_FDX_s107
                                 + 0.0255893657463298 w_k1_FDX_s110
                                 + 0.0220943473430082 w_k1_FDX_s111
                                 + 0.0453862632983334 w_k1_FDX_s112
                                 + 0.0102030574101023 w_k1_FDX_s113
                                 + 0.0149972102369151 w_k1_FDX_s114
                                 + 0.0284220698853364 w_k1_FDX_s115
                                 + 0.0133293770616527 w_k1_FDX_s116
                                 + 0.0271192895481604 w_k1_FDX_s117
                                 + 0.0505578101474505 w_k1_IBM_s000
                                 + 0.0533134336059453 w_k1_IBM_s001
                                 + 0.0190630171603794 w_k1_IBM_s002
                                 + 0.0309636498755868 w_k1_IBM_s003
                                 + 0.0329884857635426 w_k1_IBM_s004
                                 + 0.0265464117654793 w_k1_IBM_s005
                                 + 0.0640636389701969 w_k1_IBM_s006
                                 + 0.0325889572485352 w_k1_IBM_s007
                                 + 0.0319359773088871 w_k1_IBM_s010
                                 + 0.0546169079726536 w_k1_IBM_s011
                                 + 0.048535804748499 w_k1_IBM_s012
                                 + 0.0586018730695301 w_k1_IBM_s013
                                 + 0.0388334851965322 w_k1_IBM_s014
                                 + 0.0345454538697631 w_k1_IBM_s015
                                 + 0.0250611746733836 w_k1_IBM_s016
                                 + 0.0744100086319376 w_k1_IBM_s017
                                 + 0.0175580118430085 w_k1_IBM_s100
                                 + 0.0120857763798207 w_k1_IBM_s101
                                 + 0.0100095906012165 w_k1_IBM_s102
                                 + 0.0224411637744619 w_k1_IBM_s103
                                 + 0.0151864710362692 w_k1_IBM_s104
                                 + 0.0494136345595274 w_k1_IBM_s105
                                 + 0.0194937608291127 w_k1_IBM_s106
                                 + 0.016568207230137 w_k1_IBM_s107
                                 + 0.0259615005154524 w_k1_IBM_s110
                                 + 0.0232016469588754 w_k1_IBM_s111
                                 + 0.0426695736117863 w_k1_IBM_s112
                                 + 0.0100003158899706 w_k1_IBM_s113
                                 + 0.015837779530804 w_k1_IBM_s114
                                 + 0.0298471649325892 w_k1_IBM_s115
                                 + 0.0133418746635892 w_k1_IBM_s116
                                 + 0.0271490463174927 w_k1_IBM_s117
                                 + 0.0448229464458811 w_k1_INTC_s000
                                 + 0.057483894276217 w_k1_INTC_s001
                                 + 0.0184906239446416 w_k1_INTC_s002
                                 + 0.0298970145095827 w_k1_INTC_s003
                                 + 0.0378028550766265 w_k1_INTC_s004
                                 + 0.0263927171708438 w_k1_INTC_s005
                                 + 0.0671256462067134 w_k1_INTC_s006
                                 + 0.0390416088240549 w_k1_INTC_s007
                                 + 0.0319439042627588 w_k1_INTC_s010
                                 + 0.0534035653370899 w_k1_INTC_s011
                                 + 0.0429182060388402 w_k1_INTC_s012
                                 + 0.0624863614284429 w_k1_INTC_s013
                                 + 0.0428949584655778 w_k1_INTC_s014
                                 + 0.0306502354621594 w_k1_INTC_s015
                                 + 0.0251579578897239 w_k1_INTC_s016
                                 + 0.0770022331639344 w_k1_INTC_s017
                                 + 0.0164071723079778 w_k1_INTC_s100
                                 + 0.0108332144634333 w_k1_INTC_s101
                                 + 0.00943547331875418 w_k1_INTC_s102
                                 + 0.0202798127023968 w_k1_INTC_s103
                                 + 0.0146886110960429 w_k1_INTC_s104
                                 + 0.0521938048669889 w_k1_INTC_s105
                                 + 0.0209941488558849 w_k1_INTC_s106
                                 + 0.0145048121902359 w_k1_INTC_s107
                                 + 0.0246091354870642 w_k1_INTC_s110
                                 + 0.0225490308006908 w_k1_INTC_s111
                                 + 0.0450708394587499 w_k1_INTC_s112
                                 + 0.00915024058589709 w_k1_INTC_s113
                                 + 0.0146011211908736 w_k1_INTC_s114
                                 + 0.0294707734627078 w_k1_INTC_s115
                                 + 0.0128715071705182 w_k1_INTC_s116
                                 + 0.0288568261834522 w_k1_INTC_s117
                                 + 0.0502893845255696 w_k1_KO_s000
                                 + 0.0497748741147685 w_k1_KO_s001
                                 + 0.0205984542876992 w_k1_KO_s002
                                 + 0.0304421843064179 w_k1_KO_s003
                                 + 0.031894332829107 w_k1_KO_s004
                                 + 0.0250248518938939 w_k1_KO_s005
                                 + 0.0621637246708817 w_k1_KO_s006
                                 + 0.031415465487519 w_k1_KO_s007
                                 + 0.0312079009871852 w_k1_KO_s010
                                 + 0.0499909447222096 w_k1_KO_s011
                                 + 0.0503206309183468 w_k1_KO_s012
                                 + 0.0617562150350121 w_k1_KO_s013
                                 + 0.0377795144562616 w_k1_KO_s014
                                 + 0.0359684967836466 w_k1_KO_s015
                                 + 0.0256066647164176 w_k1_KO_s016
                                 + 0.0709528004880057 w_k1_KO_s017
                                 + 0.0179135585222725 w_k1_KO_s100
                                 + 0.0129386860849341 w_k1_KO_s101
                                 + 0.00969875345984969 w_k1_KO_s102
                                 + 0.0202454904705594 w_k1_KO_s103
                                 + 0.0149516376183997 w_k1_KO_s104
                                 + 0.0543146323547229 w_k1_KO_s105
                                 + 0.0207354446822981 w_k1_KO_s106
                                 + 0.01566022353053 w_k1_KO_s107
                                 + 0.024207190723764 w_k1_KO_s110
                                 + 0.0218352144927527 w_k1_KO_s111
                                 + 0.0478765760882798 w_k1_KO_s112
                                 + 0.00938843957624756 w_k1_KO_s113
                                 + 0.0143497969297047 w_k1_KO_s114
                                 + 0.0318259761799206 w_k1_KO_s115
                                 + 0.0148270716825318 w_k1_KO_s116
                                 + 0.0275884418322665 w_k1_KO_s117
                                 + 0.0508157605535487 w_k1_MAT_s000
                                 + 0.0648860640041269 w_k1_MAT_s001
                                 + 0.0196930733559209 w_k1_MAT_s002
                                 + 0.0264664307276185 w_k1_MAT_s003
                                 + 0.0294330462010014 w_k1_MAT_s004
                                 + 0.0275539997334918 w_k1_MAT_s005
                                 + 0.0570630389598262 w_k1_MAT_s006
                                 + 0.0296370988114124 w_k1_MAT_s007
                                 + 0.0352716335005452 w_k1_MAT_s010
                                 + 0.0374565982077058 w_k1_MAT_s011
                                 + 0.0447920189528017 w_k1_MAT_s012
                                 + 0.0578950025143046 w_k1_MAT_s013
                                 + 0.0339847728065953 w_k1_MAT_s014
                                 + 0.0420781371740208 w_k1_MAT_s015
                                 + 0.0278014689425219 w_k1_MAT_s016
                                 + 0.0717770922103331 w_k1_MAT_s017
                                 + 0.017452427958402 w_k1_MAT_s100
                                 + 0.0127483345317777 w_k1_MAT_s101
                                 + 0.009214418568739 w_k1_MAT_s102
                                 + 0.0223215285844545 w_k1_MAT_s103
                                 + 0.0169114975574172 w_k1_MAT_s104
                                 + 0.0487571764937429 w_k1_MAT_s105
                                 + 0.0183873482637658 w_k1_MAT_s106
                                 + 0.0205423786638037 w_k1_MAT_s107
                                 + 0.0292134593572456 w_k1_MAT_s110
                                 + 0.0260031824111461 w_k1_MAT_s111
                                 + 0.0477954259350307 w_k1_MAT_s112
                                 + 0.00886464946412602 w_k1_MAT_s113
                                 + 0.0143324075071855 w_k1_MAT_s114
                                 + 0.0323567069326085 w_k1_MAT_s115
                                 + 0.015863314468278 w_k1_MAT_s116
                                 + 0.035670082786874 w_k1_MAT_s117
                                 + 0.0455505962267586 w_k1_MSFT_s000
                                 + 0.0496578926127052 w_k1_MSFT_s001
                                 + 0.0209391068562493 w_k1_MSFT_s002
                                 + 0.0295947397775111 w_k1_MSFT_s003
                                 + 0.0325211919246893 w_k1_MSFT_s004
                                 + 0.0249823323487621 w_k1_MSFT_s005
                                 + 0.0592325648759638 w_k1_MSFT_s006
                                 + 0.0312092198032432 w_k1_MSFT_s007
                                 + 0.0300148757078856 w_k1_MSFT_s010
                                 + 0.051956918635851 w_k1_MSFT_s011
                                 + 0.0477789969154648 w_k1_MSFT_s012
                                 + 0.0584007267559581 w_k1_MSFT_s013
                                 + 0.0376400601287677 w_k1_MSFT_s014
                                 + 0.0318581092018507 w_k1_MSFT_s015
                                 + 0.0255108162506167 w_k1_MSFT_s016
                                 + 0.077930185056814 w_k1_MSFT_s017
                                 + 0.0167862356237879 w_k1_MSFT_s100
                                 + 0.0115856222913215 w_k1_MSFT_s101
                                 + 0.00927044591473761 w_k1_MSFT_s102
                                 + 0.0200192326213287 w_k1_MSFT_s103
                                 + 0.0147758519239755 w_k1_MSFT_s104
                                 + 0.0500284309376672 w_k1_MSFT_s105
                                 + 0.0191836885336751 w_k1_MSFT_s106
                                 + 0.016404223568264 w_k1_MSFT_s107
                                 + 0.027179217001094 w_k1_MSFT_s110
                                 + 0.0225683621411899 w_k1_MSFT_s111
                                 + 0.0474060040596546 w_k1_MSFT_s112
                                 + 0.00989338542927512 w_k1_MSFT_s113
                                 + 0.0159013772822159 w_k1_MSFT_s114
                                 + 0.0287093786244451 w_k1_MSFT_s115
                                 + 0.0133373567086736 w_k1_MSFT_s116
                                 + 0.0279449863931184 w_k1_MSFT_s117
                                 + 0.0480960705734004 w_k1_PG_s000
                                 + 0.0489324635850931 w_k1_PG_s001
                                 + 0.0179665030579576 w_k1_PG_s002
                                 + 0.0283310046673965 w_k1_PG_s003
                                 + 0.0350494917323955 w_k1_PG_s004
                                 + 0.0265219942645749 w_k1_PG_s005
                                 + 0.060008219629241 w_k1_PG_s006
                                 + 0.0326543524764262 w_k1_PG_s007
                                 + 0.0326463331444815 w_k1_PG_s010
                                 + 0.0509095396657975 w_k1_PG_s011
                                 + 0.0488183233478995 w_k1_PG_s012
                                 + 0.0627890201101741 w_k1_PG_s013
                                 + 0.0371886052877372 w_k1_PG_s014
                                 + 0.0335165137518676 w_k1_PG_s015
                                 + 0.0244328056522706 w_k1_PG_s016
                                 + 0.0765920548914373 w_k1_PG_s017
                                 + 0.0178197040637998 w_k1_PG_s100
                                 + 0.0110949894161588 w_k1_PG_s101
                                 + 0.0109647960076859 w_k1_PG_s102
                                 + 0.0216940526034848 w_k1_PG_s103
                                 + 0.0151830824738299 w_k1_PG_s104
                                 + 0.0506942599470118 w_k1_PG_s105
                                 + 0.0192810244170852 w_k1_PG_s106
                                 + 0.0146426588175548 w_k1_PG_s107
                                 + 0.0243335018591489 w_k1_PG_s110
                                 + 0.0227022505291076 w_k1_PG_s111
                                 + 0.0421713390123195 w_k1_PG_s112
                                 + 0.0100306270770324 w_k1_PG_s113
                                 + 0.0161303483971542 w_k1_PG_s114
                                 + 0.0271183342910007 w_k1_PG_s115
                                 + 0.0127724872472352 w_k1_PG_s116
                                 + 0.0253132416809679 w_k1_PG_s117 >= 1.15
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
