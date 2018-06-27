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
 w_k1_ADM_s0_return_constraint:  1.11694 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.928098 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.958634 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.20118 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.808406 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.26913 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.44644 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.04546 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.07855 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.22769 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.16221 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.28677 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.928413 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.923732 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.02247 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.73236 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.929342 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.841026 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.2212 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.17944 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.09761 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.21093 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.01428 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.28605 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.880216 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.15702 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.928167 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.67796 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.23426 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.870016 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.754613 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.9254 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.06327 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.894785 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.81469 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.00603 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.96562 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.668775 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.31755 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.939654 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.02209 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.24754 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.800821 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.20193 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.26838 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.971836 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.10781 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.940952 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.0297 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.45236 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.66698 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.824546 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.04192 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.562794 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.06379 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.6865 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.47024 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.21207 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.55471 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.16131 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.976321 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.12866 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.67498 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.812248 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.954076 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.47277 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.1002 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.696738 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.6766 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.916786 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.37258 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.03969 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.78935 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.33193 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.0946 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.776838 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.41174 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.929981 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.911936 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.45334 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.775374 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.24711 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.63838 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.25684 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.12178 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.975342 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.931975 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.41052 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.697213 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.81696 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.875631 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.63179 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.980909 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.795371 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.66731 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.1664 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.56531 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.03743 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.03518 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.17666 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.60929 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.742384 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.952931 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.81629 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.32084 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.656581 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.54561 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.731138 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.46547 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.18033 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.807773 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.01358 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.05741 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.683024 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.2652 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.935564 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.96295 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.26988 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.869282 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.14441 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.4235 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.09969 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.965156 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.07862 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.856968 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.13722 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.767807 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.805552 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.860811 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.66805 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.816982 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.838588 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.4068 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.06931 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.20615 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.12109 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.993618 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.27092 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.778881 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.884889 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.808415 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.35034 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.15958 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.716839 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.759383 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.960106 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.16011 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.17111 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.836193 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.08297 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.19453 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.690079 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.2113 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.988504 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.994323 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.10202 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.811188 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.22362 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.05264 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.0055 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.14222 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.567054 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.43134 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.75021 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.755365 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.17586 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.08733 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.845963 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.52596 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.686081 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.06485 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.24892 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.86314 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.780909 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.867566 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.690862 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.533712 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.756192 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.62355 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.56797 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.16733 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.02395 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.691288 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.696779 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.43887 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.640682 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.973575 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.07846 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.807699 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.2593 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18311 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.973811 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.929219 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.09554 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.893324 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.999898 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.21208 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.22158 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.03444 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.04997 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.0905 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.18217 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.911989 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.923763 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.09764 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.729741 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.04129 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.03099 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.32864 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.09018 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.33913 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.18006 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.963096 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.22123 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.874264 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.911743 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.01165 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.40569 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.13377 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.841659 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.844388 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.936075 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.26845 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.33723 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.00934 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.15091 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.23094 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.851957 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.20924 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.908566 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.939769 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.23715 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.903011 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.32891 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.36173 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.09823 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.21474 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.824425 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.2598 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.56285 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.761528 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.10504 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.01657 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.791973 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.22617 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.871525 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.1401 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.11965 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.47909 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.935193 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.00402 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.915679 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.645357 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.788619 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.17595 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.57825 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.21055 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.835326 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.696112 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.737658 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.28227 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.841119 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.876773 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.956945 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.927711 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.950993 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.33145 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.928088 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.10602 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.09123 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.884016 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.36076 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.0685 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.732762 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.15737 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.503333 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.14041 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.70779 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.457227 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.05111 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.03608 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.789444 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.29803 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.616808 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.34604 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.18822 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.07696 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.7581 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.903783 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.726171 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.475056 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.662379 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.52246 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.55936 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.18024 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.80782 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.536744 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.739285 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.45741 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.503002 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.918463 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.37517 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.788197 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.1227 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.20187 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.949813 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.945266 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.22147 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.913265 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.11677 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.41102 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.17721 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.05967 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.02998 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.03888 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.07512 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.806393 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.81731 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.960848 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.72139 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.868657 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.984892 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.33398 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.00819 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.17593 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.07211 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.998946 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.1708 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.801199 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.811146 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.94923 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.34321 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.14976 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.798283 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.754639 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.903978 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.16178 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.12495 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.882167 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.05843 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.15086 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.784021 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.1394 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.94376 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.965997 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.2402 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.886032 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.13954 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.19215 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.02493 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.04858 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.946595 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.03066 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.32494 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.891438 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.992604 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.970424 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.783593 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.926525 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.851358 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.04234 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.961861 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.06161 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.969225 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.00248 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.12243 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.807448 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.914687 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.05731 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.33846 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.15536 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.908729 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.81414 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.876819 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.21037 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.960856 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.831259 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.02132 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.03874 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.875159 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.542001248291352 z_k1_s0000
                                 - 0.239000088754087 z_k1_s0001
                                 - 0.299001340864227 z_k1_s0002
                                 - 0.162000600287179 z_k1_s0003
                                 - 0.250000583108584 z_k1_s0004
                                 - 0.121000472881254 z_k1_s0005
                                 - 0.122000427549514 z_k1_s0006
                                 - 0.251001530780486 z_k1_s0007
                                 - 0.140000290839923 z_k1_s0010
                                 - 0.173000270466451 z_k1_s0011
                                 - 0.0809999726287557 z_k1_s0012
                                 - 0.0950000017127481 z_k1_s0013
                                 - 0.268999758470762 z_k1_s0014
                                 - 0.101000031963036 z_k1_s0015
                                 - 0.2169998283017 z_k1_s0016
                                 - 0.169000084299544 z_k1_s0017
                                 - 0.10299993222985 z_k1_s0020
                                 - 0.200999456435179 z_k1_s0021
                                 - 0.0989999076500655 z_k1_s0022
                                 - 0.104999944519743 z_k1_s0023
                                 - 0.13999978229307 z_k1_s0024
                                 - 0.106999956809635 z_k1_s0025
                                 - 0.119999659394146 z_k1_s0026
                                 - 0.203000223324653 z_k1_s0027
                                 - 0.509001345425608 z_k1_s0030
                                 - 0.867000144502793 z_k1_s0031
                                 - 0.364999374001073 z_k1_s0032
                                 - 0.181000082517359 z_k1_s0033
                                 - 0.185000359173816 z_k1_s0034
                                 - 0.39800156564175 z_k1_s0035
                                 - 0.861001726570083 z_k1_s0036
                                 - 0.265000082697201 z_k1_s0037
                                 - 0.201000689488546 z_k1_s0040
                                 - 0.190000060964484 z_k1_s0041
                                 - 0.326000321991625 z_k1_s0042
                                 - 0.107999944074577 z_k1_s0043
                                 - 0.282999429578888 z_k1_s0044
                                 - 0.248999364322102 z_k1_s0045
                                 - 0.181999134492587 z_k1_s0046
                                 - 0.181999134492587 z_k1_s0047
                                 - 0.460998374013609 z_k1_s0050
                                 - 0.254999842460635 z_k1_s0051
                                 - 0.382000597354374 z_k1_s0052
                                 - 0.33099930070348 z_k1_s0053
                                 - 0.559998775102603 z_k1_s0054
                                 - 0.347999732920444 z_k1_s0055
                                 - 0.222999793869168 z_k1_s0056
                                 - 0.206999907923306 z_k1_s0057
                                 - 0.261999734991362 z_k1_s0060
                                 - 0.327000076635263 z_k1_s0061
                                 - 1.29300003127714 z_k1_s0062
                                 - 1.22199743458841 z_k1_s0063
                                 - 0.280999811285741 z_k1_s0064
                                 - 0.603999576190709 z_k1_s0065
                                 - 0.426999698695419 z_k1_s0066
                                 - 0.690997291402715 z_k1_s0067
                                 - 0.261999612016871 z_k1_s0070
                                 - 0.224000287022204 z_k1_s0071
                                 - 0.261999612016871 z_k1_s0072
                                 - 0.272000777384207 z_k1_s0073
                                 - 0.224999953258831 z_k1_s0074
                                 - 0.0679999691959983 z_k1_s0075
                                 - 0.124000040951549 z_k1_s0076
                                 - 0.0639999533491707 z_k1_s0077
                                 - 0.0209999708470056 z_k1_s0100
                                 - 0.0809997355100674 z_k1_s0101
                                 - 0.0359999395377369 z_k1_s0102
                                 - 0.0269999730032799 z_k1_s0103
                                 - 0.0450000161720573 z_k1_s0104
                                 - 0.0589999478034551 z_k1_s0105
                                 - 0.0299999740814171 z_k1_s0106
                                 - 0.0679998776379576 z_k1_s0107
                                 - 0.0499998810042459 z_k1_s0110
                                 - 0.0989998633882703 z_k1_s0111
                                 - 0.0289999399824502 z_k1_s0112
                                 - 0.0449999379037592 z_k1_s0113
                                 - 0.0589998685849978 z_k1_s0114
                                 - 0.0700001034055717 z_k1_s0115
                                 - 0.0559997227249542 z_k1_s0116
                                 - 0.0419999270435293 z_k1_s0117
                                 - 0.0469999529915895 z_k1_s0120
                                 - 0.0399999742375143 z_k1_s0121
                                 - 0.0459999780981368 z_k1_s0122
                                 - 0.0650001705733319 z_k1_s0123
                                 - 0.0349999967703133 z_k1_s0124
                                 - 0.148000095228687 z_k1_s0125
                                 - 0.0389999993440616 z_k1_s0126
                                 - 0.0949999323766004 z_k1_s0127
                                 - 0.204999580882455 z_k1_s0130
                                 - 0.159999734249624 z_k1_s0131
                                 - 0.176000210873788 z_k1_s0132
                                 - 0.206999797710227 z_k1_s0133
                                 - 0.051999976531028 z_k1_s0134
                                 - 0.17899927811744 z_k1_s0135
                                 - 0.051999976531028 z_k1_s0136
                                 - 0.226999449991928 z_k1_s0137
                                 - 0.089999738650747 z_k1_s0140
                                 - 0.028999966787427 z_k1_s0141
                                 - 0.059999978767057 z_k1_s0142
                                 - 0.0709999060243992 z_k1_s0143
                                 - 0.0520000733980514 z_k1_s0144
                                 - 0.0299999893835285 z_k1_s0145
                                 - 0.0299999893835285 z_k1_s0146
                                 - 0.0969999427234249 z_k1_s0147
                                 - 0.130000134848687 z_k1_s0150
                                 - 0.10999992025675 z_k1_s0151
                                 - 0.0379999541250751 z_k1_s0152
                                 - 0.0559998952580377 z_k1_s0153
                                 - 0.0319999905474062 z_k1_s0154
                                 - 0.0559998952580377 z_k1_s0155
                                 - 0.0269999620994 z_k1_s0156
                                 - 0.054999960128375 z_k1_s0157
                                 - 0.112000026457309 z_k1_s0160
                                 - 0.0610000276419096 z_k1_s0161
                                 - 0.0369999584582843 z_k1_s0162
                                 - 0.0599996789597141 z_k1_s0163
                                 - 0.031999944045029 z_k1_s0164
                                 - 0.0259999761490576 z_k1_s0165
                                 - 0.0910001141214413 z_k1_s0166
                                 - 0.0749999691991546 z_k1_s0167
                                 - 0.0849998248093871 z_k1_s0170
                                 - 0.237998884027811 z_k1_s0171
                                 - 0.13899979169163 z_k1_s0172
                                 - 0.161998988548957 z_k1_s0173
                                 - 0.173999589254637 z_k1_s0174
                                 - 0.191999187316339 z_k1_s0175
                                 - 0.120999672431201 z_k1_s0176
                                 - 0.191999187316339 z_k1_s0177
                                 - 0.511999298287148 z_k1_s1000
                                 - 0.91799923578116 z_k1_s1001
                                 - 1.16900074858973 z_k1_s1002
                                 - 0.445999077749122 z_k1_s1003
                                 - 0.592997440223956 z_k1_s1004
                                 - 1.12499630863866 z_k1_s1005
                                 - 0.527998837309401 z_k1_s1006
                                 - 0.562999255830546 z_k1_s1007
                                 - 0.271999354348529 z_k1_s1010
                                 - 0.351999209168588 z_k1_s1011
                                 - 0.699998254551541 z_k1_s1012
                                 - 0.258999582243579 z_k1_s1013
                                 - 0.623997423219572 z_k1_s1014
                                 - 0.233999318781726 z_k1_s1015
                                 - 1.07299718329717 z_k1_s1016
                                 - 0.286999208346295 z_k1_s1017
                                 - 0.175000058354244 z_k1_s1020
                                 - 0.51600036417138 z_k1_s1021
                                 - 0.273998966959154 z_k1_s1022
                                 - 0.610000641303387 z_k1_s1023
                                 - 0.192999927336958 z_k1_s1024
                                 - 0.137999901934196 z_k1_s1025
                                 - 0.412999534547982 z_k1_s1026
                                 - 0.152999998753136 z_k1_s1027
                                 - 0.787999491718711 z_k1_s1030
                                 - 0.244000091935741 z_k1_s1031
                                 - 0.5839994498626 z_k1_s1032
                                 - 0.595001545746976 z_k1_s1033
                                 - 0.794999660645145 z_k1_s1034
                                 - 0.233999972640834 z_k1_s1035
                                 - 0.671999132937789 z_k1_s1036
                                 - 0.359000182527175 z_k1_s1037
                                 - 0.498000362469757 z_k1_s1040
                                 - 0.245999824650595 z_k1_s1041
                                 - 0.282999665229522 z_k1_s1042
                                 - 0.557999208166512 z_k1_s1043
                                 - 0.67599997623273 z_k1_s1044
                                 - 1.07299827688498 z_k1_s1045
                                 - 0.357999465248802 z_k1_s1046
                                 - 0.450998090378217 z_k1_s1047
                                 - 0.289999602343255 z_k1_s1050
                                 - 0.161999934970989 z_k1_s1051
                                 - 0.309998913401706 z_k1_s1052
                                 - 0.239998926698199 z_k1_s1053
                                 - 0.535000754804994 z_k1_s1054
                                 - 0.451998817914566 z_k1_s1055
                                 - 0.129999838278003 z_k1_s1056
                                 - 0.278999981261107 z_k1_s1057
                                 - 1.35100200426169 z_k1_s1060
                                 - 0.62200065725137 z_k1_s1061
                                 - 0.914001113041471 z_k1_s1062
                                 - 1.01900221787054 z_k1_s1063
                                 - 0.571000556734597 z_k1_s1064
                                 - 0.816998347931901 z_k1_s1065
                                 - 1.23800255971312 z_k1_s1066
                                 - 1.10000067153757 z_k1_s1067
                                 - 0.915000092373388 z_k1_s1070
                                 - 0.198999776701965 z_k1_s1071
                                 - 0.714997490794764 z_k1_s1072
                                 - 0.507000183941026 z_k1_s1073
                                 - 0.899995960638719 z_k1_s1074
                                 - 0.370999314665566 z_k1_s1075
                                 - 0.445998828085792 z_k1_s1076
                                 - 0.445998828085792 z_k1_s1077
                                 - 0.881000411847521 z_k1_s1100
                                 - 0.291000609395269 z_k1_s1101
                                 - 0.84500347561105 z_k1_s1102
                                 - 0.877002223269582 z_k1_s1103
                                 - 0.699003397341174 z_k1_s1104
                                 - 0.316000761837739 z_k1_s1105
                                 - 0.290000620949617 z_k1_s1106
                                 - 0.214000616477722 z_k1_s1107
                                 - 0.344000168385398 z_k1_s1110
                                 - 0.97999823652772 z_k1_s1111
                                 - 0.294999862858933 z_k1_s1112
                                 - 0.557998995900477 z_k1_s1113
                                 - 0.437999211835858 z_k1_s1114
                                 - 0.414999253223473 z_k1_s1115
                                 - 0.300000247561705 z_k1_s1116
                                 - 0.606998907726863 z_k1_s1117
                                 - 1.09999863980058 z_k1_s1120
                                 - 0.834998664349015 z_k1_s1121
                                 - 1.26500235526557 z_k1_s1122
                                 - 1.80899314797804 z_k1_s1123
                                 - 2.19799424220547 z_k1_s1124
                                 - 2.18600175779789 z_k1_s1125
                                 - 0.47199961631417 z_k1_s1126
                                 - 1.8350022141615 z_k1_s1127
                                 - 0.572997668062425 z_k1_s1130
                                 - 0.690999214735513 z_k1_s1131
                                 - 0.731001132656571 z_k1_s1132
                                 - 1.01099914112636 z_k1_s1133
                                 - 0.484999101527891 z_k1_s1134
                                 - 0.763998603747412 z_k1_s1135
                                 - 0.327999282176548 z_k1_s1136
                                 - 0.555997880744483 z_k1_s1137
                                 - 0.590999679300544 z_k1_s1140
                                 - 0.462001283549506 z_k1_s1141
                                 - 0.267000087577379 z_k1_s1142
                                 - 0.158000060972381 z_k1_s1143
                                 - 0.638998940370712 z_k1_s1144
                                 - 0.3679994232041 z_k1_s1145
                                 - 0.229999868477539 z_k1_s1146
                                 - 0.338000648285167 z_k1_s1147
                                 - 0.201000175454609 z_k1_s1150
                                 - 0.397999941243387 z_k1_s1151
                                 - 0.627999154524616 z_k1_s1152
                                 - 0.684001893845751 z_k1_s1153
                                 - 0.245000187778002 z_k1_s1154
                                 - 0.364000204084447 z_k1_s1155
                                 - 0.239999862945744 z_k1_s1156
                                 - 0.98499882894389 z_k1_s1157
                                 - 0.704000871615613 z_k1_s1160
                                 - 1.03399985581136 z_k1_s1161
                                 - 0.291999874645431 z_k1_s1162
                                 - 0.361999769668699 z_k1_s1163
                                 - 0.525997414021737 z_k1_s1164
                                 - 0.914999669632044 z_k1_s1165
                                 - 0.264999667702332 z_k1_s1166
                                 - 0.459997617182587 z_k1_s1167
                                 - 0.890001594060027 z_k1_s1170
                                 - 0.348000236875091 z_k1_s1171
                                 - 0.571001310274515 z_k1_s1172
                                 - 0.436998796680824 z_k1_s1173
                                 - 0.648002068257341 z_k1_s1174
                                 - 0.316999983648248 z_k1_s1175
                                 - 0.329999935040124 z_k1_s1176
                                 - 0.458001548222024 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.851106 w_k1_ADM_s000
                                 + 0.798567 w_k1_AXP_s000
                                 + 0.667476 w_k1_DIS_s000
                                 + 0.786344 w_k1_FDX_s000
                                 + 0.832129 w_k1_IBM_s000
                                 + 0.78136 w_k1_INTC_s000
                                 + 0.696129 w_k1_KO_s000
                                 + 0.629417 w_k1_MAT_s000
                                 + 0.745367 w_k1_MSFT_s000
                                 + 0.941316 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.09583 w_k1_ADM_s000
                                 + 1.33193 w_k1_AXP_s000
                                 + 0.911684 w_k1_DIS_s000
                                 + 1.04852 w_k1_FDX_s000
                                 + 0.869149 w_k1_IBM_s000
                                 + 1.17169 w_k1_INTC_s000
                                 + 0.913691 w_k1_KO_s000
                                 + 1.07978 w_k1_MAT_s000
                                 + 0.904397 w_k1_MSFT_s000
                                 + 1.21319 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.915696 w_k1_ADM_s000
                                 + 0.917703 w_k1_AXP_s000
                                 + 0.890286 w_k1_DIS_s000
                                 + 0.919989 w_k1_FDX_s000
                                 + 0.701525 w_k1_IBM_s000
                                 + 1.10284 w_k1_INTC_s000
                                 + 0.781593 w_k1_KO_s000
                                 + 0.649697 w_k1_MAT_s000
                                 + 0.962989 w_k1_MSFT_s000
                                 + 0.967442 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.23848 w_k1_ADM_s000
                                 + 1.20834 w_k1_AXP_s000
                                 + 1.57295 w_k1_DIS_s000
                                 + 1.25407 w_k1_FDX_s000
                                 + 1.13986 w_k1_IBM_s000
                                 + 1.25467 w_k1_INTC_s000
                                 + 1.37312 w_k1_KO_s000 + 1.04419 w_k1_MAT_s000
                                 + 1.32181 w_k1_MSFT_s000
                                 + 1.29883 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.815574 w_k1_ADM_s000
                                 + 1.10028 w_k1_AXP_s000
                                 + 1.02533 w_k1_DIS_s000
                                 + 0.942345 w_k1_FDX_s000
                                 + 1.2961 w_k1_IBM_s000
                                 + 0.850013 w_k1_INTC_s000
                                 + 1.11693 w_k1_KO_s000 + 1.01079 w_k1_MAT_s000
                                 + 0.905714 w_k1_MSFT_s000
                                 + 0.962416 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.58622 w_k1_ADM_s000
                                 + 1.11445 w_k1_AXP_s000
                                 + 1.12855 w_k1_DIS_s000 + 1.2021 w_k1_FDX_s000
                                 + 0.86277 w_k1_IBM_s000
                                 + 1.06631 w_k1_INTC_s000
                                 + 1.00516 w_k1_KO_s000
                                 + 0.728763 w_k1_MAT_s000
                                 + 1.02453 w_k1_MSFT_s000
                                 + 1.08574 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.62741 w_k1_ADM_s000
                                 + 1.76464 w_k1_AXP_s000
                                 + 1.84265 w_k1_DIS_s000
                                 + 1.28562 w_k1_FDX_s000
                                 + 1.63782 w_k1_IBM_s000
                                 + 1.26178 w_k1_INTC_s000
                                 + 1.46432 w_k1_KO_s000 + 2.44728 w_k1_MAT_s000
                                 + 1.27998 w_k1_MSFT_s000
                                 + 1.45269 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.05351 w_k1_ADM_s000
                                 + 1.40674 w_k1_AXP_s000
                                 + 1.13711 w_k1_DIS_s000
                                 + 1.03893 w_k1_FDX_s000
                                 + 1.10085 w_k1_IBM_s000
                                 + 1.16536 w_k1_INTC_s000
                                 + 1.22851 w_k1_KO_s000
                                 + 0.848635 w_k1_MAT_s000
                                 + 0.946661 w_k1_MSFT_s000
                                 + 1.11546 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.33126 w_k1_ADM_s001
                                 + 0.861797 w_k1_AXP_s001
                                 + 1.21504 w_k1_DIS_s001
                                 + 1.04749 w_k1_FDX_s001
                                 + 0.997041 w_k1_IBM_s001
                                 + 1.19432 w_k1_INTC_s001
                                 + 1.17189 w_k1_KO_s001
                                 + 0.708039 w_k1_MAT_s001
                                 + 1.14427 w_k1_MSFT_s001
                                 + 1.08126 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.10983 w_k1_ADM_s001
                                 + 1.0833 w_k1_AXP_s001 + 1.03401 w_k1_DIS_s001
                                 + 1.03485 w_k1_FDX_s001
                                 + 0.783089 w_k1_IBM_s001
                                 + 1.01642 w_k1_INTC_s001
                                 + 0.831301 w_k1_KO_s001
                                 + 0.678123 w_k1_MAT_s001
                                 + 0.90595 w_k1_MSFT_s001
                                 + 0.96682 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.881103 w_k1_ADM_s001
                                 + 0.734978 w_k1_AXP_s001
                                 + 0.677397 w_k1_DIS_s001
                                 + 0.692783 w_k1_FDX_s001
                                 + 0.981635 w_k1_IBM_s001
                                 + 0.748718 w_k1_INTC_s001
                                 + 0.894981 w_k1_KO_s001
                                 + 0.921939 w_k1_MAT_s001
                                 + 0.716393 w_k1_MSFT_s001
                                 + 0.767314 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.934405 w_k1_ADM_s001
                                 + 0.563685 w_k1_AXP_s001
                                 + 0.609454 w_k1_DIS_s001
                                 + 0.73942 w_k1_FDX_s001
                                 + 0.634279 w_k1_IBM_s001
                                 + 0.714665 w_k1_INTC_s001
                                 + 0.733143 w_k1_KO_s001
                                 + 0.455769 w_k1_MAT_s001
                                 + 0.753429 w_k1_MSFT_s001
                                 + 0.790618 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.2393 w_k1_ADM_s001
                                 + 0.932709 w_k1_AXP_s001
                                 + 1.09212 w_k1_DIS_s001
                                 + 0.937889 w_k1_FDX_s001
                                 + 1.36792 w_k1_IBM_s001
                                 + 0.958928 w_k1_INTC_s001
                                 + 1.23377 w_k1_KO_s001 + 1.2294 w_k1_MAT_s001
                                 + 0.97701 w_k1_MSFT_s001
                                 + 1.06217 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.958867 w_k1_ADM_s001
                                 + 0.677673 w_k1_AXP_s001
                                 + 0.842188 w_k1_DIS_s001
                                 + 0.859809 w_k1_FDX_s001
                                 + 0.739239 w_k1_IBM_s001
                                 + 0.884561 w_k1_INTC_s001
                                 + 0.851524 w_k1_KO_s001
                                 + 0.548534 w_k1_MAT_s001
                                 + 0.871565 w_k1_MSFT_s001
                                 + 0.942402 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.21601 w_k1_ADM_s001
                                 + 1.2149 w_k1_AXP_s001 + 1.42048 w_k1_DIS_s001
                                 + 1.27626 w_k1_FDX_s001
                                 + 1.17977 w_k1_IBM_s001
                                 + 1.1139 w_k1_INTC_s001 + 1.30371 w_k1_KO_s001
                                 + 1.48574 w_k1_MAT_s001
                                 + 1.17399 w_k1_MSFT_s001
                                 + 1.21962 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.07753 w_k1_ADM_s001
                                 + 1.0516 w_k1_AXP_s001 + 0.79088 w_k1_DIS_s001
                                 + 0.873685 w_k1_FDX_s001
                                 + 1.02151 w_k1_IBM_s001
                                 + 0.822005 w_k1_INTC_s001
                                 + 1.10103 w_k1_KO_s001 + 1.31521 w_k1_MAT_s001
                                 + 0.904849 w_k1_MSFT_s001
                                 + 0.997034 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.1846 w_k1_ADM_s002
                                 + 1.28746 w_k1_AXP_s002
                                 + 1.65873 w_k1_DIS_s002
                                 + 1.29696 w_k1_FDX_s002
                                 + 1.10308 w_k1_IBM_s002
                                 + 1.31953 w_k1_INTC_s002
                                 + 1.29782 w_k1_KO_s002
                                 + 0.998342 w_k1_MAT_s002
                                 + 1.09303 w_k1_MSFT_s002 + 1.1693 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.38044 w_k1_ADM_s002
                                 + 1.22287 w_k1_AXP_s002 + 1.3663 w_k1_DIS_s002
                                 + 1.14906 w_k1_FDX_s002
                                 + 1.32528 w_k1_IBM_s002
                                 + 1.04076 w_k1_INTC_s002
                                 + 1.20772 w_k1_KO_s002 + 1.44765 w_k1_MAT_s002
                                 + 1.03579 w_k1_MSFT_s002
                                 + 1.16187 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.704379 w_k1_ADM_s002
                                 + 0.68991 w_k1_AXP_s002
                                 + 0.669763 w_k1_DIS_s002
                                 + 0.738467 w_k1_FDX_s002
                                 + 0.73567 w_k1_IBM_s002
                                 + 0.889392 w_k1_INTC_s002
                                 + 0.693533 w_k1_KO_s002
                                 + 0.672824 w_k1_MAT_s002
                                 + 0.886276 w_k1_MSFT_s002
                                 + 0.895883 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.01293 w_k1_ADM_s002
                                 + 1.06226 w_k1_AXP_s002
                                 + 1.07468 w_k1_DIS_s002
                                 + 1.21774 w_k1_FDX_s002
                                 + 0.801033 w_k1_IBM_s002
                                 + 1.20213 w_k1_INTC_s002
                                 + 1.01125 w_k1_KO_s002
                                 + 0.920568 w_k1_MAT_s002
                                 + 1.15454 w_k1_MSFT_s002
                                 + 1.02868 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.705632 w_k1_ADM_s002
                                 + 0.781864 w_k1_AXP_s002
                                 + 0.90249 w_k1_DIS_s002
                                 + 0.897543 w_k1_FDX_s002
                                 + 1.30289 w_k1_IBM_s002
                                 + 0.920282 w_k1_INTC_s002
                                 + 1.02698 w_k1_KO_s002 + 1.27848 w_k1_MAT_s002
                                 + 0.935299 w_k1_MSFT_s002
                                 + 0.997224 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.796444 w_k1_ADM_s002
                                 + 0.948699 w_k1_AXP_s002
                                 + 0.777676 w_k1_DIS_s002
                                 + 0.934634 w_k1_FDX_s002
                                 + 0.642307 w_k1_IBM_s002
                                 + 0.915817 w_k1_INTC_s002
                                 + 0.699689 w_k1_KO_s002
                                 + 0.592739 w_k1_MAT_s002
                                 + 0.853973 w_k1_MSFT_s002
                                 + 0.845958 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.06158 w_k1_ADM_s002
                                 + 0.882757 w_k1_AXP_s002
                                 + 0.905954 w_k1_DIS_s002
                                 + 1.09107 w_k1_FDX_s002
                                 + 0.917645 w_k1_IBM_s002
                                 + 0.971666 w_k1_INTC_s002
                                 + 0.861374 w_k1_KO_s002
                                 + 0.93164 w_k1_MAT_s002
                                 + 1.02859 w_k1_MSFT_s002
                                 + 1.20256 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.914817 w_k1_ADM_s002
                                 + 0.845967 w_k1_AXP_s002
                                 + 0.872117 w_k1_DIS_s002
                                 + 0.932194 w_k1_FDX_s002
                                 + 0.937873 w_k1_IBM_s002
                                 + 0.80255 w_k1_INTC_s002
                                 + 0.814082 w_k1_KO_s002
                                 + 0.826335 w_k1_MAT_s002
                                 + 0.795949 w_k1_MSFT_s002
                                 + 0.919037 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.966091 w_k1_ADM_s003
                                 + 1.00121 w_k1_AXP_s003
                                 + 1.11651 w_k1_DIS_s003
                                 + 1.04038 w_k1_FDX_s003
                                 + 1.50826 w_k1_IBM_s003
                                 + 1.14363 w_k1_INTC_s003
                                 + 1.32457 w_k1_KO_s003 + 1.43582 w_k1_MAT_s003
                                 + 1.03866 w_k1_MSFT_s003
                                 + 1.08714 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.818438 w_k1_ADM_s003
                                 + 1.04445 w_k1_AXP_s003
                                 + 0.929126 w_k1_DIS_s003
                                 + 0.967112 w_k1_FDX_s003
                                 + 1.11973 w_k1_IBM_s003
                                 + 1.0544 w_k1_INTC_s003 + 1.00321 w_k1_KO_s003
                                 + 1.75685 w_k1_MAT_s003
                                 + 1.02275 w_k1_MSFT_s003
                                 + 1.07799 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.21945 w_k1_ADM_s003
                                 + 1.32596 w_k1_AXP_s003 + 1.5751 w_k1_DIS_s003
                                 + 1.36231 w_k1_FDX_s003
                                 + 1.57272 w_k1_IBM_s003
                                 + 1.28127 w_k1_INTC_s003
                                 + 1.55122 w_k1_KO_s003 + 1.83071 w_k1_MAT_s003
                                 + 1.43849 w_k1_MSFT_s003
                                 + 1.30621 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.92346 w_k1_ADM_s003
                                 + 0.874262 w_k1_AXP_s003
                                 + 1.22179 w_k1_DIS_s003
                                 + 0.99425 w_k1_FDX_s003
                                 + 0.992202 w_k1_IBM_s003
                                 + 0.972152 w_k1_INTC_s003
                                 + 1.21627 w_k1_KO_s003 + 1.07832 w_k1_MAT_s003
                                 + 1.1925 w_k1_MSFT_s003 + 1.00277 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.08821 w_k1_ADM_s003
                                 + 1.71266 w_k1_AXP_s003
                                 + 1.32765 w_k1_DIS_s003
                                 + 1.09162 w_k1_FDX_s003
                                 + 1.94678 w_k1_IBM_s003
                                 + 1.26368 w_k1_INTC_s003 + 1.5315 w_k1_KO_s003
                                 + 1.98441 w_k1_MAT_s003
                                 + 1.17973 w_k1_MSFT_s003
                                 + 1.23438 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.870516 w_k1_ADM_s003
                                 + 0.735066 w_k1_AXP_s003
                                 + 0.899057 w_k1_DIS_s003
                                 + 0.983384 w_k1_FDX_s003
                                 + 0.807711 w_k1_IBM_s003
                                 + 0.927095 w_k1_INTC_s003
                                 + 0.92729 w_k1_KO_s003
                                 + 0.655055 w_k1_MAT_s003
                                 + 0.99038 w_k1_MSFT_s003
                                 + 0.91057 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.729451 w_k1_ADM_s003
                                 + 0.721291 w_k1_AXP_s003
                                 + 0.653193 w_k1_DIS_s003
                                 + 0.769642 w_k1_FDX_s003
                                 + 0.89396 w_k1_IBM_s003
                                 + 0.820084 w_k1_INTC_s003
                                 + 0.795506 w_k1_KO_s003
                                 + 0.846199 w_k1_MAT_s003
                                 + 0.832361 w_k1_MSFT_s003
                                 + 0.921495 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.24717 w_k1_ADM_s003
                                 + 1.31883 w_k1_AXP_s003
                                 + 1.28504 w_k1_DIS_s003
                                 + 1.33662 w_k1_FDX_s003
                                 + 0.924403 w_k1_IBM_s003
                                 + 1.33857 w_k1_INTC_s003
                                 + 1.07172 w_k1_KO_s003
                                 + 0.929263 w_k1_MAT_s003
                                 + 1.11081 w_k1_MSFT_s003
                                 + 1.09564 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.807016 w_k1_ADM_s004
                                 + 0.797343 w_k1_AXP_s004
                                 + 0.78638 w_k1_DIS_s004
                                 + 0.864208 w_k1_FDX_s004
                                 + 0.645522 w_k1_IBM_s004
                                 + 0.698222 w_k1_INTC_s004
                                 + 0.567726 w_k1_KO_s004
                                 + 0.612649 w_k1_MAT_s004
                                 + 0.66067 w_k1_MSFT_s004
                                 + 0.845896 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.942525 w_k1_ADM_s004
                                 + 1.10607 w_k1_AXP_s004
                                 + 0.990956 w_k1_DIS_s004
                                 + 1.01783 w_k1_FDX_s004
                                 + 1.27874 w_k1_IBM_s004
                                 + 0.901453 w_k1_INTC_s004
                                 + 1.14893 w_k1_KO_s004 + 1.42647 w_k1_MAT_s004
                                 + 1.00603 w_k1_MSFT_s004
                                 + 1.00992 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.934282 w_k1_ADM_s004
                                 + 0.918139 w_k1_AXP_s004
                                 + 0.765002 w_k1_DIS_s004
                                 + 0.984903 w_k1_FDX_s004
                                 + 0.830777 w_k1_IBM_s004
                                 + 0.761826 w_k1_INTC_s004
                                 + 0.718844 w_k1_KO_s004
                                 + 1.11266 w_k1_MAT_s004
                                 + 0.756324 w_k1_MSFT_s004
                                 + 0.911085 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.925062 w_k1_ADM_s004
                                 + 1.47894 w_k1_AXP_s004
                                 + 1.19994 w_k1_DIS_s004 + 1.1563 w_k1_FDX_s004
                                 + 1.78637 w_k1_IBM_s004
                                 + 1.03769 w_k1_INTC_s004
                                 + 1.23638 w_k1_KO_s004 + 2.17018 w_k1_MAT_s004
                                 + 0.850942 w_k1_MSFT_s004
                                 + 1.22555 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.922464 w_k1_ADM_s004
                                 + 0.736253 w_k1_AXP_s004
                                 + 0.968086 w_k1_DIS_s004
                                 + 0.933911 w_k1_FDX_s004
                                 + 1.2087 w_k1_IBM_s004
                                 + 0.800996 w_k1_INTC_s004
                                 + 0.995191 w_k1_KO_s004
                                 + 1.09937 w_k1_MAT_s004
                                 + 0.910681 w_k1_MSFT_s004
                                 + 0.928972 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.29823 w_k1_ADM_s004
                                 + 1.4243 w_k1_AXP_s004 + 1.20198 w_k1_DIS_s004
                                 + 1.32133 w_k1_FDX_s004
                                 + 0.933103 w_k1_IBM_s004
                                 + 0.898929 w_k1_INTC_s004
                                 + 0.844624 w_k1_KO_s004
                                 + 1.16534 w_k1_MAT_s004
                                 + 0.976819 w_k1_MSFT_s004
                                 + 1.08307 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.17466 w_k1_ADM_s004
                                 + 1.07125 w_k1_AXP_s004
                                 + 1.08704 w_k1_DIS_s004
                                 + 1.16142 w_k1_FDX_s004
                                 + 0.826704 w_k1_IBM_s004
                                 + 0.816092 w_k1_INTC_s004
                                 + 0.791465 w_k1_KO_s004
                                 + 0.789192 w_k1_MAT_s004
                                 + 0.893971 w_k1_MSFT_s004
                                 + 1.01591 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.88742 w_k1_ADM_s004
                                 + 1.24939 w_k1_AXP_s004
                                 + 1.63247 w_k1_DIS_s004
                                 + 1.27585 w_k1_FDX_s004
                                 + 1.26687 w_k1_IBM_s004
                                 + 1.06332 w_k1_INTC_s004
                                 + 1.37329 w_k1_KO_s004 + 1.64547 w_k1_MAT_s004
                                 + 1.04213 w_k1_MSFT_s004
                                 + 1.17123 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.0503 w_k1_ADM_s005
                                 + 0.970587 w_k1_AXP_s005
                                 + 1.11676 w_k1_DIS_s005
                                 + 1.14761 w_k1_FDX_s005
                                 + 0.904915 w_k1_IBM_s005
                                 + 1.11259 w_k1_INTC_s005
                                 + 0.878158 w_k1_KO_s005
                                 + 0.709673 w_k1_MAT_s005
                                 + 1.08168 w_k1_MSFT_s005
                                 + 0.892328 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.5933 w_k1_ADM_s005
                                 + 1.42671 w_k1_AXP_s005
                                 + 1.35477 w_k1_DIS_s005 + 1.3538 w_k1_FDX_s005
                                 + 0.828124 w_k1_IBM_s005
                                 + 1.02692 w_k1_INTC_s005
                                 + 0.997665 w_k1_KO_s005
                                 + 1.30657 w_k1_MAT_s005
                                 + 1.21061 w_k1_MSFT_s005 + 1.1499 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.713034 w_k1_ADM_s005
                                 + 0.818678 w_k1_AXP_s005
                                 + 0.858289 w_k1_DIS_s005
                                 + 0.831276 w_k1_FDX_s005
                                 + 1.08539 w_k1_IBM_s005
                                 + 1.01326 w_k1_INTC_s005
                                 + 0.930692 w_k1_KO_s005
                                 + 1.00632 w_k1_MAT_s005
                                 + 0.976401 w_k1_MSFT_s005
                                 + 0.851796 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.13589 w_k1_ADM_s005
                                 + 1.06771 w_k1_AXP_s005
                                 + 0.859006 w_k1_DIS_s005
                                 + 0.971336 w_k1_FDX_s005
                                 + 0.740639 w_k1_IBM_s005
                                 + 1.14052 w_k1_INTC_s005
                                 + 0.824591 w_k1_KO_s005
                                 + 0.897686 w_k1_MAT_s005
                                 + 1.12499 w_k1_MSFT_s005
                                 + 0.967098 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.01296 w_k1_ADM_s005
                                 + 1.07808 w_k1_AXP_s005
                                 + 1.18011 w_k1_DIS_s005
                                 + 1.04539 w_k1_FDX_s005 + 1.3832 w_k1_IBM_s005
                                 + 1.46194 w_k1_INTC_s005
                                 + 1.29072 w_k1_KO_s005 + 1.31861 w_k1_MAT_s005
                                 + 1.31583 w_k1_MSFT_s005
                                 + 1.12567 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.99151 w_k1_ADM_s005
                                 + 0.769041 w_k1_AXP_s005
                                 + 0.858268 w_k1_DIS_s005
                                 + 0.96638 w_k1_FDX_s005
                                 + 0.64583 w_k1_IBM_s005
                                 + 1.01437 w_k1_INTC_s005
                                 + 0.652384 w_k1_KO_s005
                                 + 0.588665 w_k1_MAT_s005
                                 + 1.06501 w_k1_MSFT_s005
                                 + 0.714548 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.833967 w_k1_ADM_s005
                                 + 0.766764 w_k1_AXP_s005
                                 + 0.559607 w_k1_DIS_s005
                                 + 0.802795 w_k1_FDX_s005
                                 + 0.626995 w_k1_IBM_s005
                                 + 1.01604 w_k1_INTC_s005
                                 + 0.621348 w_k1_KO_s005
                                 + 0.439525 w_k1_MAT_s005
                                 + 0.774505 w_k1_MSFT_s005
                                 + 0.767111 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.22049 w_k1_ADM_s005
                                 + 1.67601 w_k1_AXP_s005
                                 + 1.70542 w_k1_DIS_s005
                                 + 1.55384 w_k1_FDX_s005
                                 + 1.55006 w_k1_IBM_s005
                                 + 1.29646 w_k1_INTC_s005
                                 + 1.43545 w_k1_KO_s005 + 1.73081 w_k1_MAT_s005
                                 + 1.53589 w_k1_MSFT_s005
                                 + 1.27347 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.662288 w_k1_ADM_s006
                                 + 0.693978 w_k1_AXP_s006
                                 + 0.622705 w_k1_DIS_s006
                                 + 0.659695 w_k1_FDX_s006
                                 + 1.19005 w_k1_IBM_s006
                                 + 0.715848 w_k1_INTC_s006
                                 + 0.898642 w_k1_KO_s006
                                 + 1.43175 w_k1_MAT_s006
                                 + 0.735421 w_k1_MSFT_s006
                                 + 0.88729 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.39264 w_k1_ADM_s006
                                 + 1.36597 w_k1_AXP_s006
                                 + 1.22242 w_k1_DIS_s006
                                 + 1.11074 w_k1_FDX_s006
                                 + 1.24797 w_k1_IBM_s006
                                 + 1.06797 w_k1_INTC_s006
                                 + 1.29248 w_k1_KO_s006 + 2.00885 w_k1_MAT_s006
                                 + 1.14056 w_k1_MSFT_s006
                                 + 1.15442 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.753412 w_k1_ADM_s006
                                 + 0.68162 w_k1_AXP_s006
                                 + 0.652562 w_k1_DIS_s006
                                 + 0.760211 w_k1_FDX_s006
                                 + 0.697453 w_k1_IBM_s006
                                 + 0.831324 w_k1_INTC_s006
                                 + 0.753913 w_k1_KO_s006
                                 + 0.904815 w_k1_MAT_s006
                                 + 0.770741 w_k1_MSFT_s006
                                 + 0.814951 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.893498 w_k1_ADM_s006
                                 + 1.03647 w_k1_AXP_s006
                                 + 0.997244 w_k1_DIS_s006
                                 + 1.00085 w_k1_FDX_s006
                                 + 1.07284 w_k1_IBM_s006
                                 + 1.03503 w_k1_INTC_s006
                                 + 1.02975 w_k1_KO_s006 + 1.02776 w_k1_MAT_s006
                                 + 1.02932 w_k1_MSFT_s006
                                 + 0.870491 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.32193 w_k1_ADM_s006
                                 + 1.81573 w_k1_AXP_s006
                                 + 2.03972 w_k1_DIS_s006
                                 + 1.44433 w_k1_FDX_s006
                                 + 1.40865 w_k1_IBM_s006
                                 + 1.04948 w_k1_INTC_s006 + 1.5613 w_k1_KO_s006
                                 + 2.23297 w_k1_MAT_s006
                                 + 1.36604 w_k1_MSFT_s006
                                 + 1.21837 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.03586 w_k1_ADM_s006
                                 + 1.26835 w_k1_AXP_s006
                                 + 0.908345 w_k1_DIS_s006
                                 + 1.07682 w_k1_FDX_s006
                                 + 0.899372 w_k1_IBM_s006
                                 + 0.928988 w_k1_INTC_s006
                                 + 1.01302 w_k1_KO_s006 + 1.52895 w_k1_MAT_s006
                                 + 0.949484 w_k1_MSFT_s006
                                 + 0.99077 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.01421 w_k1_ADM_s006
                                 + 1.20981 w_k1_AXP_s006
                                 + 1.08231 w_k1_DIS_s006
                                 + 1.00026 w_k1_FDX_s006
                                 + 1.77773 w_k1_IBM_s006
                                 + 0.914092 w_k1_INTC_s006
                                 + 1.48059 w_k1_KO_s006 + 2.31687 w_k1_MAT_s006
                                 + 1.01485 w_k1_MSFT_s006
                                 + 1.07186 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.01459 w_k1_ADM_s006
                                 + 1.2637 w_k1_AXP_s006 + 1.4286 w_k1_DIS_s006
                                 + 1.31285 w_k1_FDX_s006
                                 + 1.00046 w_k1_IBM_s006
                                 + 1.12203 w_k1_INTC_s006
                                 + 1.06905 w_k1_KO_s006 + 1.38174 w_k1_MAT_s006
                                 + 1.24624 w_k1_MSFT_s006
                                 + 1.07477 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.867967 w_k1_ADM_s007
                                 + 1.10125 w_k1_AXP_s007
                                 + 1.26798 w_k1_DIS_s007
                                 + 0.959277 w_k1_FDX_s007
                                 + 1.4936 w_k1_IBM_s007
                                 + 1.00292 w_k1_INTC_s007
                                 + 1.24162 w_k1_KO_s007 + 1.4528 w_k1_MAT_s007
                                 + 1.14442 w_k1_MSFT_s007
                                 + 1.02212 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.75817 w_k1_ADM_s007
                                 + 0.692241 w_k1_AXP_s007
                                 + 0.709026 w_k1_DIS_s007
                                 + 0.779121 w_k1_FDX_s007
                                 + 0.893281 w_k1_IBM_s007
                                 + 0.958315 w_k1_INTC_s007
                                 + 0.748408 w_k1_KO_s007
                                 + 0.700805 w_k1_MAT_s007
                                 + 0.921826 w_k1_MSFT_s007
                                 + 0.798077 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.49408 w_k1_ADM_s007
                                 + 0.964399 w_k1_AXP_s007
                                 + 1.10767 w_k1_DIS_s007
                                 + 1.14098 w_k1_FDX_s007
                                 + 0.863715 w_k1_IBM_s007
                                 + 1.10095 w_k1_INTC_s007
                                 + 1.04941 w_k1_KO_s007
                                 + 0.780052 w_k1_MAT_s007
                                 + 1.33526 w_k1_MSFT_s007
                                 + 1.10273 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.13312 w_k1_ADM_s007
                                 + 1.08072 w_k1_AXP_s007
                                 + 1.04436 w_k1_DIS_s007
                                 + 0.926849 w_k1_FDX_s007
                                 + 1.06505 w_k1_IBM_s007
                                 + 0.993355 w_k1_INTC_s007
                                 + 0.998815 w_k1_KO_s007
                                 + 1.03924 w_k1_MAT_s007
                                 + 1.00203 w_k1_MSFT_s007
                                 + 1.01994 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.927992 w_k1_ADM_s007
                                 + 0.871128 w_k1_AXP_s007
                                 + 0.846468 w_k1_DIS_s007
                                 + 0.925674 w_k1_FDX_s007
                                 + 0.882929 w_k1_IBM_s007
                                 + 0.915171 w_k1_INTC_s007
                                 + 0.749841 w_k1_KO_s007
                                 + 0.548586 w_k1_MAT_s007
                                 + 0.923089 w_k1_MSFT_s007
                                 + 0.892191 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.03953 w_k1_ADM_s007
                                 + 0.890372 w_k1_AXP_s007
                                 + 0.690211 w_k1_DIS_s007
                                 + 0.93122 w_k1_FDX_s007
                                 + 0.680693 w_k1_IBM_s007
                                 + 0.857715 w_k1_INTC_s007
                                 + 0.711961 w_k1_KO_s007
                                 + 0.702023 w_k1_MAT_s007
                                 + 0.995753 w_k1_MSFT_s007
                                 + 0.782809 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.24827 w_k1_ADM_s007
                                 + 1.60442 w_k1_AXP_s007
                                 + 1.66271 w_k1_DIS_s007
                                 + 1.43806 w_k1_FDX_s007
                                 + 1.03474 w_k1_IBM_s007
                                 + 1.20474 w_k1_INTC_s007
                                 + 1.15099 w_k1_KO_s007 + 1.08166 w_k1_MAT_s007
                                 + 1.55506 w_k1_MSFT_s007
                                 + 1.09642 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.19962 w_k1_ADM_s007
                                 + 0.800969 w_k1_AXP_s007
                                 + 0.742345 w_k1_DIS_s007
                                 + 0.946993 w_k1_FDX_s007
                                 + 0.566425 w_k1_IBM_s007
                                 + 0.821661 w_k1_INTC_s007
                                 + 0.700466 w_k1_KO_s007
                                 + 0.461788 w_k1_MAT_s007
                                 + 0.884036 w_k1_MSFT_s007
                                 + 0.777478 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.729717 w_k1_ADM_s010
                                 + 0.915594 w_k1_AXP_s010
                                 + 0.773237 w_k1_DIS_s010
                                 + 0.896719 w_k1_FDX_s010
                                 + 0.702486 w_k1_IBM_s010
                                 + 0.702705 w_k1_INTC_s010
                                 + 0.726048 w_k1_KO_s010
                                 + 0.845976 w_k1_MAT_s010
                                 + 0.798688 w_k1_MSFT_s010
                                 + 0.78449 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.18457 w_k1_ADM_s010
                                 + 1.16056 w_k1_AXP_s010
                                 + 1.26244 w_k1_DIS_s010
                                 + 1.23526 w_k1_FDX_s010
                                 + 0.925597 w_k1_IBM_s010
                                 + 1.22834 w_k1_INTC_s010 + 1.0517 w_k1_KO_s010
                                 + 0.888906 w_k1_MAT_s010
                                 + 1.09471 w_k1_MSFT_s010
                                 + 0.85892 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.05839 w_k1_ADM_s010
                                 + 1.00921 w_k1_AXP_s010
                                 + 0.93041 w_k1_DIS_s010
                                 + 1.05148 w_k1_FDX_s010
                                 + 0.855664 w_k1_IBM_s010
                                 + 0.946561 w_k1_INTC_s010
                                 + 0.857773 w_k1_KO_s010
                                 + 0.690252 w_k1_MAT_s010
                                 + 0.757439 w_k1_MSFT_s010
                                 + 0.80888 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.969918 w_k1_ADM_s010
                                 + 0.79967 w_k1_AXP_s010
                                 + 0.750506 w_k1_DIS_s010
                                 + 0.834632 w_k1_FDX_s010
                                 + 1.03197 w_k1_IBM_s010
                                 + 0.775486 w_k1_INTC_s010
                                 + 0.848349 w_k1_KO_s010
                                 + 1.15698 w_k1_MAT_s010
                                 + 0.883589 w_k1_MSFT_s010
                                 + 0.833017 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.989691 w_k1_ADM_s010
                                 + 1.03121 w_k1_AXP_s010
                                 + 0.906043 w_k1_DIS_s010
                                 + 1.01524 w_k1_FDX_s010
                                 + 0.722228 w_k1_IBM_s010
                                 + 1.00841 w_k1_INTC_s010
                                 + 0.945404 w_k1_KO_s010
                                 + 0.586138 w_k1_MAT_s010
                                 + 1.08829 w_k1_MSFT_s010 + 0.9595 w_k1_PG_s010
                                 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.789101 w_k1_ADM_s010
                                 + 0.713296 w_k1_AXP_s010
                                 + 0.696426 w_k1_DIS_s010
                                 + 0.821777 w_k1_FDX_s010
                                 + 0.763197 w_k1_IBM_s010
                                 + 0.8393 w_k1_INTC_s010
                                 + 0.742725 w_k1_KO_s010
                                 + 0.799511 w_k1_MAT_s010
                                 + 0.799514 w_k1_MSFT_s010
                                 + 0.845261 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.824286 w_k1_ADM_s010
                                 + 0.673612 w_k1_AXP_s010
                                 + 1.00153 w_k1_DIS_s010
                                 + 0.856188 w_k1_FDX_s010
                                 + 1.18855 w_k1_IBM_s010
                                 + 0.883324 w_k1_INTC_s010
                                 + 1.0037 w_k1_KO_s010 + 1.0918 w_k1_MAT_s010
                                 + 0.753654 w_k1_MSFT_s010
                                 + 0.885925 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.05442 w_k1_ADM_s010
                                 + 0.940457 w_k1_AXP_s010
                                 + 1.34586 w_k1_DIS_s010
                                 + 1.10375 w_k1_FDX_s010
                                 + 1.26543 w_k1_IBM_s010
                                 + 1.24593 w_k1_INTC_s010
                                 + 1.30352 w_k1_KO_s010
                                 + 0.906484 w_k1_MAT_s010
                                 + 1.16654 w_k1_MSFT_s010
                                 + 0.947008 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.23803 w_k1_ADM_s011
                                 + 1.59544 w_k1_AXP_s011
                                 + 1.83394 w_k1_DIS_s011
                                 + 1.22512 w_k1_FDX_s011
                                 + 1.59774 w_k1_IBM_s011
                                 + 1.41881 w_k1_INTC_s011
                                 + 1.36224 w_k1_KO_s011 + 1.51881 w_k1_MAT_s011
                                 + 1.23262 w_k1_MSFT_s011
                                 + 1.19996 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.0075 w_k1_ADM_s011
                                 + 0.840097 w_k1_AXP_s011
                                 + 0.905029 w_k1_DIS_s011
                                 + 0.89236 w_k1_FDX_s011
                                 + 1.19529 w_k1_IBM_s011
                                 + 1.12738 w_k1_INTC_s011
                                 + 1.21687 w_k1_KO_s011 + 1.23662 w_k1_MAT_s011
                                 + 1.08265 w_k1_MSFT_s011
                                 + 0.894475 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.01249 w_k1_ADM_s011
                                 + 1.22862 w_k1_AXP_s011
                                 + 0.937365 w_k1_DIS_s011
                                 + 0.974945 w_k1_FDX_s011
                                 + 1.6102 w_k1_IBM_s011 + 1.2031 w_k1_INTC_s011
                                 + 1.10027 w_k1_KO_s011 + 1.64853 w_k1_MAT_s011
                                 + 0.978297 w_k1_MSFT_s011
                                 + 1.0566 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.11843 w_k1_ADM_s011
                                 + 0.819354 w_k1_AXP_s011
                                 + 0.806567 w_k1_DIS_s011
                                 + 0.981293 w_k1_FDX_s011
                                 + 0.642409 w_k1_IBM_s011
                                 + 1.0398 w_k1_INTC_s011
                                 + 0.759705 w_k1_KO_s011 + 0.6105 w_k1_MAT_s011
                                 + 1.05276 w_k1_MSFT_s011
                                 + 0.833976 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.40277 w_k1_ADM_s011
                                 + 1.10459 w_k1_AXP_s011
                                 + 1.23857 w_k1_DIS_s011
                                 + 1.14443 w_k1_FDX_s011
                                 + 1.20857 w_k1_IBM_s011
                                 + 1.16441 w_k1_INTC_s011
                                 + 1.12645 w_k1_KO_s011 + 1.09928 w_k1_MAT_s011
                                 + 1.14105 w_k1_MSFT_s011 + 1.1035 w_k1_PG_s011
                                 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.18505 w_k1_ADM_s011
                                 + 1.26802 w_k1_AXP_s011
                                 + 1.06456 w_k1_DIS_s011 + 1.1298 w_k1_FDX_s011
                                 + 0.902661 w_k1_IBM_s011
                                 + 1.06281 w_k1_INTC_s011
                                 + 1.03254 w_k1_KO_s011 + 1.09958 w_k1_MAT_s011
                                 + 1.10711 w_k1_MSFT_s011 + 1.1588 w_k1_PG_s011
                                 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.940551 w_k1_ADM_s011
                                 + 0.755097 w_k1_AXP_s011
                                 + 0.666226 w_k1_DIS_s011
                                 + 0.797251 w_k1_FDX_s011
                                 + 0.599052 w_k1_IBM_s011
                                 + 0.959408 w_k1_INTC_s011
                                 + 0.686537 w_k1_KO_s011
                                 + 0.309049 w_k1_MAT_s011
                                 + 0.952921 w_k1_MSFT_s011
                                 + 0.705599 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.81929 w_k1_ADM_s011
                                 + 0.895493 w_k1_AXP_s011
                                 + 0.772516 w_k1_DIS_s011
                                 + 0.798638 w_k1_FDX_s011
                                 + 1.01733 w_k1_IBM_s011
                                 + 0.96171 w_k1_INTC_s011
                                 + 0.907748 w_k1_KO_s011
                                 + 0.834205 w_k1_MAT_s011
                                 + 0.900348 w_k1_MSFT_s011
                                 + 0.849261 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.37873 w_k1_ADM_s012
                                 + 0.956302 w_k1_AXP_s012
                                 + 1.19176 w_k1_DIS_s012
                                 + 1.04972 w_k1_FDX_s012
                                 + 1.27566 w_k1_IBM_s012
                                 + 0.964005 w_k1_INTC_s012
                                 + 1.26872 w_k1_KO_s012 + 1.51681 w_k1_MAT_s012
                                 + 1.38728 w_k1_MSFT_s012
                                 + 1.10132 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.0255 w_k1_ADM_s012
                                 + 0.87537 w_k1_AXP_s012
                                 + 1.12762 w_k1_DIS_s012
                                 + 1.16978 w_k1_FDX_s012
                                 + 0.743713 w_k1_IBM_s012
                                 + 1.10556 w_k1_INTC_s012
                                 + 0.929025 w_k1_KO_s012
                                 + 0.722332 w_k1_MAT_s012
                                 + 1.24779 w_k1_MSFT_s012
                                 + 1.09368 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.98506 w_k1_ADM_s012
                                 + 1.24494 w_k1_AXP_s012
                                 + 1.70457 w_k1_DIS_s012
                                 + 1.32244 w_k1_FDX_s012
                                 + 1.32967 w_k1_IBM_s012
                                 + 1.69366 w_k1_INTC_s012
                                 + 1.30214 w_k1_KO_s012 + 1.57921 w_k1_MAT_s012
                                 + 1.57436 w_k1_MSFT_s012
                                 + 1.19172 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.02022 w_k1_ADM_s012
                                 + 0.811233 w_k1_AXP_s012
                                 + 0.899548 w_k1_DIS_s012
                                 + 0.884017 w_k1_FDX_s012
                                 + 1.21929 w_k1_IBM_s012
                                 + 0.72043 w_k1_INTC_s012 + 0.9547 w_k1_KO_s012
                                 + 1.47273 w_k1_MAT_s012
                                 + 0.874915 w_k1_MSFT_s012
                                 + 1.01726 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.936636 w_k1_ADM_s012
                                 + 0.634568 w_k1_AXP_s012
                                 + 1.03529 w_k1_DIS_s012
                                 + 1.03741 w_k1_FDX_s012
                                 + 1.04019 w_k1_IBM_s012
                                 + 0.958291 w_k1_INTC_s012
                                 + 1.00593 w_k1_KO_s012
                                 + 0.927016 w_k1_MAT_s012
                                 + 1.15364 w_k1_MSFT_s012
                                 + 1.03817 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.855119 w_k1_ADM_s012
                                 + 0.671888 w_k1_AXP_s012
                                 + 0.807699 w_k1_DIS_s012
                                 + 0.828107 w_k1_FDX_s012
                                 + 0.791312 w_k1_IBM_s012
                                 + 0.890227 w_k1_INTC_s012
                                 + 0.842026 w_k1_KO_s012
                                 + 0.705602 w_k1_MAT_s012
                                 + 0.911875 w_k1_MSFT_s012
                                 + 0.974592 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.16344 w_k1_ADM_s012
                                 + 1.02278 w_k1_AXP_s012
                                 + 1.15188 w_k1_DIS_s012
                                 + 1.08654 w_k1_FDX_s012
                                 + 1.64896 w_k1_IBM_s012
                                 + 1.43773 w_k1_INTC_s012
                                 + 1.24507 w_k1_KO_s012 + 1.44138 w_k1_MAT_s012
                                 + 1.16326 w_k1_MSFT_s012
                                 + 1.16652 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.00763 w_k1_ADM_s012
                                 + 0.940087 w_k1_AXP_s012
                                 + 1.15774 w_k1_DIS_s012 + 1.156 w_k1_FDX_s012
                                 + 0.92258 w_k1_IBM_s012
                                 + 1.26852 w_k1_INTC_s012
                                 + 1.10477 w_k1_KO_s012
                                 + 0.986829 w_k1_MAT_s012
                                 + 1.39654 w_k1_MSFT_s012
                                 + 1.11386 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.973159 w_k1_ADM_s013
                                 + 1.0442 w_k1_AXP_s013
                                 + 0.973232 w_k1_DIS_s013
                                 + 0.970262 w_k1_FDX_s013
                                 + 1.17907 w_k1_IBM_s013
                                 + 0.907249 w_k1_INTC_s013
                                 + 0.917619 w_k1_KO_s013 + 1.357 w_k1_MAT_s013
                                 + 0.994119 w_k1_MSFT_s013
                                 + 0.962893 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.22792 w_k1_ADM_s013
                                 + 0.804352 w_k1_AXP_s013
                                 + 1.01453 w_k1_DIS_s013
                                 + 0.861341 w_k1_FDX_s013
                                 + 1.19133 w_k1_IBM_s013
                                 + 0.910766 w_k1_INTC_s013
                                 + 0.962674 w_k1_KO_s013
                                 + 0.765678 w_k1_MAT_s013
                                 + 0.953835 w_k1_MSFT_s013
                                 + 0.932928 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.748395 w_k1_ADM_s013
                                 + 0.766705 w_k1_AXP_s013
                                 + 0.643857 w_k1_DIS_s013
                                 + 0.778958 w_k1_FDX_s013
                                 + 0.784983 w_k1_IBM_s013
                                 + 0.817885 w_k1_INTC_s013
                                 + 0.623051 w_k1_KO_s013
                                 + 0.74299 w_k1_MAT_s013
                                 + 0.713726 w_k1_MSFT_s013
                                 + 0.741131 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.24805 w_k1_ADM_s013
                                 + 1.24307 w_k1_AXP_s013 + 1.0736 w_k1_DIS_s013
                                 + 1.05907 w_k1_FDX_s013
                                 + 1.04435 w_k1_IBM_s013
                                 + 1.14219 w_k1_INTC_s013
                                 + 1.11006 w_k1_KO_s013 + 1.09794 w_k1_MAT_s013
                                 + 1.0201 w_k1_MSFT_s013 + 1.00286 w_k1_PG_s013
                                 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.84669 w_k1_ADM_s013
                                 + 1.69767 w_k1_AXP_s013
                                 + 2.02158 w_k1_DIS_s013
                                 + 1.52097 w_k1_FDX_s013
                                 + 1.12314 w_k1_IBM_s013
                                 + 1.03388 w_k1_INTC_s013 + 1.4506 w_k1_KO_s013
                                 + 1.57302 w_k1_MAT_s013
                                 + 1.39718 w_k1_MSFT_s013
                                 + 1.29027 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.11397 w_k1_ADM_s013
                                 + 1.40328 w_k1_AXP_s013
                                 + 1.53437 w_k1_DIS_s013
                                 + 1.29684 w_k1_FDX_s013 + 1.1509 w_k1_IBM_s013
                                 + 1.23118 w_k1_INTC_s013
                                 + 1.18077 w_k1_KO_s013 + 1.01232 w_k1_MAT_s013
                                 + 1.2178 w_k1_MSFT_s013 + 1.12617 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.65513 w_k1_ADM_s013
                                 + 1.11872 w_k1_AXP_s013
                                 + 1.23326 w_k1_DIS_s013
                                 + 1.33661 w_k1_FDX_s013
                                 + 0.794402 w_k1_IBM_s013
                                 + 1.00731 w_k1_INTC_s013
                                 + 0.909085 w_k1_KO_s013
                                 + 0.719082 w_k1_MAT_s013
                                 + 1.13955 w_k1_MSFT_s013
                                 + 0.958792 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.27579 w_k1_ADM_s013
                                 + 0.926164 w_k1_AXP_s013
                                 + 0.866311 w_k1_DIS_s013
                                 + 1.00296 w_k1_FDX_s013
                                 + 0.70528 w_k1_IBM_s013
                                 + 0.836257 w_k1_INTC_s013
                                 + 0.71879 w_k1_KO_s013
                                 + 0.620756 w_k1_MAT_s013
                                 + 0.829449 w_k1_MSFT_s013
                                 + 0.925934 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.04783 w_k1_ADM_s014
                                 + 1.05684 w_k1_AXP_s014
                                 + 1.19968 w_k1_DIS_s014
                                 + 1.07896 w_k1_FDX_s014
                                 + 1.16545 w_k1_IBM_s014
                                 + 1.2582 w_k1_INTC_s014 + 1.3472 w_k1_KO_s014
                                 + 0.9415 w_k1_MAT_s014 + 1.1709 w_k1_MSFT_s014
                                 + 1.02755 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.28949 w_k1_ADM_s014
                                 + 0.990076 w_k1_AXP_s014
                                 + 1.33087 w_k1_DIS_s014
                                 + 1.21649 w_k1_FDX_s014
                                 + 0.733644 w_k1_IBM_s014
                                 + 0.977171 w_k1_INTC_s014
                                 + 1.1954 w_k1_KO_s014 + 0.806611 w_k1_MAT_s014
                                 + 1.16787 w_k1_MSFT_s014
                                 + 1.13143 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.0881 w_k1_ADM_s014
                                 + 0.967753 w_k1_AXP_s014
                                 + 0.754089 w_k1_DIS_s014
                                 + 0.894835 w_k1_FDX_s014
                                 + 0.754478 w_k1_IBM_s014
                                 + 0.931816 w_k1_INTC_s014
                                 + 0.96327 w_k1_KO_s014
                                 + 0.878725 w_k1_MAT_s014
                                 + 0.899484 w_k1_MSFT_s014
                                 + 1.11435 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.11903 w_k1_ADM_s014
                                 + 1.0166 w_k1_AXP_s014
                                 + 0.982731 w_k1_DIS_s014
                                 + 0.949906 w_k1_FDX_s014
                                 + 1.17423 w_k1_IBM_s014
                                 + 0.864966 w_k1_INTC_s014
                                 + 1.17582 w_k1_KO_s014
                                 + 0.859909 w_k1_MAT_s014
                                 + 1.00148 w_k1_MSFT_s014
                                 + 1.20956 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.59522 w_k1_ADM_s014
                                 + 1.51143 w_k1_AXP_s014
                                 + 1.76391 w_k1_DIS_s014
                                 + 1.29227 w_k1_FDX_s014 + 1.3687 w_k1_IBM_s014
                                 + 1.03465 w_k1_INTC_s014
                                 + 1.44092 w_k1_KO_s014 + 1.28431 w_k1_MAT_s014
                                 + 1.21647 w_k1_MSFT_s014
                                 + 1.37358 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.24043 w_k1_ADM_s014
                                 + 1.08767 w_k1_AXP_s014
                                 + 0.961277 w_k1_DIS_s014
                                 + 1.16306 w_k1_FDX_s014
                                 + 0.749154 w_k1_IBM_s014
                                 + 1.08459 w_k1_INTC_s014
                                 + 1.04821 w_k1_KO_s014
                                 + 0.713665 w_k1_MAT_s014
                                 + 1.1409 w_k1_MSFT_s014 + 1.15274 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.54996 w_k1_ADM_s014
                                 + 1.34107 w_k1_AXP_s014
                                 + 1.13824 w_k1_DIS_s014
                                 + 1.33769 w_k1_FDX_s014
                                 + 0.954703 w_k1_IBM_s014
                                 + 1.21129 w_k1_INTC_s014
                                 + 1.06653 w_k1_KO_s014
                                 + 0.895738 w_k1_MAT_s014
                                 + 1.25096 w_k1_MSFT_s014
                                 + 1.15154 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.738631 w_k1_ADM_s014
                                 + 0.512439 w_k1_AXP_s014
                                 + 0.735598 w_k1_DIS_s014
                                 + 0.767999 w_k1_FDX_s014
                                 + 0.93324 w_k1_IBM_s014
                                 + 0.982326 w_k1_INTC_s014
                                 + 1.12264 w_k1_KO_s014
                                 + 0.637275 w_k1_MAT_s014
                                 + 0.953569 w_k1_MSFT_s014
                                 + 0.896775 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.865047 w_k1_ADM_s015
                                 + 0.768667 w_k1_AXP_s015
                                 + 0.73342 w_k1_DIS_s015
                                 + 0.794492 w_k1_FDX_s015
                                 + 0.957676 w_k1_IBM_s015
                                 + 0.730748 w_k1_INTC_s015
                                 + 0.949641 w_k1_KO_s015
                                 + 1.44592 w_k1_MAT_s015
                                 + 0.867344 w_k1_MSFT_s015
                                 + 1.03476 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.729776 w_k1_ADM_s015
                                 + 1.00298 w_k1_AXP_s015
                                 + 1.00199 w_k1_DIS_s015
                                 + 0.98652 w_k1_FDX_s015
                                 + 0.924473 w_k1_IBM_s015
                                 + 1.16158 w_k1_INTC_s015
                                 + 1.08284 w_k1_KO_s015 + 1.15719 w_k1_MAT_s015
                                 + 1.14609 w_k1_MSFT_s015
                                 + 1.11944 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.46075 w_k1_ADM_s015
                                 + 1.19306 w_k1_AXP_s015 + 1.5401 w_k1_DIS_s015
                                 + 1.2353 w_k1_FDX_s015 + 1.26367 w_k1_IBM_s015
                                 + 1.17781 w_k1_INTC_s015
                                 + 1.47445 w_k1_KO_s015 + 1.55327 w_k1_MAT_s015
                                 + 1.32555 w_k1_MSFT_s015
                                 + 1.21708 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.2828 w_k1_ADM_s015
                                 + 1.06278 w_k1_AXP_s015
                                 + 1.36364 w_k1_DIS_s015 + 1.2664 w_k1_FDX_s015
                                 + 1.02751 w_k1_IBM_s015
                                 + 1.20383 w_k1_INTC_s015 + 1.127 w_k1_KO_s015
                                 + 0.936192 w_k1_MAT_s015
                                 + 1.34044 w_k1_MSFT_s015
                                 + 1.12006 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.36237 w_k1_ADM_s015
                                 + 1.55647 w_k1_AXP_s015
                                 + 1.22361 w_k1_DIS_s015
                                 + 1.06806 w_k1_FDX_s015
                                 + 1.37067 w_k1_IBM_s015
                                 + 1.30454 w_k1_INTC_s015 + 1.2469 w_k1_KO_s015
                                 + 1.77771 w_k1_MAT_s015
                                 + 1.21442 w_k1_MSFT_s015
                                 + 1.21746 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.830008 w_k1_ADM_s015
                                 + 0.98309 w_k1_AXP_s015
                                 + 1.05693 w_k1_DIS_s015
                                 + 0.875911 w_k1_FDX_s015
                                 + 1.42561 w_k1_IBM_s015
                                 + 1.01109 w_k1_INTC_s015
                                 + 1.16732 w_k1_KO_s015 + 1.48901 w_k1_MAT_s015
                                 + 0.977133 w_k1_MSFT_s015
                                 + 1.13559 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.06528 w_k1_ADM_s015
                                 + 1.57183 w_k1_AXP_s015
                                 + 1.17557 w_k1_DIS_s015
                                 + 1.21152 w_k1_FDX_s015
                                 + 0.818607 w_k1_IBM_s015
                                 + 1.10825 w_k1_INTC_s015
                                 + 0.946669 w_k1_KO_s015
                                 + 1.42368 w_k1_MAT_s015
                                 + 1.13792 w_k1_MSFT_s015 + 1.1557 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.788289 w_k1_ADM_s015
                                 + 0.763234 w_k1_AXP_s015
                                 + 1.07527 w_k1_DIS_s015
                                 + 0.905439 w_k1_FDX_s015
                                 + 0.932598 w_k1_IBM_s015
                                 + 0.845532 w_k1_INTC_s015
                                 + 0.94563 w_k1_KO_s015 + 1.01067 w_k1_MAT_s015
                                 + 1.06617 w_k1_MSFT_s015
                                 + 0.941672 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.796637 w_k1_ADM_s016
                                 + 0.747771 w_k1_AXP_s016
                                 + 0.874859 w_k1_DIS_s016
                                 + 0.944599 w_k1_FDX_s016
                                 + 0.801661 w_k1_IBM_s016
                                 + 0.844796 w_k1_INTC_s016
                                 + 0.874532 w_k1_KO_s016
                                 + 0.772868 w_k1_MAT_s016
                                 + 0.95412 w_k1_MSFT_s016
                                 + 0.942209 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.19659 w_k1_ADM_s016
                                 + 1.11584 w_k1_AXP_s016
                                 + 1.14608 w_k1_DIS_s016
                                 + 0.937094 w_k1_FDX_s016
                                 + 1.38583 w_k1_IBM_s016
                                 + 1.08486 w_k1_INTC_s016
                                 + 1.50654 w_k1_KO_s016 + 1.2723 w_k1_MAT_s016
                                 + 1.04813 w_k1_MSFT_s016
                                 + 1.19653 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.719273 w_k1_ADM_s016
                                 + 0.608599 w_k1_AXP_s016
                                 + 0.720348 w_k1_DIS_s016
                                 + 0.665049 w_k1_FDX_s016
                                 + 1.3766 w_k1_IBM_s016
                                 + 0.717689 w_k1_INTC_s016
                                 + 1.00017 w_k1_KO_s016 + 0.96409 w_k1_MAT_s016
                                 + 0.756491 w_k1_MSFT_s016
                                 + 0.964449 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.67177 w_k1_ADM_s016
                                 + 1.71199 w_k1_AXP_s016
                                 + 1.42836 w_k1_DIS_s016
                                 + 1.39496 w_k1_FDX_s016 + 1.2114 w_k1_IBM_s016
                                 + 1.18502 w_k1_INTC_s016
                                 + 1.38116 w_k1_KO_s016 + 1.30287 w_k1_MAT_s016
                                 + 1.32079 w_k1_MSFT_s016
                                 + 1.47892 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.90525 w_k1_ADM_s016
                                 + 1.15297 w_k1_AXP_s016
                                 + 1.67855 w_k1_DIS_s016 + 1.2288 w_k1_FDX_s016
                                 + 1.57125 w_k1_IBM_s016
                                 + 1.03625 w_k1_INTC_s016
                                 + 1.49247 w_k1_KO_s016 + 1.49351 w_k1_MAT_s016
                                 + 1.17714 w_k1_MSFT_s016
                                 + 1.30455 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.43864 w_k1_ADM_s016
                                 + 1.4678 w_k1_AXP_s016 + 1.36 w_k1_DIS_s016
                                 + 1.25532 w_k1_FDX_s016
                                 + 0.82706 w_k1_IBM_s016
                                 + 1.15847 w_k1_INTC_s016
                                 + 1.04459 w_k1_KO_s016
                                 + 0.851918 w_k1_MAT_s016
                                 + 1.24055 w_k1_MSFT_s016
                                 + 1.17102 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.16513 w_k1_ADM_s016
                                 + 0.897992 w_k1_AXP_s016
                                 + 0.710937 w_k1_DIS_s016
                                 + 0.829061 w_k1_FDX_s016
                                 + 0.853241 w_k1_IBM_s016
                                 + 0.821331 w_k1_INTC_s016
                                 + 0.737115 w_k1_KO_s016
                                 + 0.808947 w_k1_MAT_s016
                                 + 0.857672 w_k1_MSFT_s016
                                 + 0.935603 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.2756 w_k1_ADM_s016
                                 + 1.08674 w_k1_AXP_s016
                                 + 1.07546 w_k1_DIS_s016
                                 + 1.14137 w_k1_FDX_s016
                                 + 0.797967 w_k1_IBM_s016
                                 + 0.926414 w_k1_INTC_s016
                                 + 1.01103 w_k1_KO_s016
                                 + 0.977258 w_k1_MAT_s016
                                 + 1.21967 w_k1_MSFT_s016
                                 + 1.09925 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.772695 w_k1_ADM_s017
                                 + 0.58088 w_k1_AXP_s017
                                 + 0.611418 w_k1_DIS_s017
                                 + 0.677934 w_k1_FDX_s017
                                 + 1.03624 w_k1_IBM_s017
                                 + 0.735494 w_k1_INTC_s017
                                 + 0.785582 w_k1_KO_s017
                                 + 0.819438 w_k1_MAT_s017
                                 + 0.796774 w_k1_MSFT_s017
                                 + 0.794667 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.13149 w_k1_ADM_s017
                                 + 1.11842 w_k1_AXP_s017
                                 + 1.63571 w_k1_DIS_s017
                                 + 1.20004 w_k1_FDX_s017
                                 + 1.04008 w_k1_IBM_s017
                                 + 1.35982 w_k1_INTC_s017
                                 + 1.21526 w_k1_KO_s017 + 1.2318 w_k1_MAT_s017
                                 + 1.51283 w_k1_MSFT_s017
                                 + 1.15561 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.965799 w_k1_ADM_s017
                                 + 0.630187 w_k1_AXP_s017
                                 + 1.03956 w_k1_DIS_s017
                                 + 0.879939 w_k1_FDX_s017
                                 + 1.15368 w_k1_IBM_s017
                                 + 0.954867 w_k1_INTC_s017
                                 + 1.34204 w_k1_KO_s017 + 1.24575 w_k1_MAT_s017
                                 + 0.9891 w_k1_MSFT_s017
                                 + 0.970154 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.01073 w_k1_ADM_s017
                                 + 1.50065 w_k1_AXP_s017 + 1.5026 w_k1_DIS_s017
                                 + 1.2081 w_k1_FDX_s017 + 1.52171 w_k1_IBM_s017
                                 + 1.05741 w_k1_INTC_s017
                                 + 1.25151 w_k1_KO_s017 + 2.01033 w_k1_MAT_s017
                                 + 1.15383 w_k1_MSFT_s017
                                 + 1.18498 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.90798 w_k1_ADM_s017
                                 + 1.09975 w_k1_AXP_s017 + 1.0833 w_k1_DIS_s017
                                 + 0.912697 w_k1_FDX_s017
                                 + 1.55226 w_k1_IBM_s017
                                 + 1.1196 w_k1_INTC_s017 + 1.22563 w_k1_KO_s017
                                 + 1.65351 w_k1_MAT_s017
                                 + 1.13475 w_k1_MSFT_s017
                                 + 0.960906 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.0431 w_k1_ADM_s017
                                 + 0.869803 w_k1_AXP_s017
                                 + 1.02188 w_k1_DIS_s017
                                 + 1.06511 w_k1_FDX_s017
                                 + 0.775181 w_k1_IBM_s017
                                 + 0.968477 w_k1_INTC_s017
                                 + 0.880178 w_k1_KO_s017
                                 + 1.28922 w_k1_MAT_s017
                                 + 1.0766 w_k1_MSFT_s017
                                 + 0.957844 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.08107 w_k1_ADM_s017
                                 + 0.946482 w_k1_AXP_s017
                                 + 0.878166 w_k1_DIS_s017
                                 + 0.963511 w_k1_FDX_s017
                                 + 0.769206 w_k1_IBM_s017
                                 + 0.991399 w_k1_INTC_s017
                                 + 1.00763 w_k1_KO_s017
                                 + 0.810871 w_k1_MAT_s017
                                 + 0.994231 w_k1_MSFT_s017
                                 + 0.994133 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.877115 w_k1_ADM_s017
                                 + 0.599447 w_k1_AXP_s017
                                 + 0.750287 w_k1_DIS_s017
                                 + 0.778978 w_k1_FDX_s017
                                 + 0.886157 w_k1_IBM_s017
                                 + 0.792267 w_k1_INTC_s017
                                 + 0.909699 w_k1_KO_s017
                                 + 0.920074 w_k1_MAT_s017
                                 + 0.851819 w_k1_MSFT_s017
                                 + 0.807127 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.02669 w_k1_ADM_s100
                                 + 1.09196 w_k1_AXP_s100
                                 + 1.49698 w_k1_DIS_s100
                                 + 1.22625 w_k1_FDX_s100
                                 + 0.949851 w_k1_IBM_s100
                                 + 1.33895 w_k1_INTC_s100 + 1.1444 w_k1_KO_s100
                                 + 0.822208 w_k1_MAT_s100
                                 + 1.20859 w_k1_MSFT_s100
                                 + 1.15219 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.903623 w_k1_ADM_s100
                                 + 1.18167 w_k1_AXP_s100
                                 + 1.04784 w_k1_DIS_s100
                                 + 1.06034 w_k1_FDX_s100
                                 + 1.31872 w_k1_IBM_s100
                                 + 1.00661 w_k1_INTC_s100
                                 + 0.973104 w_k1_KO_s100
                                 + 1.34008 w_k1_MAT_s100
                                 + 1.00847 w_k1_MSFT_s100 + 1.0843 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.0294 w_k1_ADM_s100
                                 + 0.89381 w_k1_AXP_s100
                                 + 0.935727 w_k1_DIS_s100
                                 + 0.956819 w_k1_FDX_s100
                                 + 0.907946 w_k1_IBM_s100
                                 + 1.00484 w_k1_INTC_s100 + 0.9211 w_k1_KO_s100
                                 + 0.786126 w_k1_MAT_s100
                                 + 0.976923 w_k1_MSFT_s100
                                 + 1.01181 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.855823 w_k1_ADM_s100
                                 + 0.559852 w_k1_AXP_s100
                                 + 0.685874 w_k1_DIS_s100
                                 + 0.761153 w_k1_FDX_s100
                                 + 0.675247 w_k1_IBM_s100
                                 + 0.846491 w_k1_INTC_s100
                                 + 0.767973 w_k1_KO_s100
                                 + 0.619434 w_k1_MAT_s100
                                 + 0.898608 w_k1_MSFT_s100
                                 + 0.830099 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.963933 w_k1_ADM_s100
                                 + 0.796928 w_k1_AXP_s100
                                 + 0.679352 w_k1_DIS_s100
                                 + 0.767546 w_k1_FDX_s100
                                 + 1.06803 w_k1_IBM_s100
                                 + 0.734013 w_k1_INTC_s100
                                 + 0.959095 w_k1_KO_s100 + 1.1019 w_k1_MAT_s100
                                 + 0.753614 w_k1_MSFT_s100
                                 + 0.977301 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.46855 w_k1_ADM_s100
                                 + 1.35276 w_k1_AXP_s100
                                 + 1.13266 w_k1_DIS_s100
                                 + 1.23894 w_k1_FDX_s100
                                 + 1.00563 w_k1_IBM_s100
                                 + 1.24841 w_k1_INTC_s100
                                 + 1.03665 w_k1_KO_s100 + 1.16602 w_k1_MAT_s100
                                 + 1.06943 w_k1_MSFT_s100
                                 + 1.17047 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.901133 w_k1_ADM_s100
                                 + 0.7519 w_k1_AXP_s100
                                 + 0.731552 w_k1_DIS_s100
                                 + 0.843938 w_k1_FDX_s100
                                 + 0.734865 w_k1_IBM_s100
                                 + 0.98133 w_k1_INTC_s100
                                 + 0.838549 w_k1_KO_s100
                                 + 0.864646 w_k1_MAT_s100
                                 + 0.87484 w_k1_MSFT_s100
                                 + 0.903778 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.15751 w_k1_ADM_s100
                                 + 0.864248 w_k1_AXP_s100
                                 + 1.26096 w_k1_DIS_s100
                                 + 1.00399 w_k1_FDX_s100
                                 + 1.41447 w_k1_IBM_s100
                                 + 1.21246 w_k1_INTC_s100
                                 + 1.25131 w_k1_KO_s100
                                 + 0.991807 w_k1_MAT_s100
                                 + 1.16305 w_k1_MSFT_s100
                                 + 1.09966 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.62764 w_k1_ADM_s101
                                 + 1.80302 w_k1_AXP_s101
                                 + 1.67561 w_k1_DIS_s101 + 1.4361 w_k1_FDX_s101
                                 + 1.40165 w_k1_IBM_s101
                                 + 1.22578 w_k1_INTC_s101
                                 + 1.54662 w_k1_KO_s101 + 1.75866 w_k1_MAT_s101
                                 + 1.30257 w_k1_MSFT_s101
                                 + 1.32494 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.20978 w_k1_ADM_s101
                                 + 1.22963 w_k1_AXP_s101
                                 + 1.24131 w_k1_DIS_s101
                                 + 1.22792 w_k1_FDX_s101
                                 + 0.893787 w_k1_IBM_s101
                                 + 1.11034 w_k1_INTC_s101
                                 + 0.947879 w_k1_KO_s101
                                 + 0.730096 w_k1_MAT_s101
                                 + 1.1368 w_k1_MSFT_s101 + 1.01003 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.925264 w_k1_ADM_s101
                                 + 0.693311 w_k1_AXP_s101
                                 + 0.66742 w_k1_DIS_s101
                                 + 0.786658 w_k1_FDX_s101
                                 + 0.92547 w_k1_IBM_s101
                                 + 0.68306 w_k1_INTC_s101
                                 + 0.80578 w_k1_KO_s101
                                 + 0.916187 w_k1_MAT_s101
                                 + 0.804219 w_k1_MSFT_s101
                                 + 0.843372 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.236 w_k1_ADM_s101
                                 + 1.1693 w_k1_AXP_s101 + 0.91511 w_k1_DIS_s101
                                 + 1.09142 w_k1_FDX_s101
                                 + 0.791424 w_k1_IBM_s101
                                 + 0.787788 w_k1_INTC_s101
                                 + 0.799083 w_k1_KO_s101
                                 + 0.692938 w_k1_MAT_s101
                                 + 0.852731 w_k1_MSFT_s101
                                 + 0.973984 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.1308 w_k1_ADM_s101
                                 + 1.26663 w_k1_AXP_s101
                                 + 0.914587 w_k1_DIS_s101
                                 + 0.951036 w_k1_FDX_s101
                                 + 1.4296 w_k1_IBM_s101
                                 + 0.893459 w_k1_INTC_s101
                                 + 1.0925 w_k1_KO_s101 + 1.31946 w_k1_MAT_s101
                                 + 0.914038 w_k1_MSFT_s101
                                 + 1.0345 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.28402 w_k1_ADM_s101
                                 + 0.929218 w_k1_AXP_s101
                                 + 1.37767 w_k1_DIS_s101
                                 + 1.05456 w_k1_FDX_s101 + 1.1149 w_k1_IBM_s101
                                 + 0.96212 w_k1_INTC_s101
                                 + 1.31395 w_k1_KO_s101
                                 + 0.912999 w_k1_MAT_s101
                                 + 1.156 w_k1_MSFT_s101 + 1.13723 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.03795 w_k1_ADM_s101
                                 + 1.01117 w_k1_AXP_s101
                                 + 1.05342 w_k1_DIS_s101
                                 + 0.961472 w_k1_FDX_s101
                                 + 0.952903 w_k1_IBM_s101
                                 + 0.890208 w_k1_INTC_s101
                                 + 0.947956 w_k1_KO_s101 + 1.1764 w_k1_MAT_s101
                                 + 1.01849 w_k1_MSFT_s101
                                 + 0.903215 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.38085 w_k1_ADM_s101
                                 + 1.26175 w_k1_AXP_s101
                                 + 1.76207 w_k1_DIS_s101
                                 + 1.31731 w_k1_FDX_s101
                                 + 1.04613 w_k1_IBM_s101
                                 + 1.02898 w_k1_INTC_s101
                                 + 1.37389 w_k1_KO_s101
                                 + 0.913884 w_k1_MAT_s101
                                 + 1.39222 w_k1_MSFT_s101
                                 + 1.07535 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.35438 w_k1_ADM_s102
                                 + 1.12848 w_k1_AXP_s102
                                 + 1.82345 w_k1_DIS_s102
                                 + 1.46896 w_k1_FDX_s102
                                 + 1.20855 w_k1_IBM_s102
                                 + 1.0844 w_k1_INTC_s102 + 1.45936 w_k1_KO_s102
                                 + 1.05524 w_k1_MAT_s102
                                 + 1.33605 w_k1_MSFT_s102
                                 + 1.21073 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.07324 w_k1_ADM_s102
                                 + 0.812709 w_k1_AXP_s102
                                 + 0.831343 w_k1_DIS_s102
                                 + 1.0187 w_k1_FDX_s102
                                 + 0.751743 w_k1_IBM_s102
                                 + 0.873249 w_k1_INTC_s102
                                 + 0.846645 w_k1_KO_s102
                                 + 0.847558 w_k1_MAT_s102
                                 + 0.937855 w_k1_MSFT_s102
                                 + 0.99298 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.651393 w_k1_ADM_s102
                                 + 0.559216 w_k1_AXP_s102
                                 + 0.624578 w_k1_DIS_s102
                                 + 0.692108 w_k1_FDX_s102
                                 + 0.801158 w_k1_IBM_s102
                                 + 0.601377 w_k1_INTC_s102
                                 + 0.779449 w_k1_KO_s102
                                 + 0.61179 w_k1_MAT_s102
                                 + 0.752749 w_k1_MSFT_s102
                                 + 0.810612 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.1981 w_k1_ADM_s102
                                 + 1.11931 w_k1_AXP_s102
                                 + 1.31309 w_k1_DIS_s102 + 1.2424 w_k1_FDX_s102
                                 + 1.01581 w_k1_IBM_s102
                                 + 0.960589 w_k1_INTC_s102
                                 + 1.04155 w_k1_KO_s102 + 1.02942 w_k1_MAT_s102
                                 + 1.21162 w_k1_MSFT_s102
                                 + 1.15831 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.894594 w_k1_ADM_s102
                                 + 0.96095 w_k1_AXP_s102
                                 + 0.598152 w_k1_DIS_s102
                                 + 0.783855 w_k1_FDX_s102
                                 + 0.818071 w_k1_IBM_s102
                                 + 0.712843 w_k1_INTC_s102
                                 + 0.667579 w_k1_KO_s102
                                 + 0.737125 w_k1_MAT_s102
                                 + 0.657671 w_k1_MSFT_s102
                                 + 0.840029 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.929176 w_k1_ADM_s102
                                 + 1.14206 w_k1_AXP_s102
                                 + 1.18676 w_k1_DIS_s102 + 1.0343 w_k1_FDX_s102
                                 + 1.55845 w_k1_IBM_s102
                                 + 1.07803 w_k1_INTC_s102
                                 + 1.24912 w_k1_KO_s102 + 1.17788 w_k1_MAT_s102
                                 + 1.01048 w_k1_MSFT_s102
                                 + 1.00382 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.740948 w_k1_ADM_s102
                                 + 0.747753 w_k1_AXP_s102
                                 + 0.801317 w_k1_DIS_s102
                                 + 0.835583 w_k1_FDX_s102
                                 + 1.16299 w_k1_IBM_s102
                                 + 0.846509 w_k1_INTC_s102
                                 + 0.941821 w_k1_KO_s102
                                 + 1.17946 w_k1_MAT_s102
                                 + 0.850578 w_k1_MSFT_s102
                                 + 0.945395 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.12426 w_k1_ADM_s102
                                 + 1.22179 w_k1_AXP_s102
                                 + 0.935336 w_k1_DIS_s102
                                 + 0.942521 w_k1_FDX_s102
                                 + 1.54761 w_k1_IBM_s102
                                 + 0.911121 w_k1_INTC_s102
                                 + 1.12485 w_k1_KO_s102 + 1.3272 w_k1_MAT_s102
                                 + 0.893578 w_k1_MSFT_s102
                                 + 0.998606 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.821191 w_k1_ADM_s103
                                 + 1.09161 w_k1_AXP_s103
                                 + 0.995026 w_k1_DIS_s103
                                 + 1.08157 w_k1_FDX_s103
                                 + 0.77278 w_k1_IBM_s103
                                 + 1.06131 w_k1_INTC_s103
                                 + 0.823484 w_k1_KO_s103
                                 + 0.830629 w_k1_MAT_s103
                                 + 1.00121 w_k1_MSFT_s103
                                 + 0.965318 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.994118 w_k1_ADM_s103
                                 + 0.903906 w_k1_AXP_s103
                                 + 1.10722 w_k1_DIS_s103
                                 + 1.06836 w_k1_FDX_s103
                                 + 1.00307 w_k1_IBM_s103
                                 + 1.16372 w_k1_INTC_s103
                                 + 1.02346 w_k1_KO_s103
                                 + 0.761567 w_k1_MAT_s103
                                 + 1.27671 w_k1_MSFT_s103
                                 + 1.09996 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.968105 w_k1_ADM_s103
                                 + 1.19753 w_k1_AXP_s103
                                 + 1.01574 w_k1_DIS_s103 + 0.9515 w_k1_FDX_s103
                                 + 1.45018 w_k1_IBM_s103
                                 + 0.928595 w_k1_INTC_s103
                                 + 1.13588 w_k1_KO_s103 + 1.5813 w_k1_MAT_s103
                                 + 0.935841 w_k1_MSFT_s103
                                 + 0.970025 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.07406 w_k1_ADM_s103
                                 + 1.41814 w_k1_AXP_s103
                                 + 1.26619 w_k1_DIS_s103
                                 + 1.31515 w_k1_FDX_s103
                                 + 0.98053 w_k1_IBM_s103
                                 + 1.22548 w_k1_INTC_s103
                                 + 0.994098 w_k1_KO_s103
                                 + 0.842759 w_k1_MAT_s103
                                 + 1.20821 w_k1_MSFT_s103
                                 + 1.04114 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.885244 w_k1_ADM_s103
                                 + 1.0562 w_k1_AXP_s103
                                 + 0.970476 w_k1_DIS_s103
                                 + 0.916683 w_k1_FDX_s103
                                 + 1.12416 w_k1_IBM_s103
                                 + 1.06267 w_k1_INTC_s103
                                 + 0.981819 w_k1_KO_s103
                                 + 1.02599 w_k1_MAT_s103
                                 + 0.926039 w_k1_MSFT_s103
                                 + 1.03215 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.02061 w_k1_ADM_s103
                                 + 2.07148 w_k1_AXP_s103
                                 + 1.58087 w_k1_DIS_s103
                                 + 1.43695 w_k1_FDX_s103
                                 + 1.36661 w_k1_IBM_s103
                                 + 1.34588 w_k1_INTC_s103 + 1.1954 w_k1_KO_s103
                                 + 1.6116 w_k1_MAT_s103
                                 + 1.36752 w_k1_MSFT_s103
                                 + 1.25942 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.769036 w_k1_ADM_s103
                                 + 0.746895 w_k1_AXP_s103
                                 + 0.778827 w_k1_DIS_s103
                                 + 0.810542 w_k1_FDX_s103
                                 + 0.852141 w_k1_IBM_s103
                                 + 0.866585 w_k1_INTC_s103
                                 + 0.880647 w_k1_KO_s103 + 0.8181 w_k1_MAT_s103
                                 + 0.919236 w_k1_MSFT_s103
                                 + 0.825534 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.716275 w_k1_ADM_s103
                                 + 0.764411 w_k1_AXP_s103
                                 + 0.572397 w_k1_DIS_s103
                                 + 0.729674 w_k1_FDX_s103
                                 + 0.679192 w_k1_IBM_s103
                                 + 0.824187 w_k1_INTC_s103
                                 + 0.750295 w_k1_KO_s103
                                 + 0.766334 w_k1_MAT_s103
                                 + 0.863045 w_k1_MSFT_s103
                                 + 0.874415 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.967066 w_k1_ADM_s104
                                 + 0.858015 w_k1_AXP_s104
                                 + 0.695025 w_k1_DIS_s104
                                 + 0.854922 w_k1_FDX_s104
                                 + 0.968087 w_k1_IBM_s104
                                 + 0.830896 w_k1_INTC_s104
                                 + 0.880043 w_k1_KO_s104
                                 + 1.07134 w_k1_MAT_s104
                                 + 0.877568 w_k1_MSFT_s104
                                 + 0.880616 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.914442 w_k1_ADM_s104
                                 + 0.846398 w_k1_AXP_s104
                                 + 0.891836 w_k1_DIS_s104
                                 + 0.918172 w_k1_FDX_s104
                                 + 0.77626 w_k1_IBM_s104
                                 + 0.944977 w_k1_INTC_s104
                                 + 0.880256 w_k1_KO_s104
                                 + 0.693976 w_k1_MAT_s104
                                 + 0.875271 w_k1_MSFT_s104
                                 + 0.908677 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.800796 w_k1_ADM_s104
                                 + 0.652063 w_k1_AXP_s104
                                 + 0.483883 w_k1_DIS_s104
                                 + 0.694167 w_k1_FDX_s104
                                 + 0.738229 w_k1_IBM_s104
                                 + 0.697262 w_k1_INTC_s104
                                 + 0.729448 w_k1_KO_s104
                                 + 0.513687 w_k1_MAT_s104
                                 + 0.633926 w_k1_MSFT_s104
                                 + 0.793381 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.81263 w_k1_ADM_s104
                                 + 0.924452 w_k1_AXP_s104
                                 + 0.758775 w_k1_DIS_s104
                                 + 0.973449 w_k1_FDX_s104
                                 + 0.669173 w_k1_IBM_s104
                                 + 0.78595 w_k1_INTC_s104
                                 + 0.704898 w_k1_KO_s104
                                 + 0.553798 w_k1_MAT_s104
                                 + 0.868339 w_k1_MSFT_s104
                                 + 0.853065 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.38714 w_k1_ADM_s104
                                 + 1.21705 w_k1_AXP_s104
                                 + 1.08812 w_k1_DIS_s104
                                 + 1.18223 w_k1_FDX_s104
                                 + 0.927789 w_k1_IBM_s104
                                 + 1.14703 w_k1_INTC_s104
                                 + 0.901781 w_k1_KO_s104
                                 + 0.735769 w_k1_MAT_s104
                                 + 1.03142 w_k1_MSFT_s104
                                 + 1.04437 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.01648 w_k1_ADM_s104
                                 + 1.1333 w_k1_AXP_s104 + 1.10916 w_k1_DIS_s104
                                 + 1.03284 w_k1_FDX_s104
                                 + 1.17659 w_k1_IBM_s104
                                 + 0.860497 w_k1_INTC_s104
                                 + 1.12918 w_k1_KO_s104 + 1.34363 w_k1_MAT_s104
                                 + 0.916531 w_k1_MSFT_s104
                                 + 1.00131 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.56835 w_k1_ADM_s104
                                 + 1.42077 w_k1_AXP_s104
                                 + 1.68299 w_k1_DIS_s104
                                 + 1.40714 w_k1_FDX_s104
                                 + 1.04323 w_k1_IBM_s104
                                 + 0.962587 w_k1_INTC_s104
                                 + 1.32001 w_k1_KO_s104 + 1.08087 w_k1_MAT_s104
                                 + 1.29244 w_k1_MSFT_s104
                                 + 1.31495 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.10858 w_k1_ADM_s104
                                 + 1.48799 w_k1_AXP_s104
                                 + 1.32663 w_k1_DIS_s104
                                 + 1.17654 w_k1_FDX_s104
                                 + 1.67754 w_k1_IBM_s104
                                 + 0.990565 w_k1_INTC_s104
                                 + 1.16065 w_k1_KO_s104 + 1.65469 w_k1_MAT_s104
                                 + 0.90328 w_k1_MSFT_s104
                                 + 1.24099 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.685512 w_k1_ADM_s105
                                 + 0.695264 w_k1_AXP_s105
                                 + 0.715141 w_k1_DIS_s105
                                 + 0.766529 w_k1_FDX_s105
                                 + 0.744325 w_k1_IBM_s105
                                 + 0.881628 w_k1_INTC_s105
                                 + 0.708923 w_k1_KO_s105
                                 + 0.498877 w_k1_MAT_s105
                                 + 0.772245 w_k1_MSFT_s105
                                 + 0.768892 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.11976 w_k1_ADM_s105
                                 + 1.44705 w_k1_AXP_s105
                                 + 1.54461 w_k1_DIS_s105 + 1.3554 w_k1_FDX_s105
                                 + 1.19514 w_k1_IBM_s105
                                 + 1.15224 w_k1_INTC_s105 + 1.0375 w_k1_KO_s105
                                 + 1.00521 w_k1_MAT_s105
                                 + 1.23677 w_k1_MSFT_s105
                                 + 1.21305 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.09983 w_k1_ADM_s105
                                 + 1.35061 w_k1_AXP_s105
                                 + 1.55447 w_k1_DIS_s105
                                 + 1.29852 w_k1_FDX_s105
                                 + 0.840849 w_k1_IBM_s105
                                 + 0.893424 w_k1_INTC_s105
                                 + 1.06128 w_k1_KO_s105 + 1.06517 w_k1_MAT_s105
                                 + 1.32915 w_k1_MSFT_s105
                                 + 1.15828 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.74843 w_k1_ADM_s105
                                 + 0.853558 w_k1_AXP_s105
                                 + 0.691511 w_k1_DIS_s105
                                 + 0.800959 w_k1_FDX_s105
                                 + 1.10408 w_k1_IBM_s105
                                 + 0.792484 w_k1_INTC_s105
                                 + 0.885153 w_k1_KO_s105
                                 + 1.02928 w_k1_MAT_s105
                                 + 0.9439 w_k1_MSFT_s105
                                 + 0.883274 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.777136 w_k1_ADM_s105
                                 + 0.875977 w_k1_AXP_s105
                                 + 0.958782 w_k1_DIS_s105
                                 + 0.980775 w_k1_FDX_s105
                                 + 0.811798 w_k1_IBM_s105
                                 + 0.873361 w_k1_INTC_s105
                                 + 0.887154 w_k1_KO_s105
                                 + 1.00084 w_k1_MAT_s105
                                 + 1.07281 w_k1_MSFT_s105
                                 + 0.925161 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.661191 w_k1_ADM_s105
                                 + 0.979885 w_k1_AXP_s105
                                 + 1.02558 w_k1_DIS_s105
                                 + 0.929473 w_k1_FDX_s105
                                 + 1.32183 w_k1_IBM_s105
                                 + 0.924404 w_k1_INTC_s105
                                 + 1.12145 w_k1_KO_s105 + 1.24956 w_k1_MAT_s105
                                 + 0.895824 w_k1_MSFT_s105
                                 + 1.01893 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.879403 w_k1_ADM_s105
                                 + 1.13498 w_k1_AXP_s105 + 1.4212 w_k1_DIS_s105
                                 + 1.09599 w_k1_FDX_s105
                                 + 1.67919 w_k1_IBM_s105
                                 + 1.00126 w_k1_INTC_s105
                                 + 1.31036 w_k1_KO_s105 + 2.27664 w_k1_MAT_s105
                                 + 1.26901 w_k1_MSFT_s105
                                 + 1.12718 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.969532 w_k1_ADM_s105
                                 + 1.00184 w_k1_AXP_s105
                                 + 1.05145 w_k1_DIS_s105
                                 + 1.20726 w_k1_FDX_s105
                                 + 0.660583 w_k1_IBM_s105
                                 + 0.993863 w_k1_INTC_s105
                                 + 0.799785 w_k1_KO_s105
                                 + 0.536441 w_k1_MAT_s105
                                 + 1.01346 w_k1_MSFT_s105
                                 + 0.919225 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.894394 w_k1_ADM_s106
                                 + 0.982884 w_k1_AXP_s106
                                 + 0.93063 w_k1_DIS_s106
                                 + 0.900752 w_k1_FDX_s106
                                 + 1.17011 w_k1_IBM_s106
                                 + 0.968624 w_k1_INTC_s106
                                 + 0.984445 w_k1_KO_s106
                                 + 0.996349 w_k1_MAT_s106
                                 + 0.910175 w_k1_MSFT_s106
                                 + 0.951967 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.86667 w_k1_ADM_s106
                                 + 0.743325 w_k1_AXP_s106
                                 + 0.689082 w_k1_DIS_s106
                                 + 0.682864 w_k1_FDX_s106
                                 + 1.02964 w_k1_IBM_s106
                                 + 0.632189 w_k1_INTC_s106
                                 + 0.896225 w_k1_KO_s106
                                 + 0.754115 w_k1_MAT_s106
                                 + 0.744397 w_k1_MSFT_s106
                                 + 0.881184 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.35437 w_k1_ADM_s106
                                 + 1.94971 w_k1_AXP_s106
                                 + 1.34354 w_k1_DIS_s106
                                 + 1.13025 w_k1_FDX_s106
                                 + 1.44805 w_k1_IBM_s106
                                 + 1.07314 w_k1_INTC_s106
                                 + 1.12217 w_k1_KO_s106 + 1.72124 w_k1_MAT_s106
                                 + 0.985557 w_k1_MSFT_s106
                                 + 1.17112 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.924272 w_k1_ADM_s106
                                 + 0.874438 w_k1_AXP_s106
                                 + 0.791517 w_k1_DIS_s106
                                 + 0.972948 w_k1_FDX_s106
                                 + 0.731583 w_k1_IBM_s106
                                 + 0.999118 w_k1_INTC_s106
                                 + 0.726178 w_k1_KO_s106
                                 + 0.594785 w_k1_MAT_s106
                                 + 0.936044 w_k1_MSFT_s106
                                 + 0.862048 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.761947 w_k1_ADM_s106
                                 + 0.712235 w_k1_AXP_s106
                                 + 0.6252 w_k1_DIS_s106
                                 + 0.752146 w_k1_FDX_s106
                                 + 0.658734 w_k1_IBM_s106
                                 + 0.819009 w_k1_INTC_s106
                                 + 0.77823 w_k1_KO_s106
                                 + 0.620705 w_k1_MAT_s106
                                 + 0.78692 w_k1_MSFT_s106
                                 + 0.895655 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.20611 w_k1_ADM_s106
                                 + 1.29987 w_k1_AXP_s106
                                 + 1.03931 w_k1_DIS_s106 + 1.0441 w_k1_FDX_s106
                                 + 0.941643 w_k1_IBM_s106
                                 + 0.941908 w_k1_INTC_s106
                                 + 0.936275 w_k1_KO_s106
                                 + 0.853262 w_k1_MAT_s106
                                 + 0.867524 w_k1_MSFT_s106
                                 + 0.931134 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.03008 w_k1_ADM_s106
                                 + 1.23034 w_k1_AXP_s106 + 1.1551 w_k1_DIS_s106
                                 + 1.11938 w_k1_FDX_s106
                                 + 1.09092 w_k1_IBM_s106
                                 + 1.21026 w_k1_INTC_s106
                                 + 1.00816 w_k1_KO_s106
                                 + 0.852117 w_k1_MAT_s106
                                 + 1.07257 w_k1_MSFT_s106
                                 + 1.09841 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.16069 w_k1_ADM_s106
                                 + 1.2053 w_k1_AXP_s106 + 1.57798 w_k1_DIS_s106
                                 + 1.26113 w_k1_FDX_s106 + 0.9692 w_k1_IBM_s106
                                 + 1.32168 w_k1_INTC_s106
                                 + 1.17951 w_k1_KO_s106
                                 + 0.882847 w_k1_MAT_s106
                                 + 1.28502 w_k1_MSFT_s106
                                 + 1.05862 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.721311 w_k1_ADM_s107
                                 + 0.915742 w_k1_AXP_s107
                                 + 0.754179 w_k1_DIS_s107
                                 + 0.864078 w_k1_FDX_s107
                                 + 0.724295 w_k1_IBM_s107
                                 + 0.86576 w_k1_INTC_s107
                                 + 0.796383 w_k1_KO_s107
                                 + 0.923639 w_k1_MAT_s107
                                 + 0.82028 w_k1_MSFT_s107
                                 + 0.862948 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.26769 w_k1_ADM_s107
                                 + 1.53853 w_k1_AXP_s107
                                 + 1.14518 w_k1_DIS_s107
                                 + 1.24434 w_k1_FDX_s107
                                 + 0.982613 w_k1_IBM_s107
                                 + 1.08139 w_k1_INTC_s107
                                 + 1.18148 w_k1_KO_s107 + 1.38048 w_k1_MAT_s107
                                 + 1.08659 w_k1_MSFT_s107
                                 + 1.09024 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.1785 w_k1_ADM_s107
                                 + 1.05796 w_k1_AXP_s107
                                 + 0.95221 w_k1_DIS_s107
                                 + 1.03623 w_k1_FDX_s107
                                 + 1.01979 w_k1_IBM_s107
                                 + 1.05455 w_k1_INTC_s107
                                 + 0.988686 w_k1_KO_s107
                                 + 0.90974 w_k1_MAT_s107
                                 + 0.979169 w_k1_MSFT_s107
                                 + 0.955225 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.09746 w_k1_ADM_s107
                                 + 1.41437 w_k1_AXP_s107
                                 + 1.49246 w_k1_DIS_s107
                                 + 1.32765 w_k1_FDX_s107
                                 + 0.906596 w_k1_IBM_s107
                                 + 1.31153 w_k1_INTC_s107
                                 + 1.08379 w_k1_KO_s107
                                 + 0.859776 w_k1_MAT_s107
                                 + 1.20211 w_k1_MSFT_s107
                                 + 1.01931 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.902775 w_k1_ADM_s107
                                 + 1.12708 w_k1_AXP_s107
                                 + 1.02178 w_k1_DIS_s107
                                 + 0.996123 w_k1_FDX_s107
                                 + 1.04963 w_k1_IBM_s107
                                 + 1.00797 w_k1_INTC_s107
                                 + 1.03464 w_k1_KO_s107 + 1.32309 w_k1_MAT_s107
                                 + 0.938747 w_k1_MSFT_s107
                                 + 0.892322 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.982298 w_k1_ADM_s107
                                 + 1.14416 w_k1_AXP_s107
                                 + 1.20136 w_k1_DIS_s107
                                 + 1.12324 w_k1_FDX_s107
                                 + 1.59948 w_k1_IBM_s107
                                 + 1.09218 w_k1_INTC_s107
                                 + 1.31999 w_k1_KO_s107 + 1.71792 w_k1_MAT_s107
                                 + 0.985193 w_k1_MSFT_s107
                                 + 1.06349 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.657483 w_k1_ADM_s107
                                 + 0.700711 w_k1_AXP_s107
                                 + 0.619873 w_k1_DIS_s107
                                 + 0.702388 w_k1_FDX_s107
                                 + 0.775527 w_k1_IBM_s107
                                 + 0.805914 w_k1_INTC_s107
                                 + 0.71418 w_k1_KO_s107
                                 + 0.782961 w_k1_MAT_s107
                                 + 0.716371 w_k1_MSFT_s107
                                 + 0.717877 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.31961 w_k1_ADM_s107
                                 + 1.62155 w_k1_AXP_s107 + 1.6009 w_k1_DIS_s107
                                 + 1.34893 w_k1_FDX_s107
                                 + 1.38039 w_k1_IBM_s107
                                 + 1.17176 w_k1_INTC_s107
                                 + 1.33056 w_k1_KO_s107 + 1.56328 w_k1_MAT_s107
                                 + 1.05283 w_k1_MSFT_s107 + 1.1224 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.10233 w_k1_ADM_s110
                                 + 1.24178 w_k1_AXP_s110
                                 + 1.44474 w_k1_DIS_s110
                                 + 1.09662 w_k1_FDX_s110
                                 + 1.37259 w_k1_IBM_s110
                                 + 1.1149 w_k1_INTC_s110 + 1.29566 w_k1_KO_s110
                                 + 1.57332 w_k1_MAT_s110
                                 + 1.25301 w_k1_MSFT_s110
                                 + 1.13253 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.956255 w_k1_ADM_s110
                                 + 1.30744 w_k1_AXP_s110
                                 + 0.944661 w_k1_DIS_s110
                                 + 0.878391 w_k1_FDX_s110
                                 + 1.34168 w_k1_IBM_s110
                                 + 0.933671 w_k1_INTC_s110
                                 + 0.948022 w_k1_KO_s110
                                 + 1.81376 w_k1_MAT_s110
                                 + 0.939776 w_k1_MSFT_s110
                                 + 0.965228 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.13655 w_k1_ADM_s110
                                 + 1.21631 w_k1_AXP_s110
                                 + 1.51583 w_k1_DIS_s110
                                 + 1.26447 w_k1_FDX_s110
                                 + 0.939008 w_k1_IBM_s110
                                 + 1.24727 w_k1_INTC_s110
                                 + 1.03947 w_k1_KO_s110 + 1.31012 w_k1_MAT_s110
                                 + 1.30643 w_k1_MSFT_s110
                                 + 1.04373 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.777905 w_k1_ADM_s110
                                 + 0.953123 w_k1_AXP_s110
                                 + 0.937024 w_k1_DIS_s110
                                 + 0.867278 w_k1_FDX_s110
                                 + 0.959344 w_k1_IBM_s110
                                 + 0.920813 w_k1_INTC_s110
                                 + 0.887102 w_k1_KO_s110
                                 + 1.17565 w_k1_MAT_s110
                                 + 0.965301 w_k1_MSFT_s110
                                 + 0.996275 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.8812 w_k1_ADM_s110
                                 + 0.974119 w_k1_AXP_s110
                                 + 0.913695 w_k1_DIS_s110
                                 + 1.07467 w_k1_FDX_s110
                                 + 0.590815 w_k1_IBM_s110
                                 + 1.00859 w_k1_INTC_s110
                                 + 0.630635 w_k1_KO_s110
                                 + 0.713353 w_k1_MAT_s110
                                 + 0.967029 w_k1_MSFT_s110
                                 + 0.890629 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.833334 w_k1_ADM_s110
                                 + 0.689214 w_k1_AXP_s110
                                 + 1.21886 w_k1_DIS_s110
                                 + 0.957625 w_k1_FDX_s110
                                 + 1.11891 w_k1_IBM_s110
                                 + 0.925694 w_k1_INTC_s110
                                 + 1.08841 w_k1_KO_s110 + 1.18559 w_k1_MAT_s110
                                 + 1.19785 w_k1_MSFT_s110
                                 + 1.09353 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.710357 w_k1_ADM_s110
                                 + 0.819347 w_k1_AXP_s110
                                 + 0.670268 w_k1_DIS_s110
                                 + 0.738624 w_k1_FDX_s110
                                 + 0.642582 w_k1_IBM_s110
                                 + 0.939534 w_k1_INTC_s110
                                 + 0.645688 w_k1_KO_s110
                                 + 0.78811 w_k1_MAT_s110
                                 + 0.773774 w_k1_MSFT_s110
                                 + 0.808287 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.674766 w_k1_ADM_s110
                                 + 0.750893 w_k1_AXP_s110
                                 + 0.730879 w_k1_DIS_s110
                                 + 0.667222 w_k1_FDX_s110
                                 + 1.06013 w_k1_IBM_s110
                                 + 0.71644 w_k1_INTC_s110
                                 + 0.778922 w_k1_KO_s110
                                 + 1.32393 w_k1_MAT_s110
                                 + 0.783693 w_k1_MSFT_s110
                                 + 0.885718 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.97604 w_k1_ADM_s111
                                 + 0.993085 w_k1_AXP_s111
                                 + 1.34785 w_k1_DIS_s111
                                 + 1.04739 w_k1_FDX_s111
                                 + 1.03497 w_k1_IBM_s111
                                 + 1.16067 w_k1_INTC_s111
                                 + 1.22652 w_k1_KO_s111
                                 + 0.837717 w_k1_MAT_s111
                                 + 1.14713 w_k1_MSFT_s111
                                 + 1.08203 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.904299 w_k1_ADM_s111
                                 + 0.921376 w_k1_AXP_s111
                                 + 0.916373 w_k1_DIS_s111
                                 + 1.00266 w_k1_FDX_s111
                                 + 0.693472 w_k1_IBM_s111
                                 + 0.995614 w_k1_INTC_s111
                                 + 0.855346 w_k1_KO_s111
                                 + 0.803328 w_k1_MAT_s111
                                 + 1.02478 w_k1_MSFT_s111
                                 + 0.993446 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.819425 w_k1_ADM_s111
                                 + 0.930312 w_k1_AXP_s111
                                 + 0.804503 w_k1_DIS_s111
                                 + 0.871051 w_k1_FDX_s111
                                 + 0.956768 w_k1_IBM_s111
                                 + 1.01256 w_k1_INTC_s111
                                 + 0.879583 w_k1_KO_s111
                                 + 1.07245 w_k1_MAT_s111
                                 + 0.883342 w_k1_MSFT_s111
                                 + 0.904873 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.26641 w_k1_ADM_s111
                                 + 1.53828 w_k1_AXP_s111
                                 + 1.40926 w_k1_DIS_s111
                                 + 1.31536 w_k1_FDX_s111
                                 + 1.01505 w_k1_IBM_s111
                                 + 1.36873 w_k1_INTC_s111
                                 + 1.06227 w_k1_KO_s111 + 1.64763 w_k1_MAT_s111
                                 + 1.31725 w_k1_MSFT_s111
                                 + 1.19119 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.892493 w_k1_ADM_s111
                                 + 1.0169 w_k1_AXP_s111
                                 + 0.897303 w_k1_DIS_s111
                                 + 0.890193 w_k1_FDX_s111
                                 + 0.896561 w_k1_IBM_s111
                                 + 0.927917 w_k1_INTC_s111
                                 + 0.934217 w_k1_KO_s111
                                 + 0.92036 w_k1_MAT_s111
                                 + 0.861872 w_k1_MSFT_s111
                                 + 1.10164 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.67046 w_k1_ADM_s111
                                 + 0.759672 w_k1_AXP_s111
                                 + 0.716871 w_k1_DIS_s111
                                 + 0.737834 w_k1_FDX_s111
                                 + 0.792432 w_k1_IBM_s111
                                 + 0.773981 w_k1_INTC_s111
                                 + 0.697617 w_k1_KO_s111
                                 + 0.937972 w_k1_MAT_s111
                                 + 0.753734 w_k1_MSFT_s111
                                 + 0.862107 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.967789 w_k1_ADM_s111
                                 + 1.08989 w_k1_AXP_s111
                                 + 1.00726 w_k1_DIS_s111
                                 + 0.849943 w_k1_FDX_s111
                                 + 1.41384 w_k1_IBM_s111
                                 + 1.27818 w_k1_INTC_s111
                                 + 1.15437 w_k1_KO_s111 + 1.33868 w_k1_MAT_s111
                                 + 1.04706 w_k1_MSFT_s111
                                 + 0.949946 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.934825 w_k1_ADM_s111
                                 + 1.43548 w_k1_AXP_s111
                                 + 1.28205 w_k1_DIS_s111
                                 + 1.12685 w_k1_FDX_s111
                                 + 1.33912 w_k1_IBM_s111
                                 + 1.24653 w_k1_INTC_s111
                                 + 1.16175 w_k1_KO_s111 + 1.4674 w_k1_MAT_s111
                                 + 1.07339 w_k1_MSFT_s111
                                 + 1.11172 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.08473 w_k1_ADM_s112
                                 + 0.82054 w_k1_AXP_s112
                                 + 1.22595 w_k1_DIS_s112
                                 + 1.22047 w_k1_FDX_s112
                                 + 0.647672 w_k1_IBM_s112
                                 + 1.16148 w_k1_INTC_s112
                                 + 0.949497 w_k1_KO_s112
                                 + 0.554427 w_k1_MAT_s112
                                 + 1.29519 w_k1_MSFT_s112
                                 + 1.05714 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.863459 w_k1_ADM_s112
                                 + 0.805716 w_k1_AXP_s112
                                 + 0.826548 w_k1_DIS_s112
                                 + 0.849997 w_k1_FDX_s112
                                 + 0.960582 w_k1_IBM_s112
                                 + 0.877341 w_k1_INTC_s112
                                 + 0.870811 w_k1_KO_s112
                                 + 0.82515 w_k1_MAT_s112
                                 + 0.89659 w_k1_MSFT_s112
                                 + 0.903206 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.852763 w_k1_ADM_s112
                                 + 0.900122 w_k1_AXP_s112
                                 + 0.698508 w_k1_DIS_s112
                                 + 0.780271 w_k1_FDX_s112
                                 + 1.3516 w_k1_IBM_s112
                                 + 0.907287 w_k1_INTC_s112
                                 + 0.945631 w_k1_KO_s112
                                 + 1.67107 w_k1_MAT_s112
                                 + 0.852941 w_k1_MSFT_s112
                                 + 0.947876 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.05251 w_k1_ADM_s112
                                 + 0.945157 w_k1_AXP_s112
                                 + 1.2335 w_k1_DIS_s112 + 1.11842 w_k1_FDX_s112
                                 + 0.999991 w_k1_IBM_s112
                                 + 0.916051 w_k1_INTC_s112
                                 + 1.0326 w_k1_KO_s112 + 0.939315 w_k1_MAT_s112
                                 + 1.11493 w_k1_MSFT_s112
                                 + 1.03543 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.806711 w_k1_ADM_s112
                                 + 0.56255 w_k1_AXP_s112
                                 + 0.640037 w_k1_DIS_s112
                                 + 0.730948 w_k1_FDX_s112
                                 + 0.721427 w_k1_IBM_s112
                                 + 0.73301 w_k1_INTC_s112
                                 + 0.731059 w_k1_KO_s112
                                 + 0.782033 w_k1_MAT_s112
                                 + 0.775068 w_k1_MSFT_s112
                                 + 0.864482 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.00784 w_k1_ADM_s112
                                 + 1.17911 w_k1_AXP_s112
                                 + 1.28973 w_k1_DIS_s112
                                 + 1.18368 w_k1_FDX_s112
                                 + 1.61311 w_k1_IBM_s112
                                 + 1.09013 w_k1_INTC_s112
                                 + 1.36406 w_k1_KO_s112 + 1.45903 w_k1_MAT_s112
                                 + 1.29849 w_k1_MSFT_s112
                                 + 1.12485 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.02278 w_k1_ADM_s112
                                 + 0.785883 w_k1_AXP_s112
                                 + 0.763299 w_k1_DIS_s112
                                 + 0.993839 w_k1_FDX_s112
                                 + 0.716179 w_k1_IBM_s112
                                 + 0.93511 w_k1_INTC_s112
                                 + 0.867421 w_k1_KO_s112
                                 + 0.519038 w_k1_MAT_s112
                                 + 1.06113 w_k1_MSFT_s112
                                 + 0.895647 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.98683 w_k1_ADM_s112
                                 + 1.27298 w_k1_AXP_s112
                                 + 1.16187 w_k1_DIS_s112
                                 + 1.24292 w_k1_FDX_s112
                                 + 0.880037 w_k1_IBM_s112
                                 + 1.28661 w_k1_INTC_s112
                                 + 0.957029 w_k1_KO_s112
                                 + 0.908055 w_k1_MAT_s112
                                 + 1.11048 w_k1_MSFT_s112
                                 + 0.915257 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.81326 w_k1_ADM_s113
                                 + 0.765609 w_k1_AXP_s113
                                 + 0.741686 w_k1_DIS_s113
                                 + 0.68421 w_k1_FDX_s113 + 1.1526 w_k1_IBM_s113
                                 + 0.666443 w_k1_INTC_s113
                                 + 0.954193 w_k1_KO_s113
                                 + 1.38299 w_k1_MAT_s113
                                 + 0.736363 w_k1_MSFT_s113
                                 + 0.889604 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.949514 w_k1_ADM_s113
                                 + 0.622407 w_k1_AXP_s113
                                 + 0.606849 w_k1_DIS_s113
                                 + 0.82381 w_k1_FDX_s113
                                 + 0.656976 w_k1_IBM_s113
                                 + 0.762699 w_k1_INTC_s113
                                 + 0.773354 w_k1_KO_s113
                                 + 0.60824 w_k1_MAT_s113
                                 + 0.779374 w_k1_MSFT_s113
                                 + 0.86445 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.29976 w_k1_ADM_s113
                                 + 1.27386 w_k1_AXP_s113
                                 + 1.33033 w_k1_DIS_s113
                                 + 1.08202 w_k1_FDX_s113
                                 + 1.40406 w_k1_IBM_s113
                                 + 0.949274 w_k1_INTC_s113
                                 + 1.52633 w_k1_KO_s113 + 1.78105 w_k1_MAT_s113
                                 + 1.21878 w_k1_MSFT_s113
                                 + 1.11424 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.06167 w_k1_ADM_s113
                                 + 0.879907 w_k1_AXP_s113
                                 + 0.93623 w_k1_DIS_s113
                                 + 0.961599 w_k1_FDX_s113
                                 + 0.806245 w_k1_IBM_s113
                                 + 0.876981 w_k1_INTC_s113
                                 + 0.950507 w_k1_KO_s113
                                 + 1.30291 w_k1_MAT_s113
                                 + 1.05567 w_k1_MSFT_s113
                                 + 0.987564 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.23445 w_k1_ADM_s113
                                 + 0.91348 w_k1_AXP_s113
                                 + 1.07477 w_k1_DIS_s113
                                 + 0.942853 w_k1_FDX_s113
                                 + 1.31505 w_k1_IBM_s113
                                 + 0.874917 w_k1_INTC_s113
                                 + 1.13129 w_k1_KO_s113 + 1.16649 w_k1_MAT_s113
                                 + 0.98679 w_k1_MSFT_s113 + 1.1055 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.24908 w_k1_ADM_s113
                                 + 1.17605 w_k1_AXP_s113
                                 + 1.29314 w_k1_DIS_s113 + 1.2679 w_k1_FDX_s113
                                 + 0.975113 w_k1_IBM_s113
                                 + 1.04932 w_k1_INTC_s113
                                 + 1.11262 w_k1_KO_s113 + 1.44288 w_k1_MAT_s113
                                 + 1.21135 w_k1_MSFT_s113
                                 + 1.10773 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.44239 w_k1_ADM_s113
                                 + 1.46573 w_k1_AXP_s113
                                 + 1.86271 w_k1_DIS_s113
                                 + 1.34923 w_k1_FDX_s113
                                 + 1.51156 w_k1_IBM_s113
                                 + 1.29769 w_k1_INTC_s113
                                 + 1.58966 w_k1_KO_s113 + 1.98285 w_k1_MAT_s113
                                 + 1.36303 w_k1_MSFT_s113
                                 + 1.24493 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.95409 w_k1_ADM_s113
                                 + 0.820585 w_k1_AXP_s113
                                 + 0.866849 w_k1_DIS_s113
                                 + 0.891528 w_k1_FDX_s113
                                 + 1.20104 w_k1_IBM_s113
                                 + 0.710819 w_k1_INTC_s113
                                 + 1.0939 w_k1_KO_s113 + 1.33802 w_k1_MAT_s113
                                 + 1.01062 w_k1_MSFT_s113
                                 + 1.06346 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.946135 w_k1_ADM_s114
                                 + 1.10333 w_k1_AXP_s114
                                 + 1.18721 w_k1_DIS_s114
                                 + 1.13131 w_k1_FDX_s114
                                 + 0.912997 w_k1_IBM_s114
                                 + 1.10434 w_k1_INTC_s114
                                 + 1.02273 w_k1_KO_s114
                                 + 0.738261 w_k1_MAT_s114
                                 + 1.04723 w_k1_MSFT_s114
                                 + 1.01592 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.06061 w_k1_ADM_s114
                                 + 1.48008 w_k1_AXP_s114
                                 + 1.59703 w_k1_DIS_s114
                                 + 1.36847 w_k1_FDX_s114
                                 + 1.20422 w_k1_IBM_s114
                                 + 1.38027 w_k1_INTC_s114
                                 + 1.20536 w_k1_KO_s114 + 1.59019 w_k1_MAT_s114
                                 + 1.2553 w_k1_MSFT_s114 + 1.39585 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.663 w_k1_ADM_s114
                                 + 0.820642 w_k1_AXP_s114
                                 + 0.888403 w_k1_DIS_s114
                                 + 0.857887 w_k1_FDX_s114
                                 + 1.20378 w_k1_IBM_s114
                                 + 0.862629 w_k1_INTC_s114
                                 + 1.06916 w_k1_KO_s114 + 1.08245 w_k1_MAT_s114
                                 + 0.838857 w_k1_MSFT_s114
                                 + 0.937272 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.847637 w_k1_ADM_s114
                                 + 0.716576 w_k1_AXP_s114
                                 + 0.533269 w_k1_DIS_s114
                                 + 0.724473 w_k1_FDX_s114
                                 + 0.681672 w_k1_IBM_s114
                                 + 0.625321 w_k1_INTC_s114
                                 + 0.648592 w_k1_KO_s114
                                 + 0.621467 w_k1_MAT_s114
                                 + 0.607645 w_k1_MSFT_s114
                                 + 0.891539 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.834146 w_k1_ADM_s114
                                 + 0.889593 w_k1_AXP_s114
                                 + 0.918083 w_k1_DIS_s114
                                 + 0.955339 w_k1_FDX_s114
                                 + 0.762261 w_k1_IBM_s114
                                 + 0.950517 w_k1_INTC_s114
                                 + 0.803519 w_k1_KO_s114
                                 + 0.796167 w_k1_MAT_s114
                                 + 0.88719 w_k1_MSFT_s114
                                 + 1.04301 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.1061 w_k1_ADM_s114
                                 + 1.07338 w_k1_AXP_s114
                                 + 1.12791 w_k1_DIS_s114
                                 + 1.04337 w_k1_FDX_s114
                                 + 1.19388 w_k1_IBM_s114
                                 + 0.886713 w_k1_INTC_s114
                                 + 1.14389 w_k1_KO_s114 + 1.36325 w_k1_MAT_s114
                                 + 1.11614 w_k1_MSFT_s114 + 1.2132 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.06795 w_k1_ADM_s114
                                 + 1.23908 w_k1_AXP_s114 + 1.0216 w_k1_DIS_s114
                                 + 1.07517 w_k1_FDX_s114
                                 + 1.41351 w_k1_IBM_s114
                                 + 1.06206 w_k1_INTC_s114
                                 + 1.12519 w_k1_KO_s114 + 1.43418 w_k1_MAT_s114
                                 + 0.966835 w_k1_MSFT_s114
                                 + 1.03167 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.882676 w_k1_ADM_s114
                                 + 1.02947 w_k1_AXP_s114
                                 + 0.740459 w_k1_DIS_s114
                                 + 0.875822 w_k1_FDX_s114
                                 + 0.97866 w_k1_IBM_s114
                                 + 0.901781 w_k1_INTC_s114
                                 + 0.837271 w_k1_KO_s114
                                 + 0.901729 w_k1_MAT_s114
                                 + 0.897458 w_k1_MSFT_s114
                                 + 0.904808 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.769366 w_k1_ADM_s115
                                 + 0.721153 w_k1_AXP_s115
                                 + 0.621809 w_k1_DIS_s115
                                 + 0.775693 w_k1_FDX_s115
                                 + 1.03716 w_k1_IBM_s115
                                 + 0.72875 w_k1_INTC_s115
                                 + 0.80804 w_k1_KO_s115
                                 + 0.893812 w_k1_MAT_s115
                                 + 0.780338 w_k1_MSFT_s115
                                 + 0.952648 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.5565 w_k1_ADM_s115
                                 + 1.67384 w_k1_AXP_s115
                                 + 1.53549 w_k1_DIS_s115
                                 + 1.52556 w_k1_FDX_s115
                                 + 0.892857 w_k1_IBM_s115
                                 + 1.22759 w_k1_INTC_s115
                                 + 1.11915 w_k1_KO_s115 + 1.23692 w_k1_MAT_s115
                                 + 1.30902 w_k1_MSFT_s115
                                 + 1.27234 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.15739 w_k1_ADM_s115
                                 + 1.2582 w_k1_AXP_s115 + 1.28896 w_k1_DIS_s115
                                 + 1.18038 w_k1_FDX_s115
                                 + 1.00638 w_k1_IBM_s115
                                 + 1.07331 w_k1_INTC_s115
                                 + 1.09138 w_k1_KO_s115 + 1.02195 w_k1_MAT_s115
                                 + 1.0982 w_k1_MSFT_s115 + 1.22905 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.879636 w_k1_ADM_s115
                                 + 0.875071 w_k1_AXP_s115
                                 + 0.951785 w_k1_DIS_s115
                                 + 0.894341 w_k1_FDX_s115
                                 + 1.03594 w_k1_IBM_s115
                                 + 1.0306 w_k1_INTC_s115 + 1.14908 w_k1_KO_s115
                                 + 0.812212 w_k1_MAT_s115
                                 + 0.899798 w_k1_MSFT_s115
                                 + 1.04383 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.968069 w_k1_ADM_s115
                                 + 0.897415 w_k1_AXP_s115
                                 + 0.806171 w_k1_DIS_s115
                                 + 0.771767 w_k1_FDX_s115 + 1.416 w_k1_IBM_s115
                                 + 0.881572 w_k1_INTC_s115
                                 + 1.05085 w_k1_KO_s115
                                 + 0.946521 w_k1_MAT_s115
                                 + 0.864481 w_k1_MSFT_s115
                                 + 1.00823 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.05819 w_k1_ADM_s115
                                 + 1.33671 w_k1_AXP_s115 + 1.249 w_k1_DIS_s115
                                 + 1.15768 w_k1_FDX_s115 + 1.4853 w_k1_IBM_s115
                                 + 1.1977 w_k1_INTC_s115 + 1.24551 w_k1_KO_s115
                                 + 1.52747 w_k1_MAT_s115
                                 + 1.08127 w_k1_MSFT_s115
                                 + 1.30764 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.742737 w_k1_ADM_s115
                                 + 0.70269 w_k1_AXP_s115
                                 + 0.650333 w_k1_DIS_s115
                                 + 0.709602 w_k1_FDX_s115
                                 + 0.705246 w_k1_IBM_s115
                                 + 0.67825 w_k1_INTC_s115
                                 + 0.867433 w_k1_KO_s115
                                 + 0.753459 w_k1_MAT_s115
                                 + 0.72529 w_k1_MSFT_s115
                                 + 0.854537 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.847419 w_k1_ADM_s115
                                 + 0.947008 w_k1_AXP_s115
                                 + 0.895009 w_k1_DIS_s115
                                 + 0.978077 w_k1_FDX_s115
                                 + 0.728841 w_k1_IBM_s115
                                 + 0.885498 w_k1_INTC_s115
                                 + 0.875621 w_k1_KO_s115
                                 + 0.896215 w_k1_MAT_s115
                                 + 0.89138 w_k1_MSFT_s115
                                 + 1.02699 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.23351 w_k1_ADM_s116
                                 + 1.17209 w_k1_AXP_s116 + 1.5155 w_k1_DIS_s116
                                 + 1.21857 w_k1_FDX_s116
                                 + 1.19163 w_k1_IBM_s116
                                 + 1.29793 w_k1_INTC_s116
                                 + 1.33055 w_k1_KO_s116 + 1.69551 w_k1_MAT_s116
                                 + 1.33066 w_k1_MSFT_s116
                                 + 1.26781 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.955339 w_k1_ADM_s116
                                 + 0.866201 w_k1_AXP_s116
                                 + 0.775533 w_k1_DIS_s116
                                 + 0.895229 w_k1_FDX_s116
                                 + 0.801946 w_k1_IBM_s116
                                 + 0.85396 w_k1_INTC_s116 + 0.7883 w_k1_KO_s116
                                 + 0.764363 w_k1_MAT_s116
                                 + 0.822391 w_k1_MSFT_s116
                                 + 0.874759 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.845471 w_k1_ADM_s116
                                 + 0.687689 w_k1_AXP_s116
                                 + 0.579004 w_k1_DIS_s116
                                 + 0.671884 w_k1_FDX_s116
                                 + 0.827108 w_k1_IBM_s116
                                 + 0.8734 w_k1_INTC_s116
                                 + 0.757836 w_k1_KO_s116
                                 + 0.748593 w_k1_MAT_s116
                                 + 0.771912 w_k1_MSFT_s116
                                 + 0.844049 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.3563 w_k1_ADM_s116
                                 + 1.07066 w_k1_AXP_s116
                                 + 1.37008 w_k1_DIS_s116
                                 + 1.01291 w_k1_FDX_s116
                                 + 1.73879 w_k1_IBM_s116 + 1.03 w_k1_INTC_s116
                                 + 1.43695 w_k1_KO_s116 + 1.49215 w_k1_MAT_s116
                                 + 1.10453 w_k1_MSFT_s116
                                 + 1.11746 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.894996 w_k1_ADM_s116
                                 + 0.86327 w_k1_AXP_s116
                                 + 0.878607 w_k1_DIS_s116
                                 + 0.867555 w_k1_FDX_s116
                                 + 1.13466 w_k1_IBM_s116
                                 + 0.960424 w_k1_INTC_s116
                                 + 1.04432 w_k1_KO_s116
                                 + 0.952119 w_k1_MAT_s116
                                 + 0.94404 w_k1_MSFT_s116
                                 + 0.90661 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.29004 w_k1_ADM_s116
                                 + 1.23452 w_k1_AXP_s116
                                 + 1.10284 w_k1_DIS_s116
                                 + 1.06003 w_k1_FDX_s116
                                 + 1.09654 w_k1_IBM_s116
                                 + 0.881902 w_k1_INTC_s116
                                 + 1.08951 w_k1_KO_s116 + 1.45308 w_k1_MAT_s116
                                 + 1.0312 w_k1_MSFT_s116 + 1.16016 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.785777 w_k1_ADM_s116
                                 + 0.754359 w_k1_AXP_s116
                                 + 0.650044 w_k1_DIS_s116
                                 + 0.741605 w_k1_FDX_s116
                                 + 1.29415 w_k1_IBM_s116
                                 + 0.829779 w_k1_INTC_s116
                                 + 1.03553 w_k1_KO_s116 + 1.38126 w_k1_MAT_s116
                                 + 0.700382 w_k1_MSFT_s116
                                 + 0.885477 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.17051 w_k1_ADM_s116
                                 + 1.09459 w_k1_AXP_s116
                                 + 1.26137 w_k1_DIS_s116
                                 + 1.19939 w_k1_FDX_s116
                                 + 0.86111 w_k1_IBM_s116
                                 + 1.00756 w_k1_INTC_s116
                                 + 0.889204 w_k1_KO_s116
                                 + 0.912097 w_k1_MAT_s116
                                 + 1.11268 w_k1_MSFT_s116
                                 + 1.04847 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.0205 w_k1_ADM_s117
                                 + 0.858453 w_k1_AXP_s117
                                 + 0.762038 w_k1_DIS_s117
                                 + 0.901297 w_k1_FDX_s117
                                 + 0.819245 w_k1_IBM_s117
                                 + 1.03649 w_k1_INTC_s117
                                 + 0.75208 w_k1_KO_s117
                                 + 0.643601 w_k1_MAT_s117
                                 + 0.823366 w_k1_MSFT_s117
                                 + 0.865278 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.02665 w_k1_ADM_s117
                                 + 1.60763 w_k1_AXP_s117
                                 + 1.31405 w_k1_DIS_s117
                                 + 1.08609 w_k1_FDX_s117
                                 + 1.78839 w_k1_IBM_s117
                                 + 1.23262 w_k1_INTC_s117
                                 + 1.26827 w_k1_KO_s117 + 1.83226 w_k1_MAT_s117
                                 + 1.13464 w_k1_MSFT_s117
                                 + 1.05482 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.07705 w_k1_ADM_s117
                                 + 0.935617 w_k1_AXP_s117
                                 + 0.877444 w_k1_DIS_s117
                                 + 0.92549 w_k1_FDX_s117
                                 + 1.30353 w_k1_IBM_s117
                                 + 1.1077 w_k1_INTC_s117 + 1.07784 w_k1_KO_s117
                                 + 1.16906 w_k1_MAT_s117
                                 + 0.904392 w_k1_MSFT_s117
                                 + 0.994253 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.16037 w_k1_ADM_s117
                                 + 1.09386 w_k1_AXP_s117
                                 + 1.35309 w_k1_DIS_s117
                                 + 1.25462 w_k1_FDX_s117
                                 + 0.794286 w_k1_IBM_s117
                                 + 1.10546 w_k1_INTC_s117
                                 + 0.891989 w_k1_KO_s117
                                 + 0.674636 w_k1_MAT_s117
                                 + 1.04034 w_k1_MSFT_s117
                                 + 1.13027 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.907245 w_k1_ADM_s117
                                 + 1.00553 w_k1_AXP_s117
                                 + 0.902923 w_k1_DIS_s117
                                 + 0.96144 w_k1_FDX_s117
                                 + 0.941241 w_k1_IBM_s117
                                 + 1.10968 w_k1_INTC_s117
                                 + 0.975673 w_k1_KO_s117
                                 + 0.909327 w_k1_MAT_s117
                                 + 0.993368 w_k1_MSFT_s117
                                 + 0.90825 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.25613 w_k1_ADM_s117
                                 + 1.60884 w_k1_AXP_s117
                                 + 1.49775 w_k1_DIS_s117
                                 + 1.40237 w_k1_FDX_s117
                                 + 1.37696 w_k1_IBM_s117
                                 + 1.1733 w_k1_INTC_s117 + 1.12905 w_k1_KO_s117
                                 + 1.35508 w_k1_MAT_s117
                                 + 1.16015 w_k1_MSFT_s117
                                 + 1.15976 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.952009 w_k1_ADM_s117
                                 + 0.808756 w_k1_AXP_s117
                                 + 0.63554 w_k1_DIS_s117
                                 + 0.877207 w_k1_FDX_s117
                                 + 0.592204 w_k1_IBM_s117
                                 + 0.895504 w_k1_INTC_s117
                                 + 0.559722 w_k1_KO_s117
                                 + 0.413896 w_k1_MAT_s117
                                 + 0.900581 w_k1_MSFT_s117
                                 + 0.782926 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.30396 w_k1_ADM_s117
                                 + 1.57933 w_k1_AXP_s117
                                 + 1.21082 w_k1_DIS_s117
                                 + 1.22849 w_k1_FDX_s117
                                 + 1.10169 w_k1_IBM_s117
                                 + 1.05886 w_k1_INTC_s117
                                 + 1.06264 w_k1_KO_s117 + 1.25395 w_k1_MAT_s117
                                 + 1.12023 w_k1_MSFT_s117
                                 + 1.13353 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0205643750110813 w_k1_ADM_s000
                                 + 0.0141470909320841 w_k1_ADM_s001
                                 + 0.0107267755281503 w_k1_ADM_s002
                                 + 0.0331991700125571 w_k1_ADM_s003
                                 + 0.0170538546279075 w_k1_ADM_s004
                                 + 0.0288975341195996 w_k1_ADM_s005
                                 + 0.0482619405841178 w_k1_ADM_s006
                                 + 0.0160794253580504 w_k1_ADM_s007
                                 + 0.0036308564667415 w_k1_ADM_s010
                                 + 0.00494132728457823 w_k1_ADM_s011
                                 + 0.00517886322720649 w_k1_ADM_s012
                                 + 0.014571258481265 w_k1_ADM_s013
                                 + 0.0051474698721374 w_k1_ADM_s014
                                 + 0.00482271198461245 w_k1_ADM_s015
                                 + 0.00589203545323179 w_k1_ADM_s016
                                 + 0.0129043237683106 w_k1_ADM_s017
                                 + 0.0629145299778372 w_k1_ADM_s100
                                 + 0.0435246167575933 w_k1_ADM_s101
                                 + 0.0247903731890527 w_k1_ADM_s102
                                 + 0.038106288942359 w_k1_ADM_s103
                                 + 0.0447644750650877 w_k1_ADM_s104
                                 + 0.0200021531083509 w_k1_ADM_s105
                                 + 0.0789959179638792 w_k1_ADM_s106
                                 + 0.0437001352250641 w_k1_ADM_s107
                                 + 0.0412173737010386 w_k1_ADM_s110
                                 + 0.0369728144405466 w_k1_ADM_s111
                                 + 0.111667923652104 w_k1_ADM_s112
                                 + 0.0570216042175821 w_k1_ADM_s113
                                 + 0.0284415241055358 w_k1_ADM_s114
                                 + 0.0373796420445601 w_k1_ADM_s115
                                 + 0.04991884427164 w_k1_ADM_s116
                                 + 0.0428506675771518 w_k1_ADM_s117
                                 + 0.0219960227839099 w_k1_AXP_s000
                                 + 0.011818433679266 w_k1_AXP_s001
                                 + 0.0104687633645315 w_k1_AXP_s002
                                 + 0.0363729459148743 w_k1_AXP_s003
                                 + 0.0181482235221117 w_k1_AXP_s004
                                 + 0.0286666827416308 w_k1_AXP_s005
                                 + 0.0544249714154995 w_k1_AXP_s006
                                 + 0.0149697725157929 w_k1_AXP_s007
                                 + 0.00343803264653563 w_k1_AXP_s010
                                 + 0.00469270143023817 w_k1_AXP_s011
                                 + 0.00440804409293063 w_k1_AXP_s012
                                 + 0.0134288918831944 w_k1_AXP_s013
                                 + 0.00455234357629461 w_k1_AXP_s014
                                 + 0.00504383479935516 w_k1_AXP_s015
                                 + 0.00515334212212766 w_k1_AXP_s016
                                 + 0.0123423192021177 w_k1_AXP_s017
                                 + 0.0581641252261896 w_k1_AXP_s100
                                 + 0.0416632781296411 w_k1_AXP_s101
                                 + 0.0229166958755215 w_k1_AXP_s102
                                 + 0.047246350113567 w_k1_AXP_s103
                                 + 0.0455435290229683 w_k1_AXP_s104
                                 + 0.0239820470128691 w_k1_AXP_s105
                                 + 0.0878099136678026 w_k1_AXP_s106
                                 + 0.0509217144425515 w_k1_AXP_s107
                                 + 0.0463515903880745 w_k1_AXP_s110
                                 + 0.0433633564331924 w_k1_AXP_s111
                                 + 0.109446794411773 w_k1_AXP_s112
                                 + 0.0496809269004665 w_k1_AXP_s113
                                 + 0.0326459904314295 w_k1_AXP_s114
                                 + 0.0400771568966318 w_k1_AXP_s115
                                 + 0.0459626472178098 w_k1_AXP_s116
                                 + 0.0448754698188977 w_k1_AXP_s117
                                 + 0.0200378955413334 w_k1_DIS_s000
                                 + 0.0128250096646659 w_k1_DIS_s001
                                 + 0.0111993626956447 w_k1_DIS_s002
                                 + 0.0367628990076142 w_k1_DIS_s003
                                 + 0.0179353748178533 w_k1_DIS_s004
                                 + 0.0290983919461542 w_k1_DIS_s005
                                 + 0.051963779792309 w_k1_DIS_s006
                                 + 0.0155638387829643 w_k1_DIS_s007
                                 + 0.00375678842700393 w_k1_DIS_s010
                                 + 0.00462122387015091 w_k1_DIS_s011
                                 + 0.0054868160755918 w_k1_DIS_s012
                                 + 0.013379454795832 w_k1_DIS_s013
                                 + 0.00487647105944207 w_k1_DIS_s014
                                 + 0.00529674575817703 w_k1_DIS_s015
                                 + 0.0051467741898824 w_k1_DIS_s016
                                 + 0.0146419205829165 w_k1_DIS_s017
                                 + 0.059014042267157 w_k1_DIS_s100
                                 + 0.0412601407992851 w_k1_DIS_s101
                                 + 0.0247346262704689 w_k1_DIS_s102
                                 + 0.0427113169605232 w_k1_DIS_s103
                                 + 0.0425236175731507 w_k1_DIS_s104
                                 + 0.025600835757184 w_k1_DIS_s105
                                 + 0.0809234700348117 w_k1_DIS_s106
                                 + 0.0471123343895656 w_k1_DIS_s107
                                 + 0.0502499050185746 w_k1_DIS_s110
                                 + 0.0415629862603793 w_k1_DIS_s111
                                 + 0.118721776672706 w_k1_DIS_s112
                                 + 0.0536547038222894 w_k1_DIS_s113
                                 + 0.032478986337119 w_k1_DIS_s114
                                 + 0.0388640812967508 w_k1_DIS_s115
                                 + 0.0475758395493321 w_k1_DIS_s116
                                 + 0.0405199341181328 w_k1_DIS_s117
                                 + 0.0195369451800932 w_k1_FDX_s000
                                 + 0.0121577223032229 w_k1_FDX_s001
                                 + 0.0111134326554689 w_k1_FDX_s002
                                 + 0.0365544865782652 w_k1_FDX_s003
                                 + 0.0184994083723887 w_k1_FDX_s004
                                 + 0.0293570955389035 w_k1_FDX_s005
                                 + 0.0513258275318517 w_k1_FDX_s006
                                 + 0.0148742004003891 w_k1_FDX_s007
                                 + 0.00374186163313251 w_k1_FDX_s010
                                 + 0.00446826751969389 w_k1_FDX_s011
                                 + 0.00525485669967397 w_k1_FDX_s012
                                 + 0.0130144056760053 w_k1_FDX_s013
                                 + 0.00470234072656486 w_k1_FDX_s014
                                 + 0.00495399736294143 w_k1_FDX_s015
                                 + 0.00504268992189424 w_k1_FDX_s016
                                 + 0.0129071158362601 w_k1_FDX_s017
                                 + 0.0591902575798512 w_k1_FDX_s100
                                 + 0.0390611570265277 w_k1_FDX_s101
                                 + 0.0251353768899519 w_k1_FDX_s102
                                 + 0.0432279386291249 w_k1_FDX_s103
                                 + 0.0433305006966961 w_k1_FDX_s104
                                 + 0.0246077700911286 w_k1_FDX_s105
                                 + 0.0772168845405309 w_k1_FDX_s106
                                 + 0.0468038884659027 w_k1_FDX_s107
                                 + 0.0446161681547755 w_k1_FDX_s110
                                 + 0.039689210739629 w_k1_FDX_s111
                                 + 0.120065191036506 w_k1_FDX_s112
                                 + 0.0508862724390094 w_k1_FDX_s113
                                 + 0.0318209895524952 w_k1_FDX_s114
                                 + 0.0386028369777217 w_k1_FDX_s115
                                 + 0.0452090463647488 w_k1_FDX_s116
                                 + 0.04176533727493 w_k1_FDX_s117
                                 + 0.0195770767393672 w_k1_IBM_s000
                                 + 0.0128610802878039 w_k1_IBM_s001
                                 + 0.0108857406099317 w_k1_IBM_s002
                                 + 0.041884361171572 w_k1_IBM_s003
                                 + 0.0179190459501272 w_k1_IBM_s004
                                 + 0.0274812745886111 w_k1_IBM_s005
                                 + 0.0532214956432717 w_k1_IBM_s006
                                 + 0.0151691030645889 w_k1_IBM_s007
                                 + 0.0034762672131169 w_k1_IBM_s010
                                 + 0.00484590791246618 w_k1_IBM_s011
                                 + 0.00535598684255967 w_k1_IBM_s012
                                 + 0.0125247933177863 w_k1_IBM_s013
                                 + 0.00467617149593723 w_k1_IBM_s014
                                 + 0.0052884046833922 w_k1_IBM_s015
                                 + 0.00507215456571878 w_k1_IBM_s016
                                 + 0.0142463668061003 w_k1_IBM_s017
                                 + 0.0600847860753129 w_k1_IBM_s100
                                 + 0.0402431935573231 w_k1_IBM_s101
                                 + 0.0252860944558088 w_k1_IBM_s102
                                 + 0.0431398219351468 w_k1_IBM_s103
                                 + 0.0427509156798833 w_k1_IBM_s104
                                 + 0.0236948206091187 w_k1_IBM_s105
                                 + 0.0785240238261312 w_k1_IBM_s106
                                 + 0.0454666658882456 w_k1_IBM_s107
                                 + 0.0441426965460372 w_k1_IBM_s110
                                 + 0.0384282292785278 w_k1_IBM_s111
                                 + 0.120981368009005 w_k1_IBM_s112
                                 + 0.0550223879378038 w_k1_IBM_s113
                                 + 0.0310737165406869 w_k1_IBM_s114
                                 + 0.0367915333040391 w_k1_IBM_s115
                                 + 0.0487829745045358 w_k1_IBM_s116
                                 + 0.0418933428315744 w_k1_IBM_s117
                                 + 0.0202451264454989 w_k1_INTC_s000
                                 + 0.0119951250869816 w_k1_INTC_s001
                                 + 0.0106572608139279 w_k1_INTC_s002
                                 + 0.038034779171865 w_k1_INTC_s003
                                 + 0.0146461072966169 w_k1_INTC_s004
                                 + 0.0320597277642219 w_k1_INTC_s005
                                 + 0.0489813296054864 w_k1_INTC_s006
                                 + 0.0150228139181769 w_k1_INTC_s007
                                 + 0.00375386146614141 w_k1_INTC_s010
                                 + 0.00501446739076186 w_k1_INTC_s011
                                 + 0.00556141602948992 w_k1_INTC_s012
                                 + 0.0122844105651667 w_k1_INTC_s013
                                 + 0.00476860859947992 w_k1_INTC_s014
                                 + 0.00509735261837871 w_k1_INTC_s015
                                 + 0.00465951317668642 w_k1_INTC_s016
                                 + 0.0134300847965205 w_k1_INTC_s017
                                 + 0.0620228123720693 w_k1_INTC_s100
                                 + 0.0343958640142346 w_k1_INTC_s101
                                 + 0.0216645946441303 w_k1_INTC_s102
                                 + 0.0442970681557966 w_k1_INTC_s103
                                 + 0.0377218768136312 w_k1_INTC_s104
                                 + 0.0220202005906124 w_k1_INTC_s105
                                 + 0.0789013529159742 w_k1_INTC_s106
                                 + 0.0462072064993491 w_k1_INTC_s107
                                 + 0.0453874061320996 w_k1_INTC_s110
                                 + 0.0430515078303293 w_k1_INTC_s111
                                 + 0.116115367731156 w_k1_INTC_s112
                                 + 0.0453631032183797 w_k1_INTC_s113
                                 + 0.0310223932343617 w_k1_INTC_s114
                                 + 0.0370097250012064 w_k1_INTC_s115
                                 + 0.0442011775888592 w_k1_INTC_s116
                                 + 0.0433849944247096 w_k1_INTC_s117
                                 + 0.0193968217960328 w_k1_KO_s000
                                 + 0.0133688905362768 w_k1_KO_s001
                                 + 0.0103853441603537 w_k1_KO_s002
                                 + 0.0395165927638212 w_k1_KO_s003
                                 + 0.0158621529460291 w_k1_KO_s004
                                 + 0.0267322734278638 w_k1_KO_s005
                                 + 0.0531275721004798 w_k1_KO_s006
                                 + 0.0144425114886954 w_k1_KO_s007
                                 + 0.00359333690909525 w_k1_KO_s010
                                 + 0.00469985783602466 w_k1_KO_s011
                                 + 0.00532083110163764 w_k1_KO_s012
                                 + 0.0117880387431319 w_k1_KO_s013
                                 + 0.00544459636265198 w_k1_KO_s014
                                 + 0.00544546911387233 w_k1_KO_s015
                                 + 0.00527545033379103 w_k1_KO_s016
                                 + 0.0142412843668401 w_k1_KO_s017
                                 + 0.0578073036367978 w_k1_KO_s100
                                 + 0.0392597910656569 w_k1_KO_s101
                                 + 0.0240346642145132 w_k1_KO_s102
                                 + 0.0407489185074826 w_k1_KO_s103
                                 + 0.0407179655685777 w_k1_KO_s104
                                 + 0.0229010245948048 w_k1_KO_s105
                                 + 0.0740795267288421 w_k1_KO_s106
                                 + 0.0455303336310058 w_k1_KO_s107
                                 + 0.0421239226819664 w_k1_KO_s110
                                 + 0.038625722264465 w_k1_KO_s111
                                 + 0.115900369408633 w_k1_KO_s112
                                 + 0.0568617627075645 w_k1_KO_s113
                                 + 0.0302544470090999 w_k1_KO_s114
                                 + 0.0386069023169126 w_k1_KO_s115
                                 + 0.047229317686485 w_k1_KO_s116
                                 + 0.0377750277352513 w_k1_KO_s117
                                 + 0.0181508864392598 w_k1_MAT_s000
                                 + 0.0126520239825087 w_k1_MAT_s001
                                 + 0.0107902776552637 w_k1_MAT_s002
                                 + 0.0472007057881525 w_k1_MAT_s003
                                 + 0.020356754741747 w_k1_MAT_s004
                                 + 0.0274144826527685 w_k1_MAT_s005
                                 + 0.06952888821158 w_k1_MAT_s006
                                 + 0.0135951062821265 w_k1_MAT_s007
                                 + 0.00313796180791919 w_k1_MAT_s010
                                 + 0.00467816560169362 w_k1_MAT_s011
                                 + 0.0055539181423032 w_k1_MAT_s012
                                 + 0.0120003753236312 w_k1_MAT_s013
                                 + 0.0039878531136041 w_k1_MAT_s014
                                 + 0.0066100866836641 w_k1_MAT_s015
                                 + 0.00494865316622569 w_k1_MAT_s016
                                 + 0.0167165736769235 w_k1_MAT_s017
                                 + 0.0582652197093738 w_k1_MAT_s100
                                 + 0.0411768620754007 w_k1_MAT_s101
                                 + 0.0241257441169221 w_k1_MAT_s102
                                 + 0.0428293147984842 w_k1_MAT_s103
                                 + 0.0423094319061218 w_k1_MAT_s104
                                 + 0.0243062538322768 w_k1_MAT_s105
                                 + 0.0707202389651589 w_k1_MAT_s106
                                 + 0.0508075692410349 w_k1_MAT_s107
                                 + 0.0543716218979982 w_k1_MAT_s110
                                 + 0.043958724637053 w_k1_MAT_s111
                                 + 0.119316048266904 w_k1_MAT_s112
                                 + 0.0689434843398115 w_k1_MAT_s113
                                 + 0.0320325918500992 w_k1_MAT_s114
                                 + 0.0372078673500276 w_k1_MAT_s115
                                 + 0.0535871450753193 w_k1_MAT_s116
                                 + 0.0390248555101316 w_k1_MAT_s117
                                 + 0.0186637893340286 w_k1_MSFT_s000
                                 + 0.01205050030098 w_k1_MSFT_s001
                                 + 0.0103706812591126 w_k1_MSFT_s002
                                 + 0.0377974654743763 w_k1_MSFT_s003
                                 + 0.0151572266267352 w_k1_MSFT_s004
                                 + 0.0315084467677195 w_k1_MSFT_s005
                                 + 0.0507187187545872 w_k1_MSFT_s006
                                 + 0.0165352934709449 w_k1_MSFT_s007
                                 + 0.00354646812119758 w_k1_MSFT_s010
                                 + 0.00480555220956165 w_k1_MSFT_s011
                                 + 0.00597777140304704 w_k1_MSFT_s012
                                 + 0.0123136343316891 w_k1_MSFT_s013
                                 + 0.00491831334261063 w_k1_MSFT_s014
                                 + 0.00547203862893408 w_k1_MSFT_s015
                                 + 0.00517480682619806 w_k1_MSFT_s016
                                 + 0.0144018402908914 w_k1_MSFT_s017
                                 + 0.0585408074138746 w_k1_MSFT_s100
                                 + 0.0387152327731661 w_k1_MSFT_s101
                                 + 0.0241746612200036 w_k1_MSFT_s102
                                 + 0.0434964721604912 w_k1_MSFT_s103
                                 + 0.0386702547795463 w_k1_MSFT_s104
                                 + 0.0248946940974784 w_k1_MSFT_s105
                                 + 0.0744675948078262 w_k1_MSFT_s106
                                 + 0.0427579682084324 w_k1_MSFT_s107
                                 + 0.0477447052300973 w_k1_MSFT_s110
                                 + 0.0405046857280569 w_k1_MSFT_s111
                                 + 0.123499144305719 w_k1_MSFT_s112
                                 + 0.0533172953947399 w_k1_MSFT_s113
                                 + 0.0302221129967877 w_k1_MSFT_s114
                                 + 0.036404284303187 w_k1_MSFT_s115
                                 + 0.0454991719208313 w_k1_MSFT_s116
                                 + 0.039204191818368 w_k1_MSFT_s117
                                 + 0.0212901524283199 w_k1_PG_s000
                                 + 0.0126996022653749 w_k1_PG_s001
                                 + 0.0111167685540337 w_k1_PG_s002
                                 + 0.0382076299911388 w_k1_PG_s003
                                 + 0.0172192375025241 w_k1_PG_s004
                                 + 0.0266379152041455 w_k1_PG_s005
                                 + 0.0486856806675465 w_k1_PG_s006
                                 + 0.0145259242735952 w_k1_PG_s007
                                 + 0.00321679653116264 w_k1_PG_s010
                                 + 0.00438125688018602 w_k1_PG_s011
                                 + 0.0054833688683731 w_k1_PG_s012
                                 + 0.0121341321990863 w_k1_PG_s013
                                 + 0.00505572046533388 w_k1_PG_s014
                                 + 0.00552176885317712 w_k1_PG_s015
                                 + 0.00542710809133469 w_k1_PG_s016
                                 + 0.01295757225815 w_k1_PG_s017
                                 + 0.0613094999427507 w_k1_PG_s100
                                 + 0.0374794109206239 w_k1_PG_s101
                                 + 0.0249681956493834 w_k1_PG_s102
                                 + 0.0419897083225791 w_k1_PG_s103
                                 + 0.0417345113858798 w_k1_PG_s104
                                 + 0.0234905671309559 w_k1_PG_s105
                                 + 0.0757951170542931 w_k1_PG_s106
                                 + 0.0422473049399832 w_k1_PG_s107
                                 + 0.0442638914001639 w_k1_PG_s110
                                 + 0.0407750213890896 w_k1_PG_s111
                                 + 0.114504737110504 w_k1_PG_s112
                                 + 0.0530210067922047 w_k1_PG_s113
                                 + 0.0329245664819377 w_k1_PG_s114
                                 + 0.0412336819452114 w_k1_PG_s115
                                 + 0.0470338783441332 w_k1_PG_s116
                                 + 0.0393253860846242 w_k1_PG_s117 >= 1.09
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
