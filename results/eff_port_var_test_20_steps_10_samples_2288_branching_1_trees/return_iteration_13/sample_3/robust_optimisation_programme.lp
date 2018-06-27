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
 w_k1_ADM_s0_return_constraint:  1.11516 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.891206 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.03595 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.924061 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.04628 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.02457 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.35625 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.42236 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.08144 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.851315 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.01375 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.0733 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.19951 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.982397 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.922985 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.34791 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.09871 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.709909 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.960281 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.739187 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.956333 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.37901 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.72174 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.909506 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.831104 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.20653 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.26356 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.29659 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.965028 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.03338 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.05324 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.28516 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.812346 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.816436 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.706298 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.05194 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.07188 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.610088 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.24536 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.879943 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.2221 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.781386 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.266 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.946995 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.29372 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.61409 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.00315 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.921189 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.805486 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.09811 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.09578 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.536403 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.938951 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.43058 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.19199 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.811709 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.24073 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.646102 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.845517 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.28018 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.53122 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.798903 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.580601 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.32888 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.982594 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.16003 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.881833 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.00993 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.987555 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.45837 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.72596 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.741464 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.695433 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.01338 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.884392 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.437685 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.24959 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.852276 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.32484 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.856373 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.24587 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.901943 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.44205 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.94596 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.890422 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.888085 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.754511 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.12599 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.15972 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.628601 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.827949 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.41495 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.00884 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.64714 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.01819 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.542328 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.985713 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.66991 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.8374 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.762364 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.623667 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.39829 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.24669 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.23015 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.978889 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.860719 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.22407 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.57566 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.767933 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.778736 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.571435 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.902341 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.05676 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.466101 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.12512 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.891169 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.19806 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.901381 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.11042 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.971368 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.22387 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.57193 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.903845 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.794759 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.857457 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.899106 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.13332 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.703485 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.973868 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.35845 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.23761 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.737827 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.985876 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.694568 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.919591 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.38462 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.54307 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.853782 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.701921 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.22934 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.2267 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.14387 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.88528 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.88159 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.11477 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.32531 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.738711 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.792663 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.713223 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.00612 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.1303 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.666843 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18726 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.872532 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.23404 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.871366 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.38528 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.843934 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.28666 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.00231 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.08467 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.97471 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.770422 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.53459 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.778309 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.629124 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.710242 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.20935 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.657031 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.09478 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.37986 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.620024 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.878445 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.947551 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.16024 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.90891 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.70365 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.6784 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.743247 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.04316 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.52234 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.17007 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.18778 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.41432 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.39764 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.999533 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.75788 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.926367 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.688117 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.591583 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.03325 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.931735 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.12938 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.01557 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13306 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.977126 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.2125 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.74816 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.01836 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.10894 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.991345 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.34069 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.21932 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.912427 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.02042 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.30608 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.06918 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.719685 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.00567 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.782802 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.03869 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.28304 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.2436 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.862355 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.705415 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.10609 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.00368 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.925046 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.96106 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.904937 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.07848 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.17233 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.718996 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.818922 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.793749 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.891652 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.02624 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.640312 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.14844 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.871824 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.19197 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.883387 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.21812 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.899011 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.24707 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.29693 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.02919 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.33772 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.844579 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.28117 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.01275 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.671675 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.801044 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.47985 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.830391 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.9179 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.22228 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.739805 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.00199 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.25118 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.32934 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.896063 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.703557 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.42561 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.06508 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.11032 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.24684 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.07782 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.16437 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.35179 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.941064 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.843944 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.701209 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.910217 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.894533 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.607973 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.2172 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.916477 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.24862 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.785867 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.34075 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.815684 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.588 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.47698 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.33901 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.84814 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.74176 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.55516 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.00785 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.456391 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.696142 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.59779 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.883423 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.1348 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.50286 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.608502 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.92794 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.930281 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.41509 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.865904 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.67275 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.56187 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.618582 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.23318 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.44768 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.09785 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.971623 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.3199 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.24855 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.64731 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.713559 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.870609 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.571747 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.545226 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.09312 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.914851 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.16345 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.973156 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.17521 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.961889 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.20921 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.52983 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.94598 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.867031 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.853443 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.01471 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.20286 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.814745 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.890624 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.2542 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.05196 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.740077 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.03909 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.722712 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.975205 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.41623 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.46947 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.829829 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.685028 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.13027 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.15972 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.07013 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.947857 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.808808 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.0839 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.23422 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.783325 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.805883 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.700464 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.935092 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.03468 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.702351 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09891 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.948019 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.1219 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.921615 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06446 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.903187 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.2138 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.2791 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.0314 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.02302 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.844815 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.06407 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.980389 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.72676 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.880888 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.18375 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.06485 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.79442 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.08306 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.801728 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.954118 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.1621 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.39878 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.925507 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.828089 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.26033 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.0669 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.10363 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.04739 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.01333 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.09942 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.30719 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.916192 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.934968 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.803592 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.03232 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.996235 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.731573 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.10499989561557 z_k1_s0000
                                 - 0.2979990430548 z_k1_s0001
                                 - 0.26200055236132 z_k1_s0002
                                 - 0.162999796662333 z_k1_s0003
                                 - 0.479999653213817 z_k1_s0004
                                 - 0.389000489132429 z_k1_s0005
                                 - 0.107999807873297 z_k1_s0006
                                 - 0.476998828157594 z_k1_s0007
                                 - 0.126999999407009 z_k1_s0010
                                 - 0.266000099293459 z_k1_s0011
                                 - 0.0830000869825663 z_k1_s0012
                                 - 0.166999422338136 z_k1_s0013
                                 - 0.159999819725299 z_k1_s0014
                                 - 0.256000015560592 z_k1_s0015
                                 - 0.176999506071002 z_k1_s0016
                                 - 0.283999946012505 z_k1_s0017
                                 - 0.227000128035927 z_k1_s0020
                                 - 0.325000063892849 z_k1_s0021
                                 - 0.713000624407994 z_k1_s0022
                                 - 0.508000143890976 z_k1_s0023
                                 - 0.498000304109654 z_k1_s0024
                                 - 0.234000015882852 z_k1_s0025
                                 - 0.805999887374273 z_k1_s0026
                                 - 0.453998750071887 z_k1_s0027
                                 - 0.107000052690517 z_k1_s0030
                                 - 0.0549999858380914 z_k1_s0031
                                 - 0.11800044423837 z_k1_s0032
                                 - 0.0780000533286098 z_k1_s0033
                                 - 0.162000132428664 z_k1_s0034
                                 - 0.0680000729762397 z_k1_s0035
                                 - 0.20100058164322 z_k1_s0036
                                 - 0.149999705285552 z_k1_s0037
                                 - 0.381000428658895 z_k1_s0040
                                 - 0.529000516232317 z_k1_s0041
                                 - 0.548001983395851 z_k1_s0042
                                 - 0.597001213442666 z_k1_s0043
                                 - 0.459998657590758 z_k1_s0044
                                 - 0.660000223502856 z_k1_s0045
                                 - 0.310000367160807 z_k1_s0046
                                 - 0.635000027763844 z_k1_s0047
                                 - 0.212000962928217 z_k1_s0050
                                 - 0.214000984878543 z_k1_s0051
                                 - 0.265001203660271 z_k1_s0052
                                 - 0.778003765354622 z_k1_s0053
                                 - 0.228001138530823 z_k1_s0054
                                 - 0.234001886284954 z_k1_s0055
                                 - 0.175000897798768 z_k1_s0056
                                 - 0.167000809997465 z_k1_s0057
                                 - 0.718000900898548 z_k1_s0060
                                 - 0.619999577446052 z_k1_s0061
                                 - 0.424999777184704 z_k1_s0062
                                 - 0.267999576728619 z_k1_s0063
                                 - 0.287999736878257 z_k1_s0064
                                 - 0.827999088125198 z_k1_s0065
                                 - 0.331999426168356 z_k1_s0066
                                 - 0.664999212583718 z_k1_s0067
                                 - 0.0799999778901858 z_k1_s0070
                                 - 0.0540000004933716 z_k1_s0071
                                 - 0.138999740233003 z_k1_s0072
                                 - 0.0459999938943552 z_k1_s0073
                                 - 0.216000001973486 z_k1_s0074
                                 - 0.180999642727872 z_k1_s0075
                                 - 0.0709999484412979 z_k1_s0076
                                 - 0.0940000334884534 z_k1_s0077
                                 - 0.664002524456262 z_k1_s0100
                                 - 0.47899988036207 z_k1_s0101
                                 - 0.620000433403275 z_k1_s0102
                                 - 0.175000358659564 z_k1_s0103
                                 - 0.520999485543195 z_k1_s0104
                                 - 0.272000699152353 z_k1_s0105
                                 - 0.822003543858018 z_k1_s0106
                                 - 0.204000336513808 z_k1_s0107
                                 - 0.391001122570184 z_k1_s0110
                                 - 0.301000857687004 z_k1_s0111
                                 - 0.22600063695102 z_k1_s0112
                                 - 0.513003117238538 z_k1_s0113
                                 - 0.613000591545447 z_k1_s0114
                                 - 0.75600036381361 z_k1_s0115
                                 - 0.621003773499282 z_k1_s0116
                                 - 0.809002409205555 z_k1_s0117
                                 - 0.595999074924638 z_k1_s0120
                                 - 0.402000847569013 z_k1_s0121
                                 - 0.284998670949441 z_k1_s0122
                                 - 0.23599988083644 z_k1_s0123
                                 - 0.136999898664819 z_k1_s0124
                                 - 0.394000746791575 z_k1_s0125
                                 - 0.584999639105202 z_k1_s0126
                                 - 0.175999968305105 z_k1_s0127
                                 - 1.22800009337507 z_k1_s0130
                                 - 0.671999466243891 z_k1_s0131
                                 - 0.707999268899684 z_k1_s0132
                                 - 0.601998537467729 z_k1_s0133
                                 - 0.434000560908208 z_k1_s0134
                                 - 0.366000198668923 z_k1_s0135
                                 - 0.285000406443208 z_k1_s0136
                                 - 0.430000372835181 z_k1_s0137
                                 - 0.484999470547468 z_k1_s0140
                                 - 1.02700038251217 z_k1_s0141
                                 - 1.20699794826083 z_k1_s0142
                                 - 2.23000064050869 z_k1_s0143
                                 - 0.661998768597892 z_k1_s0144
                                 - 1.17199473448166 z_k1_s0145
                                 - 0.477999608590513 z_k1_s0146
                                 - 0.546999586392861 z_k1_s0147
                                 - 0.178999995263381 z_k1_s0150
                                 - 0.155000071275568 z_k1_s0151
                                 - 0.126000005711097 z_k1_s0152
                                 - 0.21000090901867 z_k1_s0153
                                 - 0.140000006345663 z_k1_s0154
                                 - 0.254000859613012 z_k1_s0155
                                 - 0.129000018697078 z_k1_s0156
                                 - 0.606000464367742 z_k1_s0157
                                 - 0.466002458981596 z_k1_s0160
                                 - 0.493000717879489 z_k1_s0161
                                 - 0.507000936089744 z_k1_s0162
                                 - 0.600001664983904 z_k1_s0163
                                 - 0.409002771629781 z_k1_s0164
                                 - 0.267000942698546 z_k1_s0165
                                 - 0.303000975337344 z_k1_s0166
                                 - 0.318001016961942 z_k1_s0167
                                 - 0.185999302859513 z_k1_s0170
                                 - 0.159999952913206 z_k1_s0171
                                 - 0.172999890986191 z_k1_s0172
                                 - 0.166999811629498 z_k1_s0173
                                 - 0.364000370842237 z_k1_s0174
                                 - 0.396000326344901 z_k1_s0175
                                 - 0.228000559955917 z_k1_s0176
                                 - 0.0799998887566591 z_k1_s0177
                                 - 0.0919999809156863 z_k1_s1000
                                 - 0.32700067563632 z_k1_s1001
                                 - 0.110999984164579 z_k1_s1002
                                 - 0.100000031025755 z_k1_s1003
                                 - 0.129999987413472 z_k1_s1004
                                 - 0.215999633071554 z_k1_s1005
                                 - 0.183000038255129 z_k1_s1006
                                 - 0.164000431906305 z_k1_s1007
                                 - 0.540000862254271 z_k1_s1010
                                 - 1.00399981986808 z_k1_s1011
                                 - 0.272999408935676 z_k1_s1012
                                 - 0.319999444178238 z_k1_s1013
                                 - 0.35499926919322 z_k1_s1014
                                 - 0.370999262897099 z_k1_s1015
                                 - 0.550997687418042 z_k1_s1016
                                 - 0.884999383058719 z_k1_s1017
                                 - 0.33700056456166 z_k1_s1020
                                 - 0.194000176764779 z_k1_s1021
                                 - 0.493000246957791 z_k1_s1022
                                 - 0.129000058966234 z_k1_s1023
                                 - 0.167000124534612 z_k1_s1024
                                 - 0.121000146720324 z_k1_s1025
                                 - 0.411999875867154 z_k1_s1026
                                 - 0.291001015547642 z_k1_s1027
                                 - 0.090000017216531 z_k1_s1030
                                 - 0.322000066463589 z_k1_s1031
                                 - 0.104999983585954 z_k1_s1032
                                 - 0.119000214997405 z_k1_s1033
                                 - 0.0739999873891489 z_k1_s1034
                                 - 0.118000480039433 z_k1_s1035
                                 - 0.0999999947961466 z_k1_s1036
                                 - 0.166999537979584 z_k1_s1037
                                 - 0.223999357305411 z_k1_s1040
                                 - 0.157999519904786 z_k1_s1041
                                 - 0.125999710546588 z_k1_s1042
                                 - 0.147999579480349 z_k1_s1043
                                 - 0.235999977612758 z_k1_s1044
                                 - 0.540997584059733 z_k1_s1045
                                 - 0.567998460902747 z_k1_s1046
                                 - 0.304999912440385 z_k1_s1047
                                 - 0.416000817844156 z_k1_s1050
                                 - 0.269999900427743 z_k1_s1051
                                 - 0.637998881751546 z_k1_s1052
                                 - 0.199999943902031 z_k1_s1053
                                 - 0.26300047693513 z_k1_s1054
                                 - 0.237000098019894 z_k1_s1055
                                 - 0.205999994667088 z_k1_s1056
                                 - 0.153999952036564 z_k1_s1057
                                 - 0.186000347139312 z_k1_s1060
                                 - 0.100000058639404 z_k1_s1061
                                 - 0.323999513371615 z_k1_s1062
                                 - 0.238999745453145 z_k1_s1063
                                 - 0.142999889639333 z_k1_s1064
                                 - 0.0850000185184913 z_k1_s1065
                                 - 0.0720000171603692 z_k1_s1066
                                 - 0.104000010864977 z_k1_s1067
                                 - 0.205000496666135 z_k1_s1070
                                 - 0.308000852540603 z_k1_s1071
                                 - 0.368001372477962 z_k1_s1072
                                 - 0.130000218295358 z_k1_s1073
                                 - 0.521001088264232 z_k1_s1074
                                 - 0.232000606787639 z_k1_s1075
                                 - 0.203000561901761 z_k1_s1076
                                 - 0.510000703958332 z_k1_s1077
                                 - 1.32400222697986 z_k1_s1100
                                 - 0.355000365579858 z_k1_s1101
                                 - 1.03000226962654 z_k1_s1102
                                 - 0.943003345787702 z_k1_s1103
                                 - 0.855998290946481 z_k1_s1104
                                 - 0.405999882716475 z_k1_s1105
                                 - 0.391000385382316 z_k1_s1106
                                 - 0.825999296278164 z_k1_s1107
                                 - 1.21799703935531 z_k1_s1110
                                 - 0.546000027365847 z_k1_s1111
                                 - 0.481000117636557 z_k1_s1112
                                 - 1.06600192399954 z_k1_s1113
                                 - 0.589000058299048 z_k1_s1114
                                 - 0.366999852920674 z_k1_s1115
                                 - 0.54399991934805 z_k1_s1116
                                 - 1.73600210147041 z_k1_s1117
                                 - 0.265000406850096 z_k1_s1120
                                 - 0.334999325924147 z_k1_s1121
                                 - 0.102999978351236 z_k1_s1122
                                 - 0.136999914758461 z_k1_s1123
                                 - 0.0939999096846494 z_k1_s1124
                                 - 0.301000358516563 z_k1_s1125
                                 - 0.292999651479836 z_k1_s1126
                                 - 0.0870000177772443 z_k1_s1127
                                 - 0.742999472078392 z_k1_s1130
                                 - 0.936001442143511 z_k1_s1131
                                 - 0.596998871624561 z_k1_s1132
                                 - 0.251000014638234 z_k1_s1133
                                 - 0.330999796531005 z_k1_s1134
                                 - 0.476999198785406 z_k1_s1135
                                 - 0.241999939325345 z_k1_s1136
                                 - 0.417001359364877 z_k1_s1137
                                 - 0.158000073730328 z_k1_s1140
                                 - 0.373999588395863 z_k1_s1141
                                 - 0.351000266293952 z_k1_s1142
                                 - 0.415999639755822 z_k1_s1143
                                 - 0.27299919333978 z_k1_s1144
                                 - 0.301999826207348 z_k1_s1145
                                 - 0.149999922742722 z_k1_s1146
                                 - 0.257000259964515 z_k1_s1147
                                 - 0.907000017428578 z_k1_s1150
                                 - 0.465000996843377 z_k1_s1151
                                 - 0.405001052322114 z_k1_s1152
                                 - 0.532002742176666 z_k1_s1153
                                 - 0.95700472720826 z_k1_s1154
                                 - 0.771001855344689 z_k1_s1155
                                 - 0.424000844513368 z_k1_s1156
                                 - 0.295000678431935 z_k1_s1157
                                 - 0.736000012431261 z_k1_s1160
                                 - 1.23600129982404 z_k1_s1161
                                 - 0.601001121617464 z_k1_s1162
                                 - 0.605001669805438 z_k1_s1163
                                 - 1.21100067515354 z_k1_s1164
                                 - 0.371000417356611 z_k1_s1165
                                 - 0.254000631583498 z_k1_s1166
                                 - 0.588999477053541 z_k1_s1167
                                 - 0.241000740399701 z_k1_s1170
                                 - 0.162999781070611 z_k1_s1171
                                 - 0.162000109597424 z_k1_s1172
                                 - 0.189000641180726 z_k1_s1173
                                 - 0.319000078727538 z_k1_s1174
                                 - 0.229000468118206 z_k1_s1175
                                 - 0.243999592719135 z_k1_s1176
                                 - 0.0740000040345904 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.12996 w_k1_ADM_s000
                                 + 1.14922 w_k1_AXP_s000
                                 + 0.665952 w_k1_DIS_s000
                                 + 0.960936 w_k1_FDX_s000
                                 + 1.03958 w_k1_IBM_s000
                                 + 1.1014 w_k1_INTC_s000
                                 + 0.878506 w_k1_KO_s000
                                 + 0.890859 w_k1_MAT_s000
                                 + 0.986619 w_k1_MSFT_s000
                                 + 0.983876 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.2466 w_k1_ADM_s000
                                 + 1.29447 w_k1_AXP_s000
                                 + 1.30791 w_k1_DIS_s000
                                 + 1.14413 w_k1_FDX_s000
                                 + 1.30875 w_k1_IBM_s000
                                 + 0.979387 w_k1_INTC_s000
                                 + 1.40661 w_k1_KO_s000 + 1.63083 w_k1_MAT_s000
                                 + 1.09496 w_k1_MSFT_s000
                                 + 1.10418 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.929604 w_k1_ADM_s000
                                 + 0.747025 w_k1_AXP_s000
                                 + 0.587496 w_k1_DIS_s000
                                 + 0.73866 w_k1_FDX_s000 + 1.1073 w_k1_IBM_s000
                                 + 0.939165 w_k1_INTC_s000
                                 + 0.862961 w_k1_KO_s000
                                 + 0.779652 w_k1_MAT_s000
                                 + 0.762025 w_k1_MSFT_s000
                                 + 0.884213 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.731464 w_k1_ADM_s000
                                 + 0.730687 w_k1_AXP_s000
                                 + 0.585275 w_k1_DIS_s000
                                 + 0.828857 w_k1_FDX_s000
                                 + 0.594491 w_k1_IBM_s000
                                 + 0.981769 w_k1_INTC_s000
                                 + 0.779692 w_k1_KO_s000
                                 + 0.588064 w_k1_MAT_s000
                                 + 0.871801 w_k1_MSFT_s000
                                 + 0.766719 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.869681 w_k1_ADM_s000
                                 + 0.974252 w_k1_AXP_s000
                                 + 0.754352 w_k1_DIS_s000
                                 + 0.866453 w_k1_FDX_s000
                                 + 0.863041 w_k1_IBM_s000
                                 + 0.82291 w_k1_INTC_s000
                                 + 0.82678 w_k1_KO_s000
                                 + 0.833788 w_k1_MAT_s000
                                 + 0.913592 w_k1_MSFT_s000
                                 + 0.899446 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.09483 w_k1_ADM_s000
                                 + 0.865901 w_k1_AXP_s000
                                 + 1.13308 w_k1_DIS_s000
                                 + 1.16545 w_k1_FDX_s000
                                 + 0.791148 w_k1_IBM_s000
                                 + 0.927542 w_k1_INTC_s000
                                 + 0.989781 w_k1_KO_s000
                                 + 0.659309 w_k1_MAT_s000
                                 + 1.14864 w_k1_MSFT_s000
                                 + 0.946232 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.05501 w_k1_ADM_s000
                                 + 1.10291 w_k1_AXP_s000
                                 + 1.21404 w_k1_DIS_s000
                                 + 1.10523 w_k1_FDX_s000
                                 + 1.74379 w_k1_IBM_s000
                                 + 1.21735 w_k1_INTC_s000
                                 + 1.51901 w_k1_KO_s000 + 1.30326 w_k1_MAT_s000
                                 + 1.06169 w_k1_MSFT_s000 + 1.128 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.11257 w_k1_ADM_s000
                                 + 1.15512 w_k1_AXP_s000
                                 + 1.13915 w_k1_DIS_s000
                                 + 1.05028 w_k1_FDX_s000 + 1.1158 w_k1_IBM_s000
                                 + 1.16905 w_k1_INTC_s000
                                 + 1.18207 w_k1_KO_s000 + 0.99142 w_k1_MAT_s000
                                 + 1.02005 w_k1_MSFT_s000
                                 + 1.00144 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.57405 w_k1_ADM_s001
                                 + 1.45751 w_k1_AXP_s001
                                 + 1.22341 w_k1_DIS_s001
                                 + 1.19485 w_k1_FDX_s001
                                 + 0.880787 w_k1_IBM_s001
                                 + 1.02433 w_k1_INTC_s001 + 1.221 w_k1_KO_s001
                                 + 0.938206 w_k1_MAT_s001
                                 + 1.13438 w_k1_MSFT_s001
                                 + 1.10479 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.891415 w_k1_ADM_s001
                                 + 0.966927 w_k1_AXP_s001
                                 + 0.865174 w_k1_DIS_s001
                                 + 0.928577 w_k1_FDX_s001
                                 + 1.38226 w_k1_IBM_s001
                                 + 0.78488 w_k1_INTC_s001
                                 + 1.08395 w_k1_KO_s001 + 1.22445 w_k1_MAT_s001
                                 + 0.921137 w_k1_MSFT_s001
                                 + 0.961881 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.2886 w_k1_ADM_s001
                                 + 0.964841 w_k1_AXP_s001
                                 + 0.812075 w_k1_DIS_s001
                                 + 1.09938 w_k1_FDX_s001
                                 + 0.666679 w_k1_IBM_s001
                                 + 0.835119 w_k1_INTC_s001
                                 + 0.856479 w_k1_KO_s001
                                 + 0.742162 w_k1_MAT_s001
                                 + 0.972464 w_k1_MSFT_s001
                                 + 0.979917 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.10645 w_k1_ADM_s001
                                 + 1.25364 w_k1_AXP_s001
                                 + 1.13252 w_k1_DIS_s001
                                 + 1.21758 w_k1_FDX_s001
                                 + 1.21087 w_k1_IBM_s001
                                 + 0.780962 w_k1_INTC_s001
                                 + 1.20166 w_k1_KO_s001 + 0.94415 w_k1_MAT_s001
                                 + 0.929177 w_k1_MSFT_s001
                                 + 1.15587 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.02782 w_k1_ADM_s001
                                 + 0.7785 w_k1_AXP_s001 + 1.07023 w_k1_DIS_s001
                                 + 1.00577 w_k1_FDX_s001
                                 + 0.895322 w_k1_IBM_s001
                                 + 0.709208 w_k1_INTC_s001
                                 + 0.976304 w_k1_KO_s001
                                 + 0.744858 w_k1_MAT_s001
                                 + 1.02571 w_k1_MSFT_s001
                                 + 1.03465 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.994851 w_k1_ADM_s001
                                 + 1.26671 w_k1_AXP_s001
                                 + 1.53929 w_k1_DIS_s001
                                 + 1.15274 w_k1_FDX_s001
                                 + 1.46214 w_k1_IBM_s001
                                 + 0.934917 w_k1_INTC_s001
                                 + 1.44128 w_k1_KO_s001 + 1.41038 w_k1_MAT_s001
                                 + 1.08497 w_k1_MSFT_s001
                                 + 1.10795 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.944763 w_k1_ADM_s001
                                 + 0.943912 w_k1_AXP_s001
                                 + 0.59996 w_k1_DIS_s001
                                 + 0.864046 w_k1_FDX_s001 + 0.731 w_k1_IBM_s001
                                 + 0.839051 w_k1_INTC_s001
                                 + 0.807529 w_k1_KO_s001
                                 + 0.520851 w_k1_MAT_s001
                                 + 0.793278 w_k1_MSFT_s001
                                 + 0.896268 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.888234 w_k1_ADM_s001
                                 + 0.589775 w_k1_AXP_s001
                                 + 0.735687 w_k1_DIS_s001
                                 + 0.824663 w_k1_FDX_s001
                                 + 0.835012 w_k1_IBM_s001
                                 + 0.666799 w_k1_INTC_s001
                                 + 0.77167 w_k1_KO_s001
                                 + 0.572921 w_k1_MAT_s001
                                 + 0.878924 w_k1_MSFT_s001
                                 + 0.867982 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.07206 w_k1_ADM_s002
                                 + 1.07739 w_k1_AXP_s002
                                 + 0.77004 w_k1_DIS_s002
                                 + 0.969199 w_k1_FDX_s002
                                 + 0.786111 w_k1_IBM_s002
                                 + 1.02979 w_k1_INTC_s002
                                 + 0.861592 w_k1_KO_s002
                                 + 0.813163 w_k1_MAT_s002
                                 + 0.877572 w_k1_MSFT_s002
                                 + 0.854098 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.988105 w_k1_ADM_s002
                                 + 1.04386 w_k1_AXP_s002 + 1.2701 w_k1_DIS_s002
                                 + 1.15489 w_k1_FDX_s002
                                 + 1.01384 w_k1_IBM_s002
                                 + 0.974575 w_k1_INTC_s002
                                 + 1.08877 w_k1_KO_s002
                                 + 0.760816 w_k1_MAT_s002
                                 + 1.11103 w_k1_MSFT_s002
                                 + 1.02995 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.24509 w_k1_ADM_s002
                                 + 1.57107 w_k1_AXP_s002
                                 + 1.22531 w_k1_DIS_s002
                                 + 1.30269 w_k1_FDX_s002
                                 + 1.01123 w_k1_IBM_s002
                                 + 1.49448 w_k1_INTC_s002
                                 + 1.12005 w_k1_KO_s002 + 1.1821 w_k1_MAT_s002
                                 + 1.34629 w_k1_MSFT_s002
                                 + 1.13404 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.756446 w_k1_ADM_s002
                                 + 0.818354 w_k1_AXP_s002
                                 + 0.67819 w_k1_DIS_s002
                                 + 0.736735 w_k1_FDX_s002
                                 + 1.05683 w_k1_IBM_s002
                                 + 0.858348 w_k1_INTC_s002
                                 + 0.801233 w_k1_KO_s002
                                 + 1.12306 w_k1_MAT_s002
                                 + 0.851261 w_k1_MSFT_s002
                                 + 0.824576 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.979564 w_k1_ADM_s002
                                 + 0.924651 w_k1_AXP_s002
                                 + 1.03067 w_k1_DIS_s002
                                 + 1.04547 w_k1_FDX_s002
                                 + 0.928119 w_k1_IBM_s002
                                 + 0.880245 w_k1_INTC_s002
                                 + 0.978061 w_k1_KO_s002
                                 + 0.580361 w_k1_MAT_s002
                                 + 1.05641 w_k1_MSFT_s002
                                 + 0.892395 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.906005 w_k1_ADM_s002
                                 + 0.683427 w_k1_AXP_s002
                                 + 1.07329 w_k1_DIS_s002 + 0.9254 w_k1_FDX_s002
                                 + 0.804932 w_k1_IBM_s002
                                 + 1.1799 w_k1_INTC_s002 + 1.01523 w_k1_KO_s002
                                 + 0.522512 w_k1_MAT_s002
                                 + 0.984762 w_k1_MSFT_s002
                                 + 0.883156 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.0006 w_k1_ADM_s002
                                 + 1.16608 w_k1_AXP_s002
                                 + 1.31472 w_k1_DIS_s002
                                 + 1.04525 w_k1_FDX_s002
                                 + 1.43791 w_k1_IBM_s002
                                 + 1.10646 w_k1_INTC_s002
                                 + 1.19467 w_k1_KO_s002 + 1.17663 w_k1_MAT_s002
                                 + 1.0307 w_k1_MSFT_s002
                                 + 0.941257 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.786871 w_k1_ADM_s002
                                 + 0.614877 w_k1_AXP_s002
                                 + 0.703089 w_k1_DIS_s002
                                 + 0.837528 w_k1_FDX_s002
                                 + 0.743648 w_k1_IBM_s002
                                 + 0.861097 w_k1_INTC_s002
                                 + 0.747093 w_k1_KO_s002
                                 + 0.482189 w_k1_MAT_s002
                                 + 0.857408 w_k1_MSFT_s002
                                 + 0.860508 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.28042 w_k1_ADM_s003
                                 + 1.29607 w_k1_AXP_s003
                                 + 1.26396 w_k1_DIS_s003
                                 + 1.16448 w_k1_FDX_s003
                                 + 1.43527 w_k1_IBM_s003
                                 + 1.11243 w_k1_INTC_s003
                                 + 1.13071 w_k1_KO_s003 + 1.39791 w_k1_MAT_s003
                                 + 1.13541 w_k1_MSFT_s003
                                 + 1.13478 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.775789 w_k1_ADM_s003
                                 + 0.753267 w_k1_AXP_s003
                                 + 0.568344 w_k1_DIS_s003
                                 + 0.618086 w_k1_FDX_s003
                                 + 1.00106 w_k1_IBM_s003
                                 + 0.675291 w_k1_INTC_s003
                                 + 0.748104 w_k1_KO_s003
                                 + 0.649617 w_k1_MAT_s003
                                 + 0.597604 w_k1_MSFT_s003
                                 + 0.735096 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.713601 w_k1_ADM_s003
                                 + 0.545915 w_k1_AXP_s003
                                 + 0.752531 w_k1_DIS_s003
                                 + 0.818695 w_k1_FDX_s003
                                 + 0.635145 w_k1_IBM_s003
                                 + 0.626379 w_k1_INTC_s003
                                 + 0.7308 w_k1_KO_s003 + 0.536499 w_k1_MAT_s003
                                 + 0.742478 w_k1_MSFT_s003
                                 + 0.780207 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.41013 w_k1_ADM_s003
                                 + 1.88165 w_k1_AXP_s003
                                 + 1.95448 w_k1_DIS_s003
                                 + 1.45341 w_k1_FDX_s003
                                 + 1.66266 w_k1_IBM_s003
                                 + 0.935626 w_k1_INTC_s003
                                 + 1.47993 w_k1_KO_s003 + 2.20293 w_k1_MAT_s003
                                 + 1.10939 w_k1_MSFT_s003
                                 + 1.38846 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.27855 w_k1_ADM_s003
                                 + 1.3069 w_k1_AXP_s003 + 1.28065 w_k1_DIS_s003
                                 + 1.29811 w_k1_FDX_s003
                                 + 0.985232 w_k1_IBM_s003
                                 + 1.05207 w_k1_INTC_s003
                                 + 1.23712 w_k1_KO_s003 + 1.0641 w_k1_MAT_s003
                                 + 1.1511 w_k1_MSFT_s003 + 1.12035 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.29559 w_k1_ADM_s003
                                 + 1.2055 w_k1_AXP_s003 + 1.23443 w_k1_DIS_s003
                                 + 1.26832 w_k1_FDX_s003
                                 + 0.724236 w_k1_IBM_s003
                                 + 0.831935 w_k1_INTC_s003
                                 + 0.927121 w_k1_KO_s003
                                 + 0.856433 w_k1_MAT_s003
                                 + 1.15482 w_k1_MSFT_s003
                                 + 1.05359 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.27508 w_k1_ADM_s003
                                 + 1.09333 w_k1_AXP_s003
                                 + 1.10938 w_k1_DIS_s003
                                 + 1.06581 w_k1_FDX_s003
                                 + 1.13204 w_k1_IBM_s003
                                 + 0.962094 w_k1_INTC_s003
                                 + 1.14609 w_k1_KO_s003 + 1.44592 w_k1_MAT_s003
                                 + 1.0692 w_k1_MSFT_s003 + 1.02324 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.05172 w_k1_ADM_s003
                                 + 0.888119 w_k1_AXP_s003
                                 + 0.732098 w_k1_DIS_s003
                                 + 0.874951 w_k1_FDX_s003
                                 + 0.82532 w_k1_IBM_s003
                                 + 0.663981 w_k1_INTC_s003
                                 + 0.781713 w_k1_KO_s003
                                 + 0.781255 w_k1_MAT_s003
                                 + 0.72274 w_k1_MSFT_s003
                                 + 0.941817 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.962842 w_k1_ADM_s004
                                 + 0.997443 w_k1_AXP_s004
                                 + 0.858613 w_k1_DIS_s004
                                 + 0.99058 w_k1_FDX_s004
                                 + 0.951394 w_k1_IBM_s004
                                 + 0.850831 w_k1_INTC_s004
                                 + 0.878567 w_k1_KO_s004
                                 + 1.14304 w_k1_MAT_s004
                                 + 0.958221 w_k1_MSFT_s004
                                 + 1.00442 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.989418 w_k1_ADM_s004
                                 + 0.858431 w_k1_AXP_s004
                                 + 0.804753 w_k1_DIS_s004
                                 + 1.02696 w_k1_FDX_s004
                                 + 0.615969 w_k1_IBM_s004
                                 + 0.686154 w_k1_INTC_s004
                                 + 0.833827 w_k1_KO_s004
                                 + 0.735232 w_k1_MAT_s004
                                 + 0.844734 w_k1_MSFT_s004
                                 + 0.964416 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.794094 w_k1_ADM_s004
                                 + 0.939944 w_k1_AXP_s004
                                 + 1.0237 w_k1_DIS_s004 + 0.96531 w_k1_FDX_s004
                                 + 1.10712 w_k1_IBM_s004
                                 + 1.07112 w_k1_INTC_s004
                                 + 1.03725 w_k1_KO_s004 + 1.21694 w_k1_MAT_s004
                                 + 1.04134 w_k1_MSFT_s004
                                 + 1.05379 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.00662 w_k1_ADM_s004
                                 + 1.32268 w_k1_AXP_s004
                                 + 1.38522 w_k1_DIS_s004
                                 + 1.27858 w_k1_FDX_s004
                                 + 1.21801 w_k1_IBM_s004
                                 + 1.13814 w_k1_INTC_s004
                                 + 1.20535 w_k1_KO_s004 + 1.03 w_k1_MAT_s004
                                 + 1.20394 w_k1_MSFT_s004
                                 + 1.22303 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.736226 w_k1_ADM_s004
                                 + 0.710902 w_k1_AXP_s004
                                 + 0.611309 w_k1_DIS_s004
                                 + 0.674995 w_k1_FDX_s004
                                 + 0.86523 w_k1_IBM_s004
                                 + 0.663743 w_k1_INTC_s004
                                 + 0.74188 w_k1_KO_s004
                                 + 0.753679 w_k1_MAT_s004
                                 + 0.713078 w_k1_MSFT_s004
                                 + 0.830944 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.07963 w_k1_ADM_s004
                                 + 1.03335 w_k1_AXP_s004
                                 + 1.23841 w_k1_DIS_s004
                                 + 1.27945 w_k1_FDX_s004
                                 + 0.722906 w_k1_IBM_s004
                                 + 1.11168 w_k1_INTC_s004
                                 + 0.976587 w_k1_KO_s004
                                 + 0.924455 w_k1_MAT_s004
                                 + 1.26061 w_k1_MSFT_s004
                                 + 1.12745 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.824568 w_k1_ADM_s004
                                 + 0.814272 w_k1_AXP_s004
                                 + 0.821046 w_k1_DIS_s004
                                 + 0.834359 w_k1_FDX_s004
                                 + 1.09568 w_k1_IBM_s004
                                 + 0.896664 w_k1_INTC_s004
                                 + 0.944525 w_k1_KO_s004
                                 + 0.87025 w_k1_MAT_s004
                                 + 0.929222 w_k1_MSFT_s004
                                 + 0.978791 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.89276 w_k1_ADM_s004
                                 + 1.04042 w_k1_AXP_s004
                                 + 0.926155 w_k1_DIS_s004
                                 + 0.835298 w_k1_FDX_s004
                                 + 1.61006 w_k1_IBM_s004
                                 + 0.967245 w_k1_INTC_s004
                                 + 1.26766 w_k1_KO_s004 + 1.52505 w_k1_MAT_s004
                                 + 1.09787 w_k1_MSFT_s004
                                 + 1.06014 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.781301 w_k1_ADM_s005
                                 + 0.565102 w_k1_AXP_s005
                                 + 0.702991 w_k1_DIS_s005
                                 + 0.816315 w_k1_FDX_s005
                                 + 0.997479 w_k1_IBM_s005
                                 + 0.70027 w_k1_INTC_s005
                                 + 0.938603 w_k1_KO_s005
                                 + 0.908055 w_k1_MAT_s005
                                 + 0.865086 w_k1_MSFT_s005
                                 + 0.904081 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.25144 w_k1_ADM_s005
                                 + 0.762367 w_k1_AXP_s005
                                 + 1.07696 w_k1_DIS_s005
                                 + 1.01676 w_k1_FDX_s005
                                 + 0.936572 w_k1_IBM_s005
                                 + 0.784668 w_k1_INTC_s005
                                 + 0.930744 w_k1_KO_s005
                                 + 0.850414 w_k1_MAT_s005
                                 + 1.11568 w_k1_MSFT_s005
                                 + 0.906234 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.900591 w_k1_ADM_s005
                                 + 0.747121 w_k1_AXP_s005
                                 + 0.874119 w_k1_DIS_s005
                                 + 0.963367 w_k1_FDX_s005
                                 + 0.856814 w_k1_IBM_s005
                                 + 0.726622 w_k1_INTC_s005
                                 + 1.00039 w_k1_KO_s005
                                 + 0.797863 w_k1_MAT_s005
                                 + 0.963037 w_k1_MSFT_s005
                                 + 0.980029 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.10713 w_k1_ADM_s005
                                 + 1.32509 w_k1_AXP_s005 + 1.3813 w_k1_DIS_s005
                                 + 1.24646 w_k1_FDX_s005
                                 + 1.37158 w_k1_IBM_s005
                                 + 0.846479 w_k1_INTC_s005
                                 + 1.20256 w_k1_KO_s005 + 1.49958 w_k1_MAT_s005
                                 + 1.0371 w_k1_MSFT_s005 + 1.13376 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.10885 w_k1_ADM_s005
                                 + 0.84526 w_k1_AXP_s005
                                 + 0.705468 w_k1_DIS_s005
                                 + 0.904152 w_k1_FDX_s005
                                 + 0.719972 w_k1_IBM_s005
                                 + 0.747167 w_k1_INTC_s005
                                 + 0.695778 w_k1_KO_s005
                                 + 0.548624 w_k1_MAT_s005
                                 + 0.766157 w_k1_MSFT_s005
                                 + 0.922534 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.989737 w_k1_ADM_s005
                                 + 1.23543 w_k1_AXP_s005
                                 + 1.56393 w_k1_DIS_s005
                                 + 1.39065 w_k1_FDX_s005
                                 + 0.905691 w_k1_IBM_s005
                                 + 1.20442 w_k1_INTC_s005
                                 + 1.26644 w_k1_KO_s005 + 1.07163 w_k1_MAT_s005
                                 + 1.24816 w_k1_MSFT_s005
                                 + 1.24062 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.662791 w_k1_ADM_s005
                                 + 0.588285 w_k1_AXP_s005
                                 + 0.514307 w_k1_DIS_s005
                                 + 0.697567 w_k1_FDX_s005
                                 + 0.750432 w_k1_IBM_s005
                                 + 0.692869 w_k1_INTC_s005
                                 + 0.759124 w_k1_KO_s005
                                 + 0.742609 w_k1_MAT_s005
                                 + 0.734969 w_k1_MSFT_s005
                                 + 0.849637 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.32774 w_k1_ADM_s005
                                 + 1.09633 w_k1_AXP_s005
                                 + 0.966419 w_k1_DIS_s005
                                 + 1.14097 w_k1_FDX_s005
                                 + 0.960718 w_k1_IBM_s005
                                 + 0.886714 w_k1_INTC_s005
                                 + 1.0276 w_k1_KO_s005 + 1.13028 w_k1_MAT_s005
                                 + 1.07275 w_k1_MSFT_s005
                                 + 1.18449 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.892197 w_k1_ADM_s006
                                 + 0.697665 w_k1_AXP_s006
                                 + 0.878926 w_k1_DIS_s006
                                 + 0.84607 w_k1_FDX_s006
                                 + 0.968035 w_k1_IBM_s006
                                 + 0.908 w_k1_INTC_s006 + 0.917429 w_k1_KO_s006
                                 + 0.811542 w_k1_MAT_s006
                                 + 0.891115 w_k1_MSFT_s006
                                 + 1.00226 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.769012 w_k1_ADM_s006
                                 + 0.851908 w_k1_AXP_s006
                                 + 0.693137 w_k1_DIS_s006
                                 + 0.843015 w_k1_FDX_s006
                                 + 0.764347 w_k1_IBM_s006
                                 + 0.890677 w_k1_INTC_s006
                                 + 0.714918 w_k1_KO_s006
                                 + 0.676026 w_k1_MAT_s006
                                 + 0.806688 w_k1_MSFT_s006
                                 + 0.89627 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.828041 w_k1_ADM_s006
                                 + 0.671346 w_k1_AXP_s006
                                 + 1.10096 w_k1_DIS_s006
                                 + 1.01817 w_k1_FDX_s006
                                 + 0.992608 w_k1_IBM_s006
                                 + 1.1114 w_k1_INTC_s006 + 1.08796 w_k1_KO_s006
                                 + 0.70639 w_k1_MAT_s006
                                 + 1.12921 w_k1_MSFT_s006 + 1.1195 w_k1_PG_s006
                                 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.9663 w_k1_ADM_s006
                                 + 0.735739 w_k1_AXP_s006
                                 + 0.740777 w_k1_DIS_s006
                                 + 0.721024 w_k1_FDX_s006
                                 + 1.20525 w_k1_IBM_s006
                                 + 0.788315 w_k1_INTC_s006
                                 + 0.934999 w_k1_KO_s006
                                 + 0.951088 w_k1_MAT_s006
                                 + 0.759191 w_k1_MSFT_s006
                                 + 0.977635 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.24371 w_k1_ADM_s006
                                 + 1.51308 w_k1_AXP_s006
                                 + 1.76585 w_k1_DIS_s006
                                 + 1.51544 w_k1_FDX_s006
                                 + 1.29496 w_k1_IBM_s006
                                 + 1.09877 w_k1_INTC_s006
                                 + 1.31914 w_k1_KO_s006 + 1.06258 w_k1_MAT_s006
                                 + 1.10406 w_k1_MSFT_s006
                                 + 1.25902 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.02559 w_k1_ADM_s006
                                 + 1.04694 w_k1_AXP_s006
                                 + 0.936199 w_k1_DIS_s006
                                 + 1.09246 w_k1_FDX_s006
                                 + 0.911205 w_k1_IBM_s006
                                 + 1.16035 w_k1_INTC_s006
                                 + 0.903976 w_k1_KO_s006
                                 + 0.762845 w_k1_MAT_s006
                                 + 1.04215 w_k1_MSFT_s006
                                 + 1.06846 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.04462 w_k1_ADM_s006
                                 + 1.08725 w_k1_AXP_s006
                                 + 1.07352 w_k1_DIS_s006
                                 + 1.02709 w_k1_FDX_s006
                                 + 1.50286 w_k1_IBM_s006
                                 + 1.13984 w_k1_INTC_s006
                                 + 1.35233 w_k1_KO_s006 + 1.68672 w_k1_MAT_s006
                                 + 1.07218 w_k1_MSFT_s006
                                 + 1.19546 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.30219 w_k1_ADM_s006
                                 + 1.21582 w_k1_AXP_s006 + 1.307 w_k1_DIS_s006
                                 + 1.25445 w_k1_FDX_s006 + 1.0682 w_k1_IBM_s006
                                 + 1.14895 w_k1_INTC_s006
                                 + 1.13973 w_k1_KO_s006 + 1.01156 w_k1_MAT_s006
                                 + 1.08979 w_k1_MSFT_s006
                                 + 1.24914 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.905931 w_k1_ADM_s007
                                 + 1.27513 w_k1_AXP_s007
                                 + 0.965623 w_k1_DIS_s007
                                 + 0.8788 w_k1_FDX_s007 + 1.24914 w_k1_IBM_s007
                                 + 0.928648 w_k1_INTC_s007
                                 + 1.2472 w_k1_KO_s007 + 1.87818 w_k1_MAT_s007
                                 + 0.962345 w_k1_MSFT_s007
                                 + 1.20968 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.897289 w_k1_ADM_s007
                                 + 0.889397 w_k1_AXP_s007
                                 + 0.737241 w_k1_DIS_s007
                                 + 0.737037 w_k1_FDX_s007
                                 + 1.76848 w_k1_IBM_s007
                                 + 0.642143 w_k1_INTC_s007
                                 + 1.06831 w_k1_KO_s007 + 2.07919 w_k1_MAT_s007
                                 + 0.627488 w_k1_MSFT_s007
                                 + 1.05948 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.15554 w_k1_ADM_s007
                                 + 1.56328 w_k1_AXP_s007 + 1.5999 w_k1_DIS_s007
                                 + 1.38562 w_k1_FDX_s007
                                 + 0.793494 w_k1_IBM_s007
                                 + 1.20124 w_k1_INTC_s007
                                 + 1.05141 w_k1_KO_s007 + 1.18805 w_k1_MAT_s007
                                 + 1.20747 w_k1_MSFT_s007
                                 + 1.34713 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.809696 w_k1_ADM_s007
                                 + 1.00043 w_k1_AXP_s007
                                 + 0.910829 w_k1_DIS_s007
                                 + 0.83466 w_k1_FDX_s007 + 1.1237 w_k1_IBM_s007
                                 + 0.885368 w_k1_INTC_s007
                                 + 1.16088 w_k1_KO_s007 + 1.15984 w_k1_MAT_s007
                                 + 0.950455 w_k1_MSFT_s007
                                 + 1.0792 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.937186 w_k1_ADM_s007
                                 + 1.16316 w_k1_AXP_s007
                                 + 0.882225 w_k1_DIS_s007
                                 + 1.04391 w_k1_FDX_s007
                                 + 0.761321 w_k1_IBM_s007
                                 + 0.792212 w_k1_INTC_s007
                                 + 0.842886 w_k1_KO_s007
                                 + 1.01089 w_k1_MAT_s007
                                 + 0.875465 w_k1_MSFT_s007
                                 + 1.06088 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.671803 w_k1_ADM_s007
                                 + 0.808502 w_k1_AXP_s007
                                 + 0.641601 w_k1_DIS_s007
                                 + 0.786778 w_k1_FDX_s007
                                 + 0.870709 w_k1_IBM_s007
                                 + 0.843525 w_k1_INTC_s007
                                 + 0.761589 w_k1_KO_s007
                                 + 0.615855 w_k1_MAT_s007
                                 + 0.806843 w_k1_MSFT_s007
                                 + 0.883958 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.00559 w_k1_ADM_s007
                                 + 1.31124 w_k1_AXP_s007
                                 + 1.18114 w_k1_DIS_s007
                                 + 1.10969 w_k1_FDX_s007
                                 + 1.12556 w_k1_IBM_s007
                                 + 1.03886 w_k1_INTC_s007
                                 + 1.26841 w_k1_KO_s007 + 1.68229 w_k1_MAT_s007
                                 + 1.17485 w_k1_MSFT_s007
                                 + 1.25889 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.904939 w_k1_ADM_s007
                                 + 1.25548 w_k1_AXP_s007
                                 + 1.52772 w_k1_DIS_s007
                                 + 1.13857 w_k1_FDX_s007
                                 + 1.59144 w_k1_IBM_s007
                                 + 0.997715 w_k1_INTC_s007
                                 + 1.61393 w_k1_KO_s007 + 1.87515 w_k1_MAT_s007
                                 + 1.22992 w_k1_MSFT_s007
                                 + 1.24678 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.31498 w_k1_ADM_s010
                                 + 1.09828 w_k1_AXP_s010
                                 + 1.29155 w_k1_DIS_s010
                                 + 1.19245 w_k1_FDX_s010
                                 + 1.00034 w_k1_IBM_s010
                                 + 1.00937 w_k1_INTC_s010
                                 + 1.21417 w_k1_KO_s010 + 1.09733 w_k1_MAT_s010
                                 + 1.2734 w_k1_MSFT_s010 + 1.0237 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.999991 w_k1_ADM_s010
                                 + 0.950292 w_k1_AXP_s010
                                 + 1.02584 w_k1_DIS_s010
                                 + 0.995825 w_k1_FDX_s010
                                 + 1.14065 w_k1_IBM_s010
                                 + 1.01208 w_k1_INTC_s010
                                 + 1.13956 w_k1_KO_s010 + 1.44028 w_k1_MAT_s010
                                 + 1.0448 w_k1_MSFT_s010 + 1.17382 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.7795 w_k1_ADM_s010
                                 + 0.689455 w_k1_AXP_s010
                                 + 0.607679 w_k1_DIS_s010
                                 + 0.775514 w_k1_FDX_s010
                                 + 0.794118 w_k1_IBM_s010
                                 + 0.59035 w_k1_INTC_s010
                                 + 0.720331 w_k1_KO_s010
                                 + 0.851042 w_k1_MAT_s010
                                 + 0.756774 w_k1_MSFT_s010
                                 + 0.763796 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.01186 w_k1_ADM_s010
                                 + 0.965179 w_k1_AXP_s010
                                 + 0.956278 w_k1_DIS_s010
                                 + 0.96602 w_k1_FDX_s010
                                 + 1.46889 w_k1_IBM_s010
                                 + 0.712336 w_k1_INTC_s010
                                 + 1.33812 w_k1_KO_s010 + 2.25702 w_k1_MAT_s010
                                 + 1.11323 w_k1_MSFT_s010
                                 + 1.14653 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.02896 w_k1_ADM_s010
                                 + 0.926031 w_k1_AXP_s010
                                 + 0.799885 w_k1_DIS_s010
                                 + 0.867304 w_k1_FDX_s010
                                 + 1.15618 w_k1_IBM_s010
                                 + 0.736263 w_k1_INTC_s010
                                 + 0.872695 w_k1_KO_s010
                                 + 1.28755 w_k1_MAT_s010
                                 + 0.887432 w_k1_MSFT_s010
                                 + 0.933608 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.11719 w_k1_ADM_s010
                                 + 1.41371 w_k1_AXP_s010
                                 + 1.38653 w_k1_DIS_s010
                                 + 1.23658 w_k1_FDX_s010
                                 + 1.56795 w_k1_IBM_s010
                                 + 1.09304 w_k1_INTC_s010
                                 + 1.44186 w_k1_KO_s010 + 1.92171 w_k1_MAT_s010
                                 + 1.27753 w_k1_MSFT_s010
                                 + 1.27496 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.990066 w_k1_ADM_s010
                                 + 0.919968 w_k1_AXP_s010
                                 + 0.860971 w_k1_DIS_s010
                                 + 0.996766 w_k1_FDX_s010
                                 + 0.668598 w_k1_IBM_s010
                                 + 0.790871 w_k1_INTC_s010
                                 + 0.772223 w_k1_KO_s010
                                 + 0.955707 w_k1_MAT_s010
                                 + 0.953517 w_k1_MSFT_s010
                                 + 0.982524 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.16664 w_k1_ADM_s010
                                 + 1.40405 w_k1_AXP_s010
                                 + 1.69379 w_k1_DIS_s010
                                 + 1.36198 w_k1_FDX_s010
                                 + 1.27952 w_k1_IBM_s010
                                 + 1.05832 w_k1_INTC_s010
                                 + 1.30849 w_k1_KO_s010 + 1.37735 w_k1_MAT_s010
                                 + 1.30122 w_k1_MSFT_s010
                                 + 1.23378 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.37747 w_k1_ADM_s011
                                 + 1.79847 w_k1_AXP_s011
                                 + 1.78826 w_k1_DIS_s011
                                 + 1.35377 w_k1_FDX_s011
                                 + 1.41654 w_k1_IBM_s011
                                 + 1.09982 w_k1_INTC_s011
                                 + 1.34245 w_k1_KO_s011 + 1.26599 w_k1_MAT_s011
                                 + 1.28831 w_k1_MSFT_s011 + 1.2609 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.10922 w_k1_ADM_s011
                                 + 1.28525 w_k1_AXP_s011
                                 + 1.06402 w_k1_DIS_s011
                                 + 1.25179 w_k1_FDX_s011
                                 + 0.705956 w_k1_IBM_s011
                                 + 0.997766 w_k1_INTC_s011
                                 + 0.80576 w_k1_KO_s011 + 0.92489 w_k1_MAT_s011
                                 + 1.07455 w_k1_MSFT_s011
                                 + 1.08533 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.580903 w_k1_ADM_s011
                                 + 0.624755 w_k1_AXP_s011
                                 + 0.550434 w_k1_DIS_s011
                                 + 0.670985 w_k1_FDX_s011
                                 + 0.664975 w_k1_IBM_s011
                                 + 0.58829 w_k1_INTC_s011
                                 + 0.536337 w_k1_KO_s011
                                 + 0.644899 w_k1_MAT_s011
                                 + 0.629867 w_k1_MSFT_s011
                                 + 0.751323 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.908249 w_k1_ADM_s011
                                 + 1.03529 w_k1_AXP_s011
                                 + 0.88355 w_k1_DIS_s011
                                 + 0.930534 w_k1_FDX_s011
                                 + 0.690261 w_k1_IBM_s011
                                 + 0.858779 w_k1_INTC_s011
                                 + 0.768197 w_k1_KO_s011
                                 + 0.583023 w_k1_MAT_s011
                                 + 0.944055 w_k1_MSFT_s011
                                 + 0.925132 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.731699 w_k1_ADM_s011
                                 + 0.721139 w_k1_AXP_s011
                                 + 0.754709 w_k1_DIS_s011
                                 + 0.78504 w_k1_FDX_s011
                                 + 0.891482 w_k1_IBM_s011
                                 + 0.807921 w_k1_INTC_s011
                                 + 0.760988 w_k1_KO_s011
                                 + 0.534107 w_k1_MAT_s011
                                 + 0.804348 w_k1_MSFT_s011
                                 + 0.772747 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.965863 w_k1_ADM_s011
                                 + 0.909591 w_k1_AXP_s011
                                 + 1.22999 w_k1_DIS_s011
                                 + 0.947419 w_k1_FDX_s011
                                 + 1.36672 w_k1_IBM_s011
                                 + 0.899558 w_k1_INTC_s011
                                 + 1.22672 w_k1_KO_s011 + 1.00014 w_k1_MAT_s011
                                 + 1.10013 w_k1_MSFT_s011
                                 + 1.01256 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.10049 w_k1_ADM_s011
                                 + 1.53232 w_k1_AXP_s011
                                 + 1.28953 w_k1_DIS_s011
                                 + 1.17736 w_k1_FDX_s011
                                 + 1.09308 w_k1_IBM_s011
                                 + 1.07534 w_k1_INTC_s011
                                 + 1.17023 w_k1_KO_s011 + 1.13224 w_k1_MAT_s011
                                 + 1.14468 w_k1_MSFT_s011
                                 + 1.18781 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.985773 w_k1_ADM_s011
                                 + 1.00447 w_k1_AXP_s011
                                 + 0.892211 w_k1_DIS_s011
                                 + 0.935135 w_k1_FDX_s011
                                 + 1.02589 w_k1_IBM_s011
                                 + 0.951392 w_k1_INTC_s011
                                 + 0.881499 w_k1_KO_s011
                                 + 0.973447 w_k1_MAT_s011
                                 + 0.976096 w_k1_MSFT_s011
                                 + 0.985928 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.17665 w_k1_ADM_s012
                                 + 0.996627 w_k1_AXP_s012
                                 + 1.38266 w_k1_DIS_s012 + 1.1752 w_k1_FDX_s012
                                 + 0.904929 w_k1_IBM_s012
                                 + 1.2054 w_k1_INTC_s012 + 1.09019 w_k1_KO_s012
                                 + 1.01019 w_k1_MAT_s012
                                 + 1.12722 w_k1_MSFT_s012
                                 + 0.923382 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.08557 w_k1_ADM_s012
                                 + 0.886752 w_k1_AXP_s012
                                 + 0.806624 w_k1_DIS_s012
                                 + 0.850022 w_k1_FDX_s012
                                 + 1.16477 w_k1_IBM_s012
                                 + 0.908155 w_k1_INTC_s012
                                 + 0.978205 w_k1_KO_s012
                                 + 0.944771 w_k1_MAT_s012
                                 + 0.911344 w_k1_MSFT_s012
                                 + 0.910362 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.17159 w_k1_ADM_s012
                                 + 1.07076 w_k1_AXP_s012
                                 + 1.11216 w_k1_DIS_s012
                                 + 1.00855 w_k1_FDX_s012
                                 + 1.19501 w_k1_IBM_s012
                                 + 1.11598 w_k1_INTC_s012
                                 + 1.28726 w_k1_KO_s012 + 1.19285 w_k1_MAT_s012
                                 + 1.09208 w_k1_MSFT_s012
                                 + 1.07852 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.758727 w_k1_ADM_s012
                                 + 0.686067 w_k1_AXP_s012
                                 + 0.595148 w_k1_DIS_s012
                                 + 0.714179 w_k1_FDX_s012
                                 + 0.780193 w_k1_IBM_s012
                                 + 0.837499 w_k1_INTC_s012
                                 + 0.753019 w_k1_KO_s012
                                 + 0.84588 w_k1_MAT_s012
                                 + 0.795847 w_k1_MSFT_s012
                                 + 0.799056 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.868856 w_k1_ADM_s012
                                 + 0.792032 w_k1_AXP_s012
                                 + 0.773878 w_k1_DIS_s012
                                 + 0.934382 w_k1_FDX_s012
                                 + 0.541801 w_k1_IBM_s012
                                 + 0.949481 w_k1_INTC_s012
                                 + 0.699865 w_k1_KO_s012
                                 + 0.574134 w_k1_MAT_s012
                                 + 0.957842 w_k1_MSFT_s012
                                 + 0.786205 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.12461 w_k1_ADM_s012
                                 + 1.10563 w_k1_AXP_s012
                                 + 1.52465 w_k1_DIS_s012
                                 + 1.22117 w_k1_FDX_s012
                                 + 1.54292 w_k1_IBM_s012
                                 + 1.11361 w_k1_INTC_s012
                                 + 1.42421 w_k1_KO_s012 + 1.01801 w_k1_MAT_s012
                                 + 1.12141 w_k1_MSFT_s012
                                 + 1.12953 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.00669 w_k1_ADM_s012
                                 + 0.840333 w_k1_AXP_s012
                                 + 0.853254 w_k1_DIS_s012
                                 + 0.972257 w_k1_FDX_s012
                                 + 0.75329 w_k1_IBM_s012
                                 + 0.788404 w_k1_INTC_s012
                                 + 0.930537 w_k1_KO_s012
                                 + 1.01118 w_k1_MAT_s012
                                 + 0.98136 w_k1_MSFT_s012
                                 + 0.930935 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.22988 w_k1_ADM_s012
                                 + 1.28051 w_k1_AXP_s012
                                 + 1.00032 w_k1_DIS_s012 + 1.04 w_k1_FDX_s012
                                 + 1.12631 w_k1_IBM_s012
                                 + 0.914232 w_k1_INTC_s012
                                 + 0.969332 w_k1_KO_s012
                                 + 1.13965 w_k1_MAT_s012
                                 + 0.867037 w_k1_MSFT_s012
                                 + 0.966181 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.987223 w_k1_ADM_s013
                                 + 0.825474 w_k1_AXP_s013
                                 + 0.722972 w_k1_DIS_s013
                                 + 0.845843 w_k1_FDX_s013
                                 + 0.797219 w_k1_IBM_s013
                                 + 0.885422 w_k1_INTC_s013
                                 + 0.794903 w_k1_KO_s013
                                 + 0.828282 w_k1_MAT_s013
                                 + 0.846582 w_k1_MSFT_s013
                                 + 0.948713 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.871608 w_k1_ADM_s013
                                 + 0.781633 w_k1_AXP_s013
                                 + 0.831422 w_k1_DIS_s013
                                 + 0.761166 w_k1_FDX_s013
                                 + 1.13966 w_k1_IBM_s013
                                 + 0.991907 w_k1_INTC_s013
                                 + 0.933929 w_k1_KO_s013
                                 + 1.18559 w_k1_MAT_s013
                                 + 0.85966 w_k1_MSFT_s013
                                 + 1.02256 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.29773 w_k1_ADM_s013
                                 + 1.44782 w_k1_AXP_s013
                                 + 1.29138 w_k1_DIS_s013
                                 + 1.37296 w_k1_FDX_s013
                                 + 1.12761 w_k1_IBM_s013
                                 + 1.59064 w_k1_INTC_s013
                                 + 1.27739 w_k1_KO_s013 + 1.36815 w_k1_MAT_s013
                                 + 1.35895 w_k1_MSFT_s013
                                 + 1.29382 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.24422 w_k1_ADM_s013
                                 + 1.31646 w_k1_AXP_s013
                                 + 1.42979 w_k1_DIS_s013
                                 + 1.03566 w_k1_FDX_s013
                                 + 1.60224 w_k1_IBM_s013
                                 + 1.22529 w_k1_INTC_s013
                                 + 1.55396 w_k1_KO_s013 + 1.88835 w_k1_MAT_s013
                                 + 0.988812 w_k1_MSFT_s013
                                 + 1.34888 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.2305 w_k1_ADM_s013
                                 + 0.909584 w_k1_AXP_s013
                                 + 1.05036 w_k1_DIS_s013
                                 + 0.968769 w_k1_FDX_s013
                                 + 1.03117 w_k1_IBM_s013
                                 + 1.01341 w_k1_INTC_s013
                                 + 1.00205 w_k1_KO_s013
                                 + 0.942532 w_k1_MAT_s013
                                 + 0.969069 w_k1_MSFT_s013
                                 + 1.09425 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.28043 w_k1_ADM_s013
                                 + 1.04969 w_k1_AXP_s013
                                 + 0.921093 w_k1_DIS_s013
                                 + 1.1092 w_k1_FDX_s013
                                 + 0.648972 w_k1_IBM_s013
                                 + 1.15921 w_k1_INTC_s013
                                 + 0.825038 w_k1_KO_s013
                                 + 0.926177 w_k1_MAT_s013
                                 + 1.1635 w_k1_MSFT_s013 + 1.09146 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.21679 w_k1_ADM_s013
                                 + 1.24278 w_k1_AXP_s013
                                 + 1.21238 w_k1_DIS_s013
                                 + 1.17744 w_k1_FDX_s013 + 0.8795 w_k1_IBM_s013
                                 + 1.15753 w_k1_INTC_s013
                                 + 0.918075 w_k1_KO_s013
                                 + 0.949559 w_k1_MAT_s013
                                 + 1.15556 w_k1_MSFT_s013
                                 + 1.15869 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.21722 w_k1_ADM_s013
                                 + 1.01158 w_k1_AXP_s013
                                 + 1.37538 w_k1_DIS_s013
                                 + 1.13022 w_k1_FDX_s013 + 1.03 w_k1_IBM_s013
                                 + 1.15651 w_k1_INTC_s013
                                 + 1.18382 w_k1_KO_s013
                                 + 0.898869 w_k1_MAT_s013
                                 + 1.18034 w_k1_MSFT_s013
                                 + 1.18991 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.921402 w_k1_ADM_s014
                                 + 1.15197 w_k1_AXP_s014
                                 + 0.950953 w_k1_DIS_s014
                                 + 0.937382 w_k1_FDX_s014
                                 + 1.05959 w_k1_IBM_s014
                                 + 1.08894 w_k1_INTC_s014
                                 + 0.919004 w_k1_KO_s014
                                 + 1.09781 w_k1_MAT_s014
                                 + 0.873222 w_k1_MSFT_s014
                                 + 0.900144 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.887569 w_k1_ADM_s014
                                 + 0.684281 w_k1_AXP_s014
                                 + 0.78998 w_k1_DIS_s014
                                 + 0.775774 w_k1_FDX_s014
                                 + 1.14105 w_k1_IBM_s014
                                 + 0.881341 w_k1_INTC_s014
                                 + 0.866301 w_k1_KO_s014
                                 + 0.961601 w_k1_MAT_s014
                                 + 0.79964 w_k1_MSFT_s014
                                 + 0.911289 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.01423 w_k1_ADM_s014
                                 + 0.985827 w_k1_AXP_s014
                                 + 1.14221 w_k1_DIS_s014
                                 + 1.05752 w_k1_FDX_s014
                                 + 1.15433 w_k1_IBM_s014
                                 + 0.986811 w_k1_INTC_s014
                                 + 1.13553 w_k1_KO_s014
                                 + 0.983199 w_k1_MAT_s014
                                 + 0.956746 w_k1_MSFT_s014
                                 + 0.969933 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.951115 w_k1_ADM_s014
                                 + 0.880612 w_k1_AXP_s014
                                 + 0.774253 w_k1_DIS_s014
                                 + 0.92988 w_k1_FDX_s014
                                 + 0.764919 w_k1_IBM_s014
                                 + 0.831534 w_k1_INTC_s014
                                 + 0.669724 w_k1_KO_s014
                                 + 0.856393 w_k1_MAT_s014
                                 + 0.834516 w_k1_MSFT_s014
                                 + 0.812321 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.08969 w_k1_ADM_s014
                                 + 1.20489 w_k1_AXP_s014
                                 + 1.08729 w_k1_DIS_s014
                                 + 1.20121 w_k1_FDX_s014
                                 + 0.838174 w_k1_IBM_s014
                                 + 1.25201 w_k1_INTC_s014
                                 + 0.900024 w_k1_KO_s014
                                 + 0.927299 w_k1_MAT_s014
                                 + 0.990437 w_k1_MSFT_s014
                                 + 1.0518 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.936019 w_k1_ADM_s014
                                 + 1.07623 w_k1_AXP_s014
                                 + 1.11511 w_k1_DIS_s014
                                 + 1.01559 w_k1_FDX_s014
                                 + 1.69366 w_k1_IBM_s014
                                 + 1.1945 w_k1_INTC_s014 + 1.4272 w_k1_KO_s014
                                 + 1.74904 w_k1_MAT_s014
                                 + 1.05328 w_k1_MSFT_s014
                                 + 1.09427 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.57934 w_k1_ADM_s014
                                 + 1.62091 w_k1_AXP_s014 + 1.7113 w_k1_DIS_s014
                                 + 1.39379 w_k1_FDX_s014 + 1.284 w_k1_IBM_s014
                                 + 1.24105 w_k1_INTC_s014
                                 + 1.29314 w_k1_KO_s014 + 1.55498 w_k1_MAT_s014
                                 + 1.24658 w_k1_MSFT_s014
                                 + 1.23772 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.24416 w_k1_ADM_s014
                                 + 1.19068 w_k1_AXP_s014
                                 + 1.30647 w_k1_DIS_s014
                                 + 1.35964 w_k1_FDX_s014
                                 + 0.797017 w_k1_IBM_s014
                                 + 0.988166 w_k1_INTC_s014
                                 + 0.836252 w_k1_KO_s014
                                 + 0.849055 w_k1_MAT_s014
                                 + 1.0574 w_k1_MSFT_s014 + 1.07953 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.16953 w_k1_ADM_s015
                                 + 1.39916 w_k1_AXP_s015
                                 + 1.34016 w_k1_DIS_s015
                                 + 1.32003 w_k1_FDX_s015
                                 + 1.09994 w_k1_IBM_s015
                                 + 1.14492 w_k1_INTC_s015
                                 + 1.08016 w_k1_KO_s015
                                 + 0.961989 w_k1_MAT_s015
                                 + 1.09935 w_k1_MSFT_s015 + 1.1013 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.992963 w_k1_ADM_s015
                                 + 1.71703 w_k1_AXP_s015
                                 + 1.62475 w_k1_DIS_s015
                                 + 1.27555 w_k1_FDX_s015
                                 + 1.55893 w_k1_IBM_s015
                                 + 1.29499 w_k1_INTC_s015
                                 + 1.12697 w_k1_KO_s015 + 1.29038 w_k1_MAT_s015
                                 + 1.22825 w_k1_MSFT_s015
                                 + 1.12598 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.20042 w_k1_ADM_s015
                                 + 1.23696 w_k1_AXP_s015
                                 + 1.16386 w_k1_DIS_s015
                                 + 1.35436 w_k1_FDX_s015
                                 + 0.729607 w_k1_IBM_s015
                                 + 1.19369 w_k1_INTC_s015
                                 + 0.776916 w_k1_KO_s015
                                 + 0.748309 w_k1_MAT_s015
                                 + 1.26392 w_k1_MSFT_s015
                                 + 0.92959 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.13975 w_k1_ADM_s015
                                 + 0.900618 w_k1_AXP_s015
                                 + 0.861651 w_k1_DIS_s015
                                 + 0.976452 w_k1_FDX_s015
                                 + 0.583498 w_k1_IBM_s015
                                 + 1.02662 w_k1_INTC_s015
                                 + 0.662449 w_k1_KO_s015
                                 + 0.53191 w_k1_MAT_s015
                                 + 0.882773 w_k1_MSFT_s015
                                 + 0.889384 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.949663 w_k1_ADM_s015
                                 + 0.938787 w_k1_AXP_s015
                                 + 1.29314 w_k1_DIS_s015
                                 + 0.899051 w_k1_FDX_s015
                                 + 1.45298 w_k1_IBM_s015
                                 + 1.09649 w_k1_INTC_s015
                                 + 1.10026 w_k1_KO_s015 + 1.30295 w_k1_MAT_s015
                                 + 1.04492 w_k1_MSFT_s015
                                 + 1.08245 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.00917 w_k1_ADM_s015
                                 + 1.06613 w_k1_AXP_s015
                                 + 0.903656 w_k1_DIS_s015
                                 + 0.923133 w_k1_FDX_s015
                                 + 1.35522 w_k1_IBM_s015
                                 + 0.999583 w_k1_INTC_s015
                                 + 1.0446 w_k1_KO_s015 + 1.05362 w_k1_MAT_s015
                                 + 0.886535 w_k1_MSFT_s015
                                 + 1.03386 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.09439 w_k1_ADM_s015
                                 + 1.07533 w_k1_AXP_s015
                                 + 1.19989 w_k1_DIS_s015
                                 + 1.13445 w_k1_FDX_s015
                                 + 0.82804 w_k1_IBM_s015
                                 + 0.991198 w_k1_INTC_s015
                                 + 0.879809 w_k1_KO_s015
                                 + 0.666971 w_k1_MAT_s015
                                 + 0.977619 w_k1_MSFT_s015
                                 + 0.971623 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.79723 w_k1_ADM_s015
                                 + 0.717199 w_k1_AXP_s015
                                 + 0.72717 w_k1_DIS_s015
                                 + 0.799957 w_k1_FDX_s015
                                 + 0.933789 w_k1_IBM_s015
                                 + 0.886647 w_k1_INTC_s015
                                 + 0.843111 w_k1_KO_s015
                                 + 0.745787 w_k1_MAT_s015
                                 + 0.846955 w_k1_MSFT_s015
                                 + 0.876008 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.0315 w_k1_ADM_s016
                                 + 0.925206 w_k1_AXP_s016
                                 + 0.84481 w_k1_DIS_s016
                                 + 0.97346 w_k1_FDX_s016
                                 + 0.861095 w_k1_IBM_s016
                                 + 0.780906 w_k1_INTC_s016
                                 + 0.907142 w_k1_KO_s016
                                 + 1.09506 w_k1_MAT_s016
                                 + 0.936877 w_k1_MSFT_s016
                                 + 1.00718 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.971363 w_k1_ADM_s016
                                 + 1.0004 w_k1_AXP_s016 + 1.05817 w_k1_DIS_s016
                                 + 1.19066 w_k1_FDX_s016
                                 + 0.909517 w_k1_IBM_s016
                                 + 0.969876 w_k1_INTC_s016
                                 + 0.81465 w_k1_KO_s016
                                 + 0.894964 w_k1_MAT_s016
                                 + 1.03538 w_k1_MSFT_s016
                                 + 0.946987 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.09421 w_k1_ADM_s016
                                 + 1.27779 w_k1_AXP_s016
                                 + 1.06102 w_k1_DIS_s016
                                 + 1.05883 w_k1_FDX_s016
                                 + 1.43531 w_k1_IBM_s016
                                 + 0.865738 w_k1_INTC_s016
                                 + 1.00212 w_k1_KO_s016 + 1.62901 w_k1_MAT_s016
                                 + 1.01547 w_k1_MSFT_s016
                                 + 1.09453 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.784967 w_k1_ADM_s016
                                 + 0.858039 w_k1_AXP_s016
                                 + 0.916574 w_k1_DIS_s016
                                 + 0.85314 w_k1_FDX_s016 + 1.1632 w_k1_IBM_s016
                                 + 0.912076 w_k1_INTC_s016
                                 + 1.00884 w_k1_KO_s016 + 1.0251 w_k1_MAT_s016
                                 + 0.882628 w_k1_MSFT_s016
                                 + 0.997298 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.14365 w_k1_ADM_s016
                                 + 1.33884 w_k1_AXP_s016
                                 + 1.61286 w_k1_DIS_s016
                                 + 1.31486 w_k1_FDX_s016
                                 + 1.26375 w_k1_IBM_s016
                                 + 1.07267 w_k1_INTC_s016
                                 + 1.20464 w_k1_KO_s016 + 1.36649 w_k1_MAT_s016
                                 + 1.24564 w_k1_MSFT_s016
                                 + 1.18262 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.987499 w_k1_ADM_s016
                                 + 1.26366 w_k1_AXP_s016
                                 + 1.43905 w_k1_DIS_s016
                                 + 1.22581 w_k1_FDX_s016
                                 + 1.25314 w_k1_IBM_s016
                                 + 0.955962 w_k1_INTC_s016 + 1.284 w_k1_KO_s016
                                 + 1.38407 w_k1_MAT_s016
                                 + 1.05174 w_k1_MSFT_s016
                                 + 1.17483 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.735339 w_k1_ADM_s016
                                 + 0.791834 w_k1_AXP_s016
                                 + 0.637471 w_k1_DIS_s016
                                 + 0.774734 w_k1_FDX_s016
                                 + 0.91626 w_k1_IBM_s016
                                 + 0.74439 w_k1_INTC_s016
                                 + 0.761515 w_k1_KO_s016
                                 + 0.66523 w_k1_MAT_s016
                                 + 0.760354 w_k1_MSFT_s016
                                 + 0.826735 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.887894 w_k1_ADM_s016
                                 + 0.719407 w_k1_AXP_s016
                                 + 0.693637 w_k1_DIS_s016
                                 + 0.857804 w_k1_FDX_s016
                                 + 0.710338 w_k1_IBM_s016
                                 + 0.757937 w_k1_INTC_s016
                                 + 0.655198 w_k1_KO_s016
                                 + 0.521063 w_k1_MAT_s016
                                 + 0.795229 w_k1_MSFT_s016
                                 + 0.854006 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.968235 w_k1_ADM_s017
                                 + 1.35731 w_k1_AXP_s017
                                 + 1.22583 w_k1_DIS_s017
                                 + 1.11997 w_k1_FDX_s017
                                 + 1.40715 w_k1_IBM_s017
                                 + 1.1085 w_k1_INTC_s017 + 1.07492 w_k1_KO_s017
                                 + 1.70421 w_k1_MAT_s017
                                 + 1.02202 w_k1_MSFT_s017
                                 + 1.08423 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.04277 w_k1_ADM_s017
                                 + 1.24409 w_k1_AXP_s017
                                 + 1.20575 w_k1_DIS_s017 + 1.1797 w_k1_FDX_s017
                                 + 0.916537 w_k1_IBM_s017
                                 + 1.03498 w_k1_INTC_s017
                                 + 0.96567 w_k1_KO_s017 + 1.14556 w_k1_MAT_s017
                                 + 0.99517 w_k1_MSFT_s017
                                 + 0.983793 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.01716 w_k1_ADM_s017
                                 + 1.11201 w_k1_AXP_s017
                                 + 1.16657 w_k1_DIS_s017
                                 + 1.29963 w_k1_FDX_s017
                                 + 0.670112 w_k1_IBM_s017
                                 + 1.11977 w_k1_INTC_s017
                                 + 0.880898 w_k1_KO_s017
                                 + 1.03568 w_k1_MAT_s017
                                 + 1.04227 w_k1_MSFT_s017
                                 + 1.00918 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.14682 w_k1_ADM_s017
                                 + 1.67368 w_k1_AXP_s017
                                 + 1.75809 w_k1_DIS_s017
                                 + 1.36413 w_k1_FDX_s017
                                 + 1.42747 w_k1_IBM_s017
                                 + 1.2238 w_k1_INTC_s017 + 1.6159 w_k1_KO_s017
                                 + 2.02172 w_k1_MAT_s017
                                 + 1.3343 w_k1_MSFT_s017 + 1.31068 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.761699 w_k1_ADM_s017
                                 + 0.724705 w_k1_AXP_s017
                                 + 0.72106 w_k1_DIS_s017
                                 + 0.819298 w_k1_FDX_s017
                                 + 0.833827 w_k1_IBM_s017
                                 + 0.62134 w_k1_INTC_s017
                                 + 0.812671 w_k1_KO_s017
                                 + 0.695019 w_k1_MAT_s017
                                 + 0.78471 w_k1_MSFT_s017
                                 + 0.904412 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.851266 w_k1_ADM_s017
                                 + 0.838036 w_k1_AXP_s017
                                 + 0.964914 w_k1_DIS_s017
                                 + 0.924445 w_k1_FDX_s017
                                 + 1.30589 w_k1_IBM_s017
                                 + 0.954406 w_k1_INTC_s017
                                 + 1.04818 w_k1_KO_s017 + 1.16695 w_k1_MAT_s017
                                 + 0.849192 w_k1_MSFT_s017
                                 + 0.96228 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.904119 w_k1_ADM_s017
                                 + 0.897209 w_k1_AXP_s017
                                 + 0.882975 w_k1_DIS_s017
                                 + 1.00791 w_k1_FDX_s017
                                 + 0.699047 w_k1_IBM_s017
                                 + 0.842181 w_k1_INTC_s017
                                 + 0.758745 w_k1_KO_s017
                                 + 0.675472 w_k1_MAT_s017
                                 + 0.890634 w_k1_MSFT_s017
                                 + 0.917014 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.24969 w_k1_ADM_s017
                                 + 1.20571 w_k1_AXP_s017
                                 + 1.73903 w_k1_DIS_s017
                                 + 1.34878 w_k1_FDX_s017
                                 + 1.00768 w_k1_IBM_s017
                                 + 1.10981 w_k1_INTC_s017
                                 + 1.20062 w_k1_KO_s017 + 1.33396 w_k1_MAT_s017
                                 + 1.54442 w_k1_MSFT_s017
                                 + 1.30243 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.11578 w_k1_ADM_s100
                                 + 1.23451 w_k1_AXP_s100
                                 + 1.60671 w_k1_DIS_s100
                                 + 1.39299 w_k1_FDX_s100
                                 + 0.856478 w_k1_IBM_s100
                                 + 1.33505 w_k1_INTC_s100
                                 + 1.28038 w_k1_KO_s100
                                 + 0.767837 w_k1_MAT_s100
                                 + 1.29225 w_k1_MSFT_s100
                                 + 1.10884 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.923598 w_k1_ADM_s100
                                 + 0.738628 w_k1_AXP_s100
                                 + 0.742369 w_k1_DIS_s100
                                 + 0.785159 w_k1_FDX_s100
                                 + 0.757484 w_k1_IBM_s100
                                 + 0.814915 w_k1_INTC_s100
                                 + 0.733464 w_k1_KO_s100
                                 + 0.406942 w_k1_MAT_s100
                                 + 0.7341 w_k1_MSFT_s100
                                 + 0.782709 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.29877 w_k1_ADM_s100
                                 + 1.49435 w_k1_AXP_s100
                                 + 1.54877 w_k1_DIS_s100
                                 + 1.25452 w_k1_FDX_s100
                                 + 1.57602 w_k1_IBM_s100
                                 + 1.30132 w_k1_INTC_s100
                                 + 1.55353 w_k1_KO_s100 + 1.28536 w_k1_MAT_s100
                                 + 1.13874 w_k1_MSFT_s100
                                 + 1.14221 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.31119 w_k1_ADM_s100
                                 + 1.47536 w_k1_AXP_s100
                                 + 1.22486 w_k1_DIS_s100
                                 + 1.24305 w_k1_FDX_s100
                                 + 1.15166 w_k1_IBM_s100
                                 + 1.15994 w_k1_INTC_s100
                                 + 1.37335 w_k1_KO_s100 + 1.23471 w_k1_MAT_s100
                                 + 1.09317 w_k1_MSFT_s100
                                 + 1.03706 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.965585 w_k1_ADM_s100
                                 + 1.08064 w_k1_AXP_s100
                                 + 1.21469 w_k1_DIS_s100
                                 + 1.23917 w_k1_FDX_s100
                                 + 0.822889 w_k1_IBM_s100
                                 + 1.10255 w_k1_INTC_s100
                                 + 1.18978 w_k1_KO_s100
                                 + 0.914093 w_k1_MAT_s100
                                 + 1.06455 w_k1_MSFT_s100
                                 + 0.961318 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.784984 w_k1_ADM_s100
                                 + 0.902031 w_k1_AXP_s100
                                 + 1.09884 w_k1_DIS_s100
                                 + 0.906043 w_k1_FDX_s100
                                 + 1.30944 w_k1_IBM_s100
                                 + 0.90499 w_k1_INTC_s100
                                 + 1.24918 w_k1_KO_s100
                                 + 0.992767 w_k1_MAT_s100
                                 + 0.924517 w_k1_MSFT_s100
                                 + 0.911834 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.654405 w_k1_ADM_s100
                                 + 0.93746 w_k1_AXP_s100
                                 + 0.760958 w_k1_DIS_s100
                                 + 0.86569 w_k1_FDX_s100
                                 + 1.17526 w_k1_IBM_s100
                                 + 1.00801 w_k1_INTC_s100
                                 + 0.962691 w_k1_KO_s100
                                 + 0.977761 w_k1_MAT_s100
                                 + 0.788414 w_k1_MSFT_s100
                                 + 0.803859 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.878543 w_k1_ADM_s100
                                 + 1.10615 w_k1_AXP_s100
                                 + 0.998788 w_k1_DIS_s100
                                 + 1.09032 w_k1_FDX_s100
                                 + 0.812516 w_k1_IBM_s100
                                 + 0.976038 w_k1_INTC_s100
                                 + 1.00948 w_k1_KO_s100
                                 + 0.944292 w_k1_MAT_s100
                                 + 0.974159 w_k1_MSFT_s100
                                 + 1.03291 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.954609 w_k1_ADM_s101
                                 + 0.797989 w_k1_AXP_s101
                                 + 0.785873 w_k1_DIS_s101
                                 + 0.933744 w_k1_FDX_s101
                                 + 0.6657 w_k1_IBM_s101
                                 + 0.78392 w_k1_INTC_s101
                                 + 0.750373 w_k1_KO_s101
                                 + 0.608891 w_k1_MAT_s101
                                 + 0.771952 w_k1_MSFT_s101
                                 + 0.840587 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.945625 w_k1_ADM_s101
                                 + 0.967803 w_k1_AXP_s101
                                 + 0.995224 w_k1_DIS_s101
                                 + 0.932337 w_k1_FDX_s101
                                 + 0.970117 w_k1_IBM_s101
                                 + 1.07131 w_k1_INTC_s101
                                 + 1.08793 w_k1_KO_s101 + 1.09272 w_k1_MAT_s101
                                 + 1.0471 w_k1_MSFT_s101 + 1.00259 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.02585 w_k1_ADM_s101
                                 + 1.3102 w_k1_AXP_s101 + 1.87233 w_k1_DIS_s101
                                 + 1.44104 w_k1_FDX_s101
                                 + 1.21257 w_k1_IBM_s101
                                 + 1.0846 w_k1_INTC_s101 + 1.2492 w_k1_KO_s101
                                 + 1.10824 w_k1_MAT_s101
                                 + 1.37358 w_k1_MSFT_s101
                                 + 1.20034 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.868879 w_k1_ADM_s101
                                 + 1.22079 w_k1_AXP_s101
                                 + 0.833762 w_k1_DIS_s101
                                 + 0.890343 w_k1_FDX_s101
                                 + 1.26445 w_k1_IBM_s101
                                 + 0.806443 w_k1_INTC_s101
                                 + 1.03532 w_k1_KO_s101 + 1.52667 w_k1_MAT_s101
                                 + 0.885968 w_k1_MSFT_s101
                                 + 0.992598 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.29361 w_k1_ADM_s101
                                 + 1.16803 w_k1_AXP_s101
                                 + 1.31726 w_k1_DIS_s101
                                 + 1.16618 w_k1_FDX_s101
                                 + 1.31719 w_k1_IBM_s101
                                 + 1.1893 w_k1_INTC_s101 + 1.22777 w_k1_KO_s101
                                 + 1.68444 w_k1_MAT_s101
                                 + 1.2063 w_k1_MSFT_s101 + 1.17252 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.942018 w_k1_ADM_s101
                                 + 1.0221 w_k1_AXP_s101 + 0.88706 w_k1_DIS_s101
                                 + 0.863958 w_k1_FDX_s101
                                 + 1.65016 w_k1_IBM_s101
                                 + 0.975375 w_k1_INTC_s101
                                 + 1.09835 w_k1_KO_s101 + 1.50383 w_k1_MAT_s101
                                 + 0.90863 w_k1_MSFT_s101
                                 + 0.939169 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.906146 w_k1_ADM_s101
                                 + 0.600567 w_k1_AXP_s101
                                 + 0.668845 w_k1_DIS_s101
                                 + 0.692499 w_k1_FDX_s101
                                 + 1.02172 w_k1_IBM_s101
                                 + 0.700734 w_k1_INTC_s101
                                 + 0.861273 w_k1_KO_s101
                                 + 0.720087 w_k1_MAT_s101
                                 + 0.708945 w_k1_MSFT_s101
                                 + 0.804673 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.21217 w_k1_ADM_s101
                                 + 1.24105 w_k1_AXP_s101
                                 + 1.15837 w_k1_DIS_s101
                                 + 1.17509 w_k1_FDX_s101
                                 + 0.909145 w_k1_IBM_s101
                                 + 1.0731 w_k1_INTC_s101 + 1.02616 w_k1_KO_s101
                                 + 0.781762 w_k1_MAT_s101
                                 + 1.18149 w_k1_MSFT_s101 + 1.0495 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.0184 w_k1_ADM_s102
                                 + 1.1222 w_k1_AXP_s102 + 1.16877 w_k1_DIS_s102
                                 + 1.14453 w_k1_FDX_s102 + 1.3519 w_k1_IBM_s102
                                 + 1.16125 w_k1_INTC_s102
                                 + 1.31119 w_k1_KO_s102 + 1.28714 w_k1_MAT_s102
                                 + 1.14177 w_k1_MSFT_s102
                                 + 1.00308 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.868027 w_k1_ADM_s102
                                 + 0.891611 w_k1_AXP_s102
                                 + 0.93461 w_k1_DIS_s102
                                 + 0.88758 w_k1_FDX_s102
                                 + 1.06396 w_k1_IBM_s102
                                 + 1.07778 w_k1_INTC_s102
                                 + 1.07309 w_k1_KO_s102
                                 + 0.879219 w_k1_MAT_s102
                                 + 1.10992 w_k1_MSFT_s102
                                 + 0.998909 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.06189 w_k1_ADM_s102
                                 + 1.13164 w_k1_AXP_s102
                                 + 1.02472 w_k1_DIS_s102
                                 + 1.13751 w_k1_FDX_s102
                                 + 0.837377 w_k1_IBM_s102
                                 + 0.961291 w_k1_INTC_s102
                                 + 1.08327 w_k1_KO_s102 + 1.00563 w_k1_MAT_s102
                                 + 1.06751 w_k1_MSFT_s102
                                 + 0.99975 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.759563 w_k1_ADM_s102
                                 + 0.723192 w_k1_AXP_s102
                                 + 0.465906 w_k1_DIS_s102
                                 + 0.662601 w_k1_FDX_s102
                                 + 1.12778 w_k1_IBM_s102
                                 + 0.782564 w_k1_INTC_s102
                                 + 0.902881 w_k1_KO_s102
                                 + 1.20749 w_k1_MAT_s102
                                 + 0.646392 w_k1_MSFT_s102
                                 + 0.795997 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.781671 w_k1_ADM_s102
                                 + 1.01707 w_k1_AXP_s102
                                 + 0.825607 w_k1_DIS_s102
                                 + 0.799349 w_k1_FDX_s102
                                 + 1.30398 w_k1_IBM_s102
                                 + 0.85582 w_k1_INTC_s102
                                 + 1.03285 w_k1_KO_s102 + 1.09141 w_k1_MAT_s102
                                 + 0.78471 w_k1_MSFT_s102
                                 + 0.883232 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.821404 w_k1_ADM_s102
                                 + 0.997525 w_k1_AXP_s102
                                 + 1.06734 w_k1_DIS_s102
                                 + 0.966036 w_k1_FDX_s102
                                 + 1.86146 w_k1_IBM_s102
                                 + 1.1776 w_k1_INTC_s102 + 1.36913 w_k1_KO_s102
                                 + 1.59336 w_k1_MAT_s102
                                 + 1.05356 w_k1_MSFT_s102
                                 + 0.942853 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.73809 w_k1_ADM_s102
                                 + 0.816305 w_k1_AXP_s102
                                 + 0.625999 w_k1_DIS_s102
                                 + 0.778653 w_k1_FDX_s102
                                 + 0.767541 w_k1_IBM_s102
                                 + 0.805436 w_k1_INTC_s102
                                 + 0.706942 w_k1_KO_s102
                                 + 0.634867 w_k1_MAT_s102
                                 + 0.835743 w_k1_MSFT_s102
                                 + 0.862383 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.40675 w_k1_ADM_s102
                                 + 1.37413 w_k1_AXP_s102
                                 + 1.57454 w_k1_DIS_s102 + 1.3911 w_k1_FDX_s102
                                 + 0.950049 w_k1_IBM_s102
                                 + 1.18796 w_k1_INTC_s102
                                 + 1.32976 w_k1_KO_s102
                                 + 0.829612 w_k1_MAT_s102
                                 + 1.41965 w_k1_MSFT_s102
                                 + 1.13683 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.04178 w_k1_ADM_s103
                                 + 1.31321 w_k1_AXP_s103
                                 + 1.53704 w_k1_DIS_s103
                                 + 1.01409 w_k1_FDX_s103
                                 + 1.77456 w_k1_IBM_s103
                                 + 1.12143 w_k1_INTC_s103
                                 + 1.88342 w_k1_KO_s103 + 2.11746 w_k1_MAT_s103
                                 + 1.18054 w_k1_MSFT_s103
                                 + 1.17527 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.815792 w_k1_ADM_s103
                                 + 0.773281 w_k1_AXP_s103
                                 + 0.794785 w_k1_DIS_s103
                                 + 0.855846 w_k1_FDX_s103
                                 + 0.754629 w_k1_IBM_s103
                                 + 0.910362 w_k1_INTC_s103
                                 + 0.955729 w_k1_KO_s103
                                 + 0.740977 w_k1_MAT_s103
                                 + 0.858161 w_k1_MSFT_s103
                                 + 0.904761 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.36793 w_k1_ADM_s103
                                 + 1.47476 w_k1_AXP_s103
                                 + 1.66408 w_k1_DIS_s103
                                 + 1.31254 w_k1_FDX_s103
                                 + 1.10615 w_k1_IBM_s103
                                 + 1.27098 w_k1_INTC_s103
                                 + 1.29837 w_k1_KO_s103 + 1.31548 w_k1_MAT_s103
                                 + 1.27549 w_k1_MSFT_s103
                                 + 1.23444 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.846371 w_k1_ADM_s103
                                 + 0.813597 w_k1_AXP_s103
                                 + 0.857569 w_k1_DIS_s103
                                 + 0.822025 w_k1_FDX_s103
                                 + 1.26559 w_k1_IBM_s103
                                 + 0.778162 w_k1_INTC_s103
                                 + 1.04224 w_k1_KO_s103 + 1.30742 w_k1_MAT_s103
                                 + 0.697871 w_k1_MSFT_s103
                                 + 0.898848 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.942448 w_k1_ADM_s103
                                 + 0.96294 w_k1_AXP_s103
                                 + 1.01019 w_k1_DIS_s103
                                 + 0.938192 w_k1_FDX_s103
                                 + 1.07486 w_k1_IBM_s103
                                 + 0.840792 w_k1_INTC_s103
                                 + 1.15573 w_k1_KO_s103 + 1.61084 w_k1_MAT_s103
                                 + 0.969562 w_k1_MSFT_s103
                                 + 0.966952 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.01475 w_k1_ADM_s103
                                 + 1.17472 w_k1_AXP_s103 + 1.158 w_k1_DIS_s103
                                 + 1.05687 w_k1_FDX_s103
                                 + 1.02396 w_k1_IBM_s103
                                 + 1.1382 w_k1_INTC_s103 + 1.02116 w_k1_KO_s103
                                 + 0.95736 w_k1_MAT_s103
                                 + 1.11785 w_k1_MSFT_s103
                                 + 1.02808 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.20594 w_k1_ADM_s103
                                 + 1.30899 w_k1_AXP_s103
                                 + 1.16924 w_k1_DIS_s103
                                 + 1.10365 w_k1_FDX_s103
                                 + 1.27379 w_k1_IBM_s103
                                 + 1.02053 w_k1_INTC_s103
                                 + 1.24185 w_k1_KO_s103 + 1.55032 w_k1_MAT_s103
                                 + 1.00664 w_k1_MSFT_s103
                                 + 0.994776 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.2717 w_k1_ADM_s103
                                 + 0.82973 w_k1_AXP_s103
                                 + 1.29789 w_k1_DIS_s103
                                 + 1.04197 w_k1_FDX_s103
                                 + 0.781426 w_k1_IBM_s103
                                 + 1.00984 w_k1_INTC_s103
                                 + 1.14282 w_k1_KO_s103
                                 + 0.744305 w_k1_MAT_s103
                                 + 1.03587 w_k1_MSFT_s103
                                 + 1.05923 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.85475 w_k1_ADM_s104
                                 + 0.867016 w_k1_AXP_s104
                                 + 0.758325 w_k1_DIS_s104
                                 + 0.920663 w_k1_FDX_s104
                                 + 0.780965 w_k1_IBM_s104
                                 + 1.07413 w_k1_INTC_s104
                                 + 0.752413 w_k1_KO_s104
                                 + 0.627717 w_k1_MAT_s104
                                 + 0.883289 w_k1_MSFT_s104
                                 + 0.9484 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.763942 w_k1_ADM_s104
                                 + 0.765612 w_k1_AXP_s104
                                 + 0.745306 w_k1_DIS_s104
                                 + 0.760681 w_k1_FDX_s104
                                 + 1.12527 w_k1_IBM_s104
                                 + 1.0497 w_k1_INTC_s104
                                 + 0.820541 w_k1_KO_s104
                                 + 0.988397 w_k1_MAT_s104
                                 + 0.781891 w_k1_MSFT_s104
                                 + 0.883652 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.867415 w_k1_ADM_s104
                                 + 0.798483 w_k1_AXP_s104
                                 + 1.06515 w_k1_DIS_s104
                                 + 0.975904 w_k1_FDX_s104
                                 + 0.858005 w_k1_IBM_s104
                                 + 1.27324 w_k1_INTC_s104
                                 + 1.00109 w_k1_KO_s104
                                 + 0.836032 w_k1_MAT_s104
                                 + 0.97363 w_k1_MSFT_s104
                                 + 0.97856 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.21885 w_k1_ADM_s104
                                 + 1.27108 w_k1_AXP_s104
                                 + 0.879817 w_k1_DIS_s104
                                 + 1.20047 w_k1_FDX_s104
                                 + 0.747299 w_k1_IBM_s104
                                 + 1.22456 w_k1_INTC_s104
                                 + 0.928358 w_k1_KO_s104
                                 + 1.11626 w_k1_MAT_s104
                                 + 1.02061 w_k1_MSFT_s104 + 1.0883 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.890195 w_k1_ADM_s104
                                 + 0.709992 w_k1_AXP_s104
                                 + 0.562798 w_k1_DIS_s104
                                 + 0.674589 w_k1_FDX_s104
                                 + 0.706174 w_k1_IBM_s104
                                 + 0.701088 w_k1_INTC_s104
                                 + 0.688879 w_k1_KO_s104
                                 + 0.708503 w_k1_MAT_s104
                                 + 0.76717 w_k1_MSFT_s104
                                 + 0.851547 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.19137 w_k1_ADM_s104
                                 + 1.60314 w_k1_AXP_s104
                                 + 1.49099 w_k1_DIS_s104
                                 + 1.34867 w_k1_FDX_s104
                                 + 0.998554 w_k1_IBM_s104
                                 + 1.4725 w_k1_INTC_s104 + 1.08862 w_k1_KO_s104
                                 + 1.72687 w_k1_MAT_s104
                                 + 1.19364 w_k1_MSFT_s104
                                 + 1.02664 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.16667 w_k1_ADM_s104
                                 + 1.22239 w_k1_AXP_s104
                                 + 1.08129 w_k1_DIS_s104
                                 + 0.996249 w_k1_FDX_s104
                                 + 1.20453 w_k1_IBM_s104
                                 + 1.02664 w_k1_INTC_s104
                                 + 1.19706 w_k1_KO_s104 + 1.6048 w_k1_MAT_s104
                                 + 0.979394 w_k1_MSFT_s104
                                 + 1.09398 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.809156 w_k1_ADM_s104
                                 + 0.855296 w_k1_AXP_s104
                                 + 0.756717 w_k1_DIS_s104
                                 + 0.860567 w_k1_FDX_s104
                                 + 0.968647 w_k1_IBM_s104
                                 + 1.06779 w_k1_INTC_s104
                                 + 0.891077 w_k1_KO_s104
                                 + 1.03846 w_k1_MAT_s104
                                 + 0.862543 w_k1_MSFT_s104
                                 + 0.911946 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.14076 w_k1_ADM_s105
                                 + 1.14668 w_k1_AXP_s105
                                 + 1.10991 w_k1_DIS_s105
                                 + 1.24116 w_k1_FDX_s105
                                 + 0.883957 w_k1_IBM_s105
                                 + 1.21714 w_k1_INTC_s105
                                 + 1.13811 w_k1_KO_s105
                                 + 0.888417 w_k1_MAT_s105
                                 + 1.11302 w_k1_MSFT_s105
                                 + 1.09736 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.82669 w_k1_ADM_s105
                                 + 0.753304 w_k1_AXP_s105
                                 + 0.649274 w_k1_DIS_s105
                                 + 0.79975 w_k1_FDX_s105
                                 + 0.745564 w_k1_IBM_s105
                                 + 0.814972 w_k1_INTC_s105
                                 + 0.803718 w_k1_KO_s105
                                 + 0.616412 w_k1_MAT_s105
                                 + 0.786581 w_k1_MSFT_s105
                                 + 0.818359 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.908725 w_k1_ADM_s105
                                 + 1.00323 w_k1_AXP_s105
                                 + 0.870473 w_k1_DIS_s105
                                 + 0.905906 w_k1_FDX_s105
                                 + 1.04983 w_k1_IBM_s105
                                 + 0.974067 w_k1_INTC_s105
                                 + 0.972865 w_k1_KO_s105
                                 + 0.750879 w_k1_MAT_s105
                                 + 0.923069 w_k1_MSFT_s105
                                 + 0.911308 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.05847 w_k1_ADM_s105
                                 + 1.06431 w_k1_AXP_s105
                                 + 0.757414 w_k1_DIS_s105
                                 + 1.0034 w_k1_FDX_s105
                                 + 0.692234 w_k1_IBM_s105
                                 + 1.03093 w_k1_INTC_s105
                                 + 0.727321 w_k1_KO_s105
                                 + 0.762163 w_k1_MAT_s105
                                 + 0.857511 w_k1_MSFT_s105
                                 + 0.962656 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.964262 w_k1_ADM_s105
                                 + 1.33693 w_k1_AXP_s105
                                 + 1.46763 w_k1_DIS_s105
                                 + 1.33223 w_k1_FDX_s105 + 1.0513 w_k1_IBM_s105
                                 + 1.50275 w_k1_INTC_s105
                                 + 1.28627 w_k1_KO_s105 + 1.19974 w_k1_MAT_s105
                                 + 1.25789 w_k1_MSFT_s105
                                 + 1.25279 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.825939 w_k1_ADM_s105
                                 + 1.12695 w_k1_AXP_s105
                                 + 1.03174 w_k1_DIS_s105
                                 + 0.947226 w_k1_FDX_s105
                                 + 1.42471 w_k1_IBM_s105
                                 + 1.33492 w_k1_INTC_s105
                                 + 1.21728 w_k1_KO_s105 + 1.49001 w_k1_MAT_s105
                                 + 1.06176 w_k1_MSFT_s105
                                 + 1.09279 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.855874 w_k1_ADM_s105
                                 + 0.771607 w_k1_AXP_s105
                                 + 0.975415 w_k1_DIS_s105
                                 + 0.959098 w_k1_FDX_s105
                                 + 0.818526 w_k1_IBM_s105
                                 + 1.40183 w_k1_INTC_s105 + 1.0026 w_k1_KO_s105
                                 + 0.529988 w_k1_MAT_s105
                                 + 1.09582 w_k1_MSFT_s105
                                 + 0.993658 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.36157 w_k1_ADM_s105
                                 + 1.4317 w_k1_AXP_s105 + 1.59986 w_k1_DIS_s105
                                 + 1.34947 w_k1_FDX_s105
                                 + 1.62358 w_k1_IBM_s105
                                 + 1.64194 w_k1_INTC_s105
                                 + 1.67717 w_k1_KO_s105 + 1.54456 w_k1_MAT_s105
                                 + 1.3808 w_k1_MSFT_s105 + 1.36224 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.941273 w_k1_ADM_s106
                                 + 0.804126 w_k1_AXP_s106
                                 + 0.836658 w_k1_DIS_s106
                                 + 0.937034 w_k1_FDX_s106
                                 + 0.697761 w_k1_IBM_s106
                                 + 0.702847 w_k1_INTC_s106
                                 + 0.747465 w_k1_KO_s106
                                 + 0.534037 w_k1_MAT_s106
                                 + 0.722736 w_k1_MSFT_s106
                                 + 0.817462 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.12768 w_k1_ADM_s106
                                 + 1.04895 w_k1_AXP_s106
                                 + 1.24121 w_k1_DIS_s106
                                 + 1.13194 w_k1_FDX_s106
                                 + 0.93547 w_k1_IBM_s106
                                 + 1.15163 w_k1_INTC_s106 + 1.1297 w_k1_KO_s106
                                 + 1.05597 w_k1_MAT_s106
                                 + 1.23768 w_k1_MSFT_s106
                                 + 1.07283 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.02709 w_k1_ADM_s106
                                 + 0.71007 w_k1_AXP_s106
                                 + 0.917143 w_k1_DIS_s106
                                 + 0.883832 w_k1_FDX_s106
                                 + 0.936925 w_k1_IBM_s106
                                 + 0.819556 w_k1_INTC_s106
                                 + 0.956139 w_k1_KO_s106
                                 + 0.692628 w_k1_MAT_s106
                                 + 0.877249 w_k1_MSFT_s106
                                 + 0.997345 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.16301 w_k1_ADM_s106
                                 + 1.24944 w_k1_AXP_s106
                                 + 1.29169 w_k1_DIS_s106 + 1.104 w_k1_FDX_s106
                                 + 1.24229 w_k1_IBM_s106
                                 + 1.07996 w_k1_INTC_s106
                                 + 1.35756 w_k1_KO_s106 + 1.25169 w_k1_MAT_s106
                                 + 1.104 w_k1_MSFT_s106 + 1.12591 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.775659 w_k1_ADM_s106
                                 + 0.896081 w_k1_AXP_s106
                                 + 0.919304 w_k1_DIS_s106
                                 + 0.812882 w_k1_FDX_s106
                                 + 1.54358 w_k1_IBM_s106
                                 + 0.779331 w_k1_INTC_s106
                                 + 1.11241 w_k1_KO_s106 + 1.51198 w_k1_MAT_s106
                                 + 0.780506 w_k1_MSFT_s106
                                 + 0.875308 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.35614 w_k1_ADM_s106
                                 + 1.4839 w_k1_AXP_s106 + 1.32182 w_k1_DIS_s106
                                 + 1.33176 w_k1_FDX_s106
                                 + 0.953382 w_k1_IBM_s106
                                 + 1.01181 w_k1_INTC_s106
                                 + 1.09094 w_k1_KO_s106
                                 + 0.942166 w_k1_MAT_s106
                                 + 1.00081 w_k1_MSFT_s106
                                 + 1.08251 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.33508 w_k1_ADM_s106
                                 + 1.55668 w_k1_AXP_s106
                                 + 1.72407 w_k1_DIS_s106
                                 + 1.34409 w_k1_FDX_s106
                                 + 1.15578 w_k1_IBM_s106
                                 + 1.15497 w_k1_INTC_s106
                                 + 1.27691 w_k1_KO_s106
                                 + 0.820207 w_k1_MAT_s106
                                 + 1.07539 w_k1_MSFT_s106
                                 + 1.08874 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.999448 w_k1_ADM_s106
                                 + 0.951514 w_k1_AXP_s106
                                 + 1.16746 w_k1_DIS_s106
                                 + 1.16292 w_k1_FDX_s106
                                 + 0.682888 w_k1_IBM_s106
                                 + 1.15312 w_k1_INTC_s106
                                 + 0.795108 w_k1_KO_s106
                                 + 0.481176 w_k1_MAT_s106
                                 + 1.03269 w_k1_MSFT_s106
                                 + 0.856292 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.02529 w_k1_ADM_s107
                                 + 0.993095 w_k1_AXP_s107
                                 + 0.962112 w_k1_DIS_s107
                                 + 1.04918 w_k1_FDX_s107
                                 + 0.733013 w_k1_IBM_s107
                                 + 0.951474 w_k1_INTC_s107
                                 + 0.846665 w_k1_KO_s107
                                 + 1.30327 w_k1_MAT_s107
                                 + 0.990915 w_k1_MSFT_s107 + 1.032 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.14345 w_k1_ADM_s107
                                 + 1.13785 w_k1_AXP_s107
                                 + 1.57776 w_k1_DIS_s107
                                 + 1.22156 w_k1_FDX_s107
                                 + 1.20696 w_k1_IBM_s107
                                 + 1.15898 w_k1_INTC_s107 + 1.3976 w_k1_KO_s107
                                 + 1.20102 w_k1_MAT_s107
                                 + 1.31761 w_k1_MSFT_s107
                                 + 1.09792 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.824486 w_k1_ADM_s107
                                 + 1.08295 w_k1_AXP_s107
                                 + 1.01815 w_k1_DIS_s107
                                 + 1.04216 w_k1_FDX_s107
                                 + 1.45446 w_k1_IBM_s107
                                 + 1.0782 w_k1_INTC_s107 + 1.18269 w_k1_KO_s107
                                 + 1.77426 w_k1_MAT_s107
                                 + 1.15763 w_k1_MSFT_s107
                                 + 1.17331 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.914295 w_k1_ADM_s107
                                 + 0.782101 w_k1_AXP_s107
                                 + 1.28916 w_k1_DIS_s107
                                 + 0.986568 w_k1_FDX_s107
                                 + 0.956961 w_k1_IBM_s107
                                 + 0.968489 w_k1_INTC_s107
                                 + 1.14762 w_k1_KO_s107
                                 + 0.864494 w_k1_MAT_s107
                                 + 1.20362 w_k1_MSFT_s107
                                 + 1.09662 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.750756 w_k1_ADM_s107
                                 + 0.728564 w_k1_AXP_s107
                                 + 0.794555 w_k1_DIS_s107
                                 + 0.912438 w_k1_FDX_s107
                                 + 0.753471 w_k1_IBM_s107
                                 + 0.971159 w_k1_INTC_s107
                                 + 0.734302 w_k1_KO_s107
                                 + 0.638947 w_k1_MAT_s107
                                 + 1.05612 w_k1_MSFT_s107
                                 + 0.881011 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.27151 w_k1_ADM_s107
                                 + 1.13702 w_k1_AXP_s107 + 1.2338 w_k1_DIS_s107
                                 + 1.17637 w_k1_FDX_s107
                                 + 0.959225 w_k1_IBM_s107
                                 + 0.997415 w_k1_INTC_s107
                                 + 1.02049 w_k1_KO_s107 + 1.28105 w_k1_MAT_s107
                                 + 1.15013 w_k1_MSFT_s107
                                 + 0.99487 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.947246 w_k1_ADM_s107
                                 + 1.32182 w_k1_AXP_s107
                                 + 1.69601 w_k1_DIS_s107
                                 + 1.41518 w_k1_FDX_s107
                                 + 0.98511 w_k1_IBM_s107
                                 + 1.30833 w_k1_INTC_s107
                                 + 1.16151 w_k1_KO_s107 + 1.24948 w_k1_MAT_s107
                                 + 1.56685 w_k1_MSFT_s107
                                 + 1.02734 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.6512 w_k1_ADM_s107
                                 + 0.848141 w_k1_AXP_s107
                                 + 0.751147 w_k1_DIS_s107
                                 + 0.846568 w_k1_FDX_s107
                                 + 1.02351 w_k1_IBM_s107
                                 + 0.976946 w_k1_INTC_s107
                                 + 0.9554 w_k1_KO_s107 + 1.18468 w_k1_MAT_s107
                                 + 0.908856 w_k1_MSFT_s107
                                 + 1.01297 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.85389 w_k1_ADM_s110
                                 + 0.804559 w_k1_AXP_s110
                                 + 0.640587 w_k1_DIS_s110
                                 + 0.738016 w_k1_FDX_s110
                                 + 0.800537 w_k1_IBM_s110
                                 + 0.921789 w_k1_INTC_s110
                                 + 0.890912 w_k1_KO_s110
                                 + 0.991209 w_k1_MAT_s110
                                 + 0.81917 w_k1_MSFT_s110
                                 + 0.869154 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.763542 w_k1_ADM_s110
                                 + 0.905246 w_k1_AXP_s110
                                 + 0.776332 w_k1_DIS_s110
                                 + 0.743637 w_k1_FDX_s110
                                 + 1.4721 w_k1_IBM_s110
                                 + 0.744566 w_k1_INTC_s110
                                 + 0.949174 w_k1_KO_s110
                                 + 1.58551 w_k1_MAT_s110
                                 + 0.808933 w_k1_MSFT_s110
                                 + 0.871355 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.888726 w_k1_ADM_s110
                                 + 1.07524 w_k1_AXP_s110
                                 + 0.780088 w_k1_DIS_s110
                                 + 0.855944 w_k1_FDX_s110
                                 + 1.13576 w_k1_IBM_s110
                                 + 0.957742 w_k1_INTC_s110
                                 + 0.972187 w_k1_KO_s110 + 1.4112 w_k1_MAT_s110
                                 + 0.86674 w_k1_MSFT_s110
                                 + 0.913444 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.13191 w_k1_ADM_s110
                                 + 1.37819 w_k1_AXP_s110 + 1.7435 w_k1_DIS_s110
                                 + 1.42207 w_k1_FDX_s110 + 1.0089 w_k1_IBM_s110
                                 + 1.58663 w_k1_INTC_s110
                                 + 1.38618 w_k1_KO_s110 + 1.15492 w_k1_MAT_s110
                                 + 1.43089 w_k1_MSFT_s110
                                 + 1.19051 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.986409 w_k1_ADM_s110
                                 + 1.03113 w_k1_AXP_s110
                                 + 1.08924 w_k1_DIS_s110
                                 + 1.05651 w_k1_FDX_s110
                                 + 0.97057 w_k1_IBM_s110
                                 + 1.08894 w_k1_INTC_s110
                                 + 0.976632 w_k1_KO_s110
                                 + 0.93983 w_k1_MAT_s110
                                 + 1.11148 w_k1_MSFT_s110
                                 + 1.03732 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.0108 w_k1_ADM_s110
                                 + 0.9633 w_k1_AXP_s110 + 1.15988 w_k1_DIS_s110
                                 + 0.959114 w_k1_FDX_s110
                                 + 1.22124 w_k1_IBM_s110
                                 + 1.16067 w_k1_INTC_s110 + 1.1881 w_k1_KO_s110
                                 + 1.38493 w_k1_MAT_s110
                                 + 1.05284 w_k1_MSFT_s110
                                 + 1.01734 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.11826 w_k1_ADM_s110
                                 + 1.42049 w_k1_AXP_s110
                                 + 1.40218 w_k1_DIS_s110
                                 + 1.10541 w_k1_FDX_s110
                                 + 1.79577 w_k1_IBM_s110
                                 + 1.37094 w_k1_INTC_s110
                                 + 1.54411 w_k1_KO_s110 + 1.93916 w_k1_MAT_s110
                                 + 1.28999 w_k1_MSFT_s110
                                 + 1.31587 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.29351 w_k1_ADM_s110
                                 + 1.17129 w_k1_AXP_s110
                                 + 1.11752 w_k1_DIS_s110
                                 + 1.12574 w_k1_FDX_s110
                                 + 0.858673 w_k1_IBM_s110
                                 + 1.1489 w_k1_INTC_s110 + 1.04387 w_k1_KO_s110
                                 + 1.03915 w_k1_MAT_s110
                                 + 1.17769 w_k1_MSFT_s110 + 1.053 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.1482 w_k1_ADM_s111
                                 + 1.23035 w_k1_AXP_s111
                                 + 0.948724 w_k1_DIS_s111 + 1.101 w_k1_FDX_s111
                                 + 1.01469 w_k1_IBM_s111
                                 + 1.34964 w_k1_INTC_s111
                                 + 0.931939 w_k1_KO_s111
                                 + 0.892532 w_k1_MAT_s111
                                 + 0.967786 w_k1_MSFT_s111
                                 + 1.00252 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.837696 w_k1_ADM_s111
                                 + 0.894304 w_k1_AXP_s111
                                 + 0.488425 w_k1_DIS_s111
                                 + 0.73399 w_k1_FDX_s111
                                 + 0.700009 w_k1_IBM_s111
                                 + 0.806814 w_k1_INTC_s111
                                 + 0.676568 w_k1_KO_s111
                                 + 0.974698 w_k1_MAT_s111
                                 + 0.758532 w_k1_MSFT_s111
                                 + 0.767646 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.913188 w_k1_ADM_s111
                                 + 0.791253 w_k1_AXP_s111
                                 + 0.787657 w_k1_DIS_s111
                                 + 0.947234 w_k1_FDX_s111
                                 + 0.597828 w_k1_IBM_s111
                                 + 0.910584 w_k1_INTC_s111
                                 + 0.734371 w_k1_KO_s111
                                 + 0.785677 w_k1_MAT_s111
                                 + 0.869855 w_k1_MSFT_s111
                                 + 0.828599 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.987044 w_k1_ADM_s111
                                 + 0.998201 w_k1_AXP_s111
                                 + 1.20768 w_k1_DIS_s111
                                 + 1.10711 w_k1_FDX_s111
                                 + 1.09577 w_k1_IBM_s111
                                 + 1.27488 w_k1_INTC_s111
                                 + 1.14038 w_k1_KO_s111 + 1.18802 w_k1_MAT_s111
                                 + 1.15149 w_k1_MSFT_s111
                                 + 1.08648 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.36671 w_k1_ADM_s111
                                 + 1.4 w_k1_AXP_s111 + 1.50409 w_k1_DIS_s111
                                 + 1.46472 w_k1_FDX_s111
                                 + 0.748097 w_k1_IBM_s111
                                 + 1.19098 w_k1_INTC_s111
                                 + 1.15003 w_k1_KO_s111 + 1.23648 w_k1_MAT_s111
                                 + 1.30078 w_k1_MSFT_s111
                                 + 1.05927 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.956152 w_k1_ADM_s111
                                 + 0.807972 w_k1_AXP_s111
                                 + 0.763881 w_k1_DIS_s111
                                 + 0.856035 w_k1_FDX_s111
                                 + 0.920235 w_k1_IBM_s111
                                 + 0.756632 w_k1_INTC_s111
                                 + 0.805933 w_k1_KO_s111
                                 + 1.14644 w_k1_MAT_s111
                                 + 0.890167 w_k1_MSFT_s111
                                 + 0.882678 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.17865 w_k1_ADM_s111
                                 + 1.55942 w_k1_AXP_s111
                                 + 1.79662 w_k1_DIS_s111
                                 + 1.46906 w_k1_FDX_s111
                                 + 1.35022 w_k1_IBM_s111
                                 + 1.47218 w_k1_INTC_s111
                                 + 1.44921 w_k1_KO_s111 + 2.27957 w_k1_MAT_s111
                                 + 1.47242 w_k1_MSFT_s111
                                 + 1.27532 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.843697 w_k1_ADM_s111
                                 + 0.829284 w_k1_AXP_s111
                                 + 0.714167 w_k1_DIS_s111
                                 + 0.72815 w_k1_FDX_s111 + 1.2591 w_k1_IBM_s111
                                 + 0.847258 w_k1_INTC_s111
                                 + 0.961914 w_k1_KO_s111
                                 + 1.13005 w_k1_MAT_s111
                                 + 0.735459 w_k1_MSFT_s111
                                 + 0.816531 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.996821 w_k1_ADM_s112
                                 + 0.98049 w_k1_AXP_s112
                                 + 1.24801 w_k1_DIS_s112
                                 + 1.00379 w_k1_FDX_s112
                                 + 1.44328 w_k1_IBM_s112
                                 + 1.08513 w_k1_INTC_s112
                                 + 1.49006 w_k1_KO_s112 + 1.17898 w_k1_MAT_s112
                                 + 1.08028 w_k1_MSFT_s112
                                 + 1.16918 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.08895 w_k1_ADM_s112
                                 + 0.893522 w_k1_AXP_s112
                                 + 1.22039 w_k1_DIS_s112
                                 + 1.10673 w_k1_FDX_s112
                                 + 0.846392 w_k1_IBM_s112
                                 + 1.21679 w_k1_INTC_s112 + 1.0514 w_k1_KO_s112
                                 + 0.526717 w_k1_MAT_s112
                                 + 1.27952 w_k1_MSFT_s112 + 1.0125 w_k1_PG_s112
                                 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.817692 w_k1_ADM_s112
                                 + 0.759895 w_k1_AXP_s112
                                 + 0.712696 w_k1_DIS_s112
                                 + 0.78234 w_k1_FDX_s112
                                 + 0.92718 w_k1_IBM_s112
                                 + 0.927218 w_k1_INTC_s112
                                 + 0.914254 w_k1_KO_s112
                                 + 0.663405 w_k1_MAT_s112
                                 + 0.906774 w_k1_MSFT_s112
                                 + 0.812424 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.888024 w_k1_ADM_s112
                                 + 1.14705 w_k1_AXP_s112
                                 + 0.95461 w_k1_DIS_s112
                                 + 0.954336 w_k1_FDX_s112
                                 + 1.24541 w_k1_IBM_s112
                                 + 1.11378 w_k1_INTC_s112
                                 + 1.09176 w_k1_KO_s112 + 1.00933 w_k1_MAT_s112
                                 + 0.97546 w_k1_MSFT_s112
                                 + 1.07236 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.892866 w_k1_ADM_s112
                                 + 0.85792 w_k1_AXP_s112
                                 + 0.891136 w_k1_DIS_s112
                                 + 0.804762 w_k1_FDX_s112
                                 + 1.02635 w_k1_IBM_s112
                                 + 1.05909 w_k1_INTC_s112
                                 + 1.21039 w_k1_KO_s112
                                 + 0.839973 w_k1_MAT_s112
                                 + 0.981236 w_k1_MSFT_s112
                                 + 0.990369 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.09296 w_k1_ADM_s112
                                 + 1.40077 w_k1_AXP_s112
                                 + 1.37882 w_k1_DIS_s112
                                 + 1.22547 w_k1_FDX_s112
                                 + 1.13152 w_k1_IBM_s112
                                 + 1.10662 w_k1_INTC_s112
                                 + 1.16409 w_k1_KO_s112 + 1.2773 w_k1_MAT_s112
                                 + 1.06111 w_k1_MSFT_s112
                                 + 1.15708 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.779665 w_k1_ADM_s112
                                 + 0.921066 w_k1_AXP_s112
                                 + 0.764541 w_k1_DIS_s112
                                 + 0.898874 w_k1_FDX_s112
                                 + 0.756826 w_k1_IBM_s112
                                 + 1.1465 w_k1_INTC_s112
                                 + 0.830389 w_k1_KO_s112
                                 + 0.649341 w_k1_MAT_s112
                                 + 0.932307 w_k1_MSFT_s112
                                 + 0.854919 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.769682 w_k1_ADM_s112
                                 + 0.577127 w_k1_AXP_s112
                                 + 0.571078 w_k1_DIS_s112
                                 + 0.585473 w_k1_FDX_s112
                                 + 0.78402 w_k1_IBM_s112
                                 + 0.71376 w_k1_INTC_s112
                                 + 0.80146 w_k1_KO_s112
                                 + 0.552579 w_k1_MAT_s112
                                 + 0.686666 w_k1_MSFT_s112
                                 + 0.789436 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.871895 w_k1_ADM_s113
                                 + 0.798875 w_k1_AXP_s113
                                 + 0.967045 w_k1_DIS_s113
                                 + 0.9158 w_k1_FDX_s113
                                 + 0.843472 w_k1_IBM_s113
                                 + 0.998735 w_k1_INTC_s113
                                 + 0.948913 w_k1_KO_s113
                                 + 1.11238 w_k1_MAT_s113
                                 + 0.963478 w_k1_MSFT_s113
                                 + 0.881344 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.943146 w_k1_ADM_s113
                                 + 1.06878 w_k1_AXP_s113
                                 + 1.24208 w_k1_DIS_s113
                                 + 1.09116 w_k1_FDX_s113
                                 + 1.10049 w_k1_IBM_s113
                                 + 0.937651 w_k1_INTC_s113
                                 + 1.23705 w_k1_KO_s113 + 1.52819 w_k1_MAT_s113
                                 + 1.13549 w_k1_MSFT_s113
                                 + 1.06267 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.88764 w_k1_ADM_s113
                                 + 0.823158 w_k1_AXP_s113
                                 + 0.748714 w_k1_DIS_s113
                                 + 0.812011 w_k1_FDX_s113
                                 + 1.05257 w_k1_IBM_s113
                                 + 0.864739 w_k1_INTC_s113
                                 + 0.957749 w_k1_KO_s113
                                 + 1.28442 w_k1_MAT_s113
                                 + 0.913292 w_k1_MSFT_s113
                                 + 0.835029 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.965759 w_k1_ADM_s113
                                 + 0.795521 w_k1_AXP_s113
                                 + 0.77872 w_k1_DIS_s113
                                 + 0.93605 w_k1_FDX_s113
                                 + 0.635228 w_k1_IBM_s113
                                 + 0.837924 w_k1_INTC_s113
                                 + 0.753523 w_k1_KO_s113
                                 + 0.621285 w_k1_MAT_s113
                                 + 0.866621 w_k1_MSFT_s113
                                 + 0.853514 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.35305 w_k1_ADM_s113
                                 + 1.30262 w_k1_AXP_s113
                                 + 1.85333 w_k1_DIS_s113
                                 + 1.46502 w_k1_FDX_s113
                                 + 1.34004 w_k1_IBM_s113
                                 + 1.41615 w_k1_INTC_s113
                                 + 1.67519 w_k1_KO_s113 + 1.37829 w_k1_MAT_s113
                                 + 1.44121 w_k1_MSFT_s113
                                 + 1.27242 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.34723 w_k1_ADM_s113
                                 + 1.12391 w_k1_AXP_s113
                                 + 1.37323 w_k1_DIS_s113
                                 + 1.27569 w_k1_FDX_s113
                                 + 1.05943 w_k1_IBM_s113
                                 + 1.14793 w_k1_INTC_s113
                                 + 1.25662 w_k1_KO_s113 + 1.41574 w_k1_MAT_s113
                                 + 1.37824 w_k1_MSFT_s113
                                 + 1.22883 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.68319 w_k1_ADM_s113
                                 + 0.572717 w_k1_AXP_s113
                                 + 0.486448 w_k1_DIS_s113
                                 + 0.683645 w_k1_FDX_s113
                                 + 0.778772 w_k1_IBM_s113
                                 + 0.817571 w_k1_INTC_s113
                                 + 0.758861 w_k1_KO_s113
                                 + 1.14877 w_k1_MAT_s113
                                 + 0.699131 w_k1_MSFT_s113
                                 + 0.716401 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.862586 w_k1_ADM_s113
                                 + 0.95254 w_k1_AXP_s113
                                 + 0.934809 w_k1_DIS_s113
                                 + 0.93516 w_k1_FDX_s113
                                 + 1.46989 w_k1_IBM_s113
                                 + 1.01651 w_k1_INTC_s113
                                 + 1.44957 w_k1_KO_s113 + 1.40578 w_k1_MAT_s113
                                 + 1.00199 w_k1_MSFT_s113
                                 + 0.918386 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.3653 w_k1_ADM_s114
                                 + 1.41301 w_k1_AXP_s114
                                 + 1.52913 w_k1_DIS_s114
                                 + 1.43483 w_k1_FDX_s114
                                 + 1.23723 w_k1_IBM_s114
                                 + 1.25159 w_k1_INTC_s114
                                 + 1.27829 w_k1_KO_s114 + 1.31055 w_k1_MAT_s114
                                 + 1.34016 w_k1_MSFT_s114
                                 + 1.22794 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.20081 w_k1_ADM_s114
                                 + 1.0429 w_k1_AXP_s114 + 1.08742 w_k1_DIS_s114
                                 + 1.04059 w_k1_FDX_s114
                                 + 1.12289 w_k1_IBM_s114
                                 + 0.815006 w_k1_INTC_s114
                                 + 1.09063 w_k1_KO_s114 + 1.2018 w_k1_MAT_s114
                                 + 1.00998 w_k1_MSFT_s114
                                 + 1.04944 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.905609 w_k1_ADM_s114
                                 + 0.795648 w_k1_AXP_s114
                                 + 0.789189 w_k1_DIS_s114
                                 + 0.771594 w_k1_FDX_s114
                                 + 0.937515 w_k1_IBM_s114
                                 + 0.876545 w_k1_INTC_s114
                                 + 0.923012 w_k1_KO_s114
                                 + 0.767445 w_k1_MAT_s114
                                 + 0.853917 w_k1_MSFT_s114
                                 + 0.935184 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.864557 w_k1_ADM_s114
                                 + 0.827349 w_k1_AXP_s114
                                 + 0.759299 w_k1_DIS_s114
                                 + 0.806634 w_k1_FDX_s114
                                 + 1.34337 w_k1_IBM_s114
                                 + 0.926809 w_k1_INTC_s114
                                 + 1.11675 w_k1_KO_s114 + 1.01534 w_k1_MAT_s114
                                 + 0.969886 w_k1_MSFT_s114
                                 + 1.02296 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.40764 w_k1_ADM_s114
                                 + 1.32034 w_k1_AXP_s114
                                 + 1.17831 w_k1_DIS_s114
                                 + 1.29593 w_k1_FDX_s114
                                 + 0.953951 w_k1_IBM_s114
                                 + 1.17116 w_k1_INTC_s114 + 1.0568 w_k1_KO_s114
                                 + 0.852641 w_k1_MAT_s114
                                 + 1.31381 w_k1_MSFT_s114
                                 + 1.18383 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.982613 w_k1_ADM_s114
                                 + 0.912417 w_k1_AXP_s114
                                 + 0.880969 w_k1_DIS_s114
                                 + 0.911048 w_k1_FDX_s114
                                 + 0.696154 w_k1_IBM_s114
                                 + 0.876629 w_k1_INTC_s114
                                 + 0.771713 w_k1_KO_s114
                                 + 0.551212 w_k1_MAT_s114
                                 + 1.01517 w_k1_MSFT_s114
                                 + 0.961022 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.03347 w_k1_ADM_s114
                                 + 0.979413 w_k1_AXP_s114
                                 + 0.887398 w_k1_DIS_s114
                                 + 1.09903 w_k1_FDX_s114
                                 + 0.899965 w_k1_IBM_s114
                                 + 1.05249 w_k1_INTC_s114
                                 + 0.908309 w_k1_KO_s114
                                 + 0.764784 w_k1_MAT_s114
                                 + 1.14391 w_k1_MSFT_s114
                                 + 0.992024 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.16488 w_k1_ADM_s114
                                 + 1.218 w_k1_AXP_s114 + 1.25744 w_k1_DIS_s114
                                 + 1.15434 w_k1_FDX_s114
                                 + 1.50385 w_k1_IBM_s114
                                 + 1.13791 w_k1_INTC_s114
                                 + 1.46187 w_k1_KO_s114 + 1.67067 w_k1_MAT_s114
                                 + 1.18488 w_k1_MSFT_s114
                                 + 1.23713 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.836259 w_k1_ADM_s115
                                 + 0.760977 w_k1_AXP_s115
                                 + 0.723569 w_k1_DIS_s115
                                 + 0.839265 w_k1_FDX_s115
                                 + 0.823126 w_k1_IBM_s115
                                 + 0.880635 w_k1_INTC_s115
                                 + 0.821728 w_k1_KO_s115
                                 + 1.01839 w_k1_MAT_s115
                                 + 0.833334 w_k1_MSFT_s115
                                 + 0.828939 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.00854 w_k1_ADM_s115
                                 + 1.21368 w_k1_AXP_s115
                                 + 0.942942 w_k1_DIS_s115
                                 + 1.0451 w_k1_FDX_s115 + 1.37065 w_k1_IBM_s115
                                 + 1.03336 w_k1_INTC_s115 + 1.228 w_k1_KO_s115
                                 + 1.64985 w_k1_MAT_s115
                                 + 0.945152 w_k1_MSFT_s115
                                 + 0.932411 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.667212 w_k1_ADM_s115
                                 + 0.991223 w_k1_AXP_s115
                                 + 0.722574 w_k1_DIS_s115
                                 + 0.803561 w_k1_FDX_s115
                                 + 1.11413 w_k1_IBM_s115
                                 + 0.973714 w_k1_INTC_s115
                                 + 0.914377 w_k1_KO_s115
                                 + 1.15596 w_k1_MAT_s115
                                 + 0.822244 w_k1_MSFT_s115
                                 + 0.836602 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.905802 w_k1_ADM_s115
                                 + 0.809721 w_k1_AXP_s115
                                 + 0.798679 w_k1_DIS_s115
                                 + 0.840143 w_k1_FDX_s115
                                 + 1.1515 w_k1_IBM_s115
                                 + 0.968805 w_k1_INTC_s115
                                 + 0.934206 w_k1_KO_s115
                                 + 1.07732 w_k1_MAT_s115
                                 + 0.905173 w_k1_MSFT_s115
                                 + 0.933332 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.06722 w_k1_ADM_s115
                                 + 1.03732 w_k1_AXP_s115
                                 + 1.07769 w_k1_DIS_s115 + 1.0508 w_k1_FDX_s115
                                 + 0.960145 w_k1_IBM_s115
                                 + 1.2707 w_k1_INTC_s115 + 1.03391 w_k1_KO_s115
                                 + 0.802047 w_k1_MAT_s115
                                 + 1.04244 w_k1_MSFT_s115
                                 + 1.03607 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.23318 w_k1_ADM_s115
                                 + 1.35008 w_k1_AXP_s115
                                 + 1.36867 w_k1_DIS_s115
                                 + 1.28806 w_k1_FDX_s115
                                 + 1.02882 w_k1_IBM_s115
                                 + 1.30785 w_k1_INTC_s115 + 1.1001 w_k1_KO_s115
                                 + 1.06529 w_k1_MAT_s115
                                 + 1.17803 w_k1_MSFT_s115
                                 + 1.11311 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.27083 w_k1_ADM_s115
                                 + 1.59902 w_k1_AXP_s115
                                 + 1.64864 w_k1_DIS_s115
                                 + 1.56419 w_k1_FDX_s115
                                 + 1.12227 w_k1_IBM_s115
                                 + 1.39636 w_k1_INTC_s115
                                 + 1.35237 w_k1_KO_s115 + 1.45323 w_k1_MAT_s115
                                 + 1.56473 w_k1_MSFT_s115 + 1.2851 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.832542 w_k1_ADM_s115
                                 + 0.824101 w_k1_AXP_s115
                                 + 0.753264 w_k1_DIS_s115
                                 + 0.981685 w_k1_FDX_s115
                                 + 0.645363 w_k1_IBM_s115
                                 + 1.06162 w_k1_INTC_s115
                                 + 0.772891 w_k1_KO_s115
                                 + 0.649774 w_k1_MAT_s115
                                 + 0.882756 w_k1_MSFT_s115
                                 + 0.79319 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.00573 w_k1_ADM_s116
                                 + 1.06988 w_k1_AXP_s116
                                 + 0.857233 w_k1_DIS_s116
                                 + 0.940129 w_k1_FDX_s116
                                 + 0.881633 w_k1_IBM_s116
                                 + 1.13198 w_k1_INTC_s116
                                 + 0.89399 w_k1_KO_s116 + 1.12864 w_k1_MAT_s116
                                 + 1.05192 w_k1_MSFT_s116
                                 + 0.959082 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.741268 w_k1_ADM_s116
                                 + 0.733293 w_k1_AXP_s116
                                 + 0.54141 w_k1_DIS_s116
                                 + 0.772351 w_k1_FDX_s116
                                 + 0.739185 w_k1_IBM_s116
                                 + 0.77771 w_k1_INTC_s116
                                 + 0.687971 w_k1_KO_s116 + 1.3724 w_k1_MAT_s116
                                 + 0.794939 w_k1_MSFT_s116
                                 + 0.814967 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.23098 w_k1_ADM_s116
                                 + 1.15446 w_k1_AXP_s116
                                 + 1.28608 w_k1_DIS_s116
                                 + 1.20872 w_k1_FDX_s116
                                 + 1.12458 w_k1_IBM_s116
                                 + 1.06198 w_k1_INTC_s116
                                 + 1.22193 w_k1_KO_s116 + 1.61956 w_k1_MAT_s116
                                 + 1.2722 w_k1_MSFT_s116 + 1.27366 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.10685 w_k1_ADM_s116
                                 + 1.1491 w_k1_AXP_s116 + 1.27197 w_k1_DIS_s116
                                 + 1.0126 w_k1_FDX_s116 + 1.55042 w_k1_IBM_s116
                                 + 1.00096 w_k1_INTC_s116 + 1.3687 w_k1_KO_s116
                                 + 2.4084 w_k1_MAT_s116
                                 + 1.09255 w_k1_MSFT_s116
                                 + 1.21081 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.872418 w_k1_ADM_s116
                                 + 0.821675 w_k1_AXP_s116
                                 + 0.812901 w_k1_DIS_s116
                                 + 0.774097 w_k1_FDX_s116
                                 + 1.11048 w_k1_IBM_s116
                                 + 0.975138 w_k1_INTC_s116
                                 + 0.965677 w_k1_KO_s116 + 1.0923 w_k1_MAT_s116
                                 + 0.900347 w_k1_MSFT_s116
                                 + 1.01077 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.06478 w_k1_ADM_s116
                                 + 1.10457 w_k1_AXP_s116
                                 + 1.08556 w_k1_DIS_s116
                                 + 1.12923 w_k1_FDX_s116
                                 + 0.800029 w_k1_IBM_s116
                                 + 1.08107 w_k1_INTC_s116
                                 + 0.904004 w_k1_KO_s116
                                 + 0.640194 w_k1_MAT_s116
                                 + 1.00489 w_k1_MSFT_s116
                                 + 1.00606 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.28377 w_k1_ADM_s116
                                 + 1.72194 w_k1_AXP_s116 + 1.2 w_k1_DIS_s116
                                 + 1.23122 w_k1_FDX_s116
                                 + 0.981258 w_k1_IBM_s116
                                 + 0.981793 w_k1_INTC_s116
                                 + 1.01501 w_k1_KO_s116 + 1.25727 w_k1_MAT_s116
                                 + 1.11675 w_k1_MSFT_s116
                                 + 1.11366 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.17088 w_k1_ADM_s116
                                 + 1.5626 w_k1_AXP_s116 + 1.78299 w_k1_DIS_s116
                                 + 1.47422 w_k1_FDX_s116
                                 + 1.34109 w_k1_IBM_s116
                                 + 1.43147 w_k1_INTC_s116
                                 + 1.32021 w_k1_KO_s116 + 1.58318 w_k1_MAT_s116
                                 + 1.46687 w_k1_MSFT_s116
                                 + 1.13612 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.30679 w_k1_ADM_s117
                                 + 1.38023 w_k1_AXP_s117
                                 + 1.33898 w_k1_DIS_s117
                                 + 1.08483 w_k1_FDX_s117
                                 + 1.28192 w_k1_IBM_s117
                                 + 1.3063 w_k1_INTC_s117 + 1.2974 w_k1_KO_s117
                                 + 1.65055 w_k1_MAT_s117
                                 + 1.18944 w_k1_MSFT_s117
                                 + 1.13475 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.898138 w_k1_ADM_s117
                                 + 1.00386 w_k1_AXP_s117
                                 + 0.880948 w_k1_DIS_s117
                                 + 0.78152 w_k1_FDX_s117
                                 + 1.51032 w_k1_IBM_s117
                                 + 0.891974 w_k1_INTC_s117
                                 + 1.0414 w_k1_KO_s117 + 0.719553 w_k1_MAT_s117
                                 + 0.761956 w_k1_MSFT_s117
                                 + 0.96332 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.975605 w_k1_ADM_s117
                                 + 1.09546 w_k1_AXP_s117
                                 + 0.983367 w_k1_DIS_s117
                                 + 0.934649 w_k1_FDX_s117
                                 + 1.19553 w_k1_IBM_s117
                                 + 1.02944 w_k1_INTC_s117
                                 + 0.990775 w_k1_KO_s117
                                 + 0.97881 w_k1_MAT_s117
                                 + 0.975808 w_k1_MSFT_s117
                                 + 1.02215 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.878787 w_k1_ADM_s117
                                 + 0.971817 w_k1_AXP_s117
                                 + 0.988345 w_k1_DIS_s117
                                 + 0.846952 w_k1_FDX_s117
                                 + 0.831046 w_k1_IBM_s117
                                 + 0.988862 w_k1_INTC_s117
                                 + 0.834359 w_k1_KO_s117
                                 + 0.765398 w_k1_MAT_s117
                                 + 0.975625 w_k1_MSFT_s117
                                 + 0.931329 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.37351 w_k1_ADM_s117
                                 + 1.50823 w_k1_AXP_s117
                                 + 1.48169 w_k1_DIS_s117
                                 + 1.26467 w_k1_FDX_s117
                                 + 0.877004 w_k1_IBM_s117
                                 + 1.28675 w_k1_INTC_s117
                                 + 0.979718 w_k1_KO_s117
                                 + 0.73076 w_k1_MAT_s117
                                 + 1.17058 w_k1_MSFT_s117 + 1.0238 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.27896 w_k1_ADM_s117
                                 + 1.29771 w_k1_AXP_s117
                                 + 1.08305 w_k1_DIS_s117
                                 + 1.11499 w_k1_FDX_s117
                                 + 0.599741 w_k1_IBM_s117
                                 + 1.09472 w_k1_INTC_s117
                                 + 0.805698 w_k1_KO_s117
                                 + 0.548445 w_k1_MAT_s117
                                 + 1.00757 w_k1_MSFT_s117
                                 + 1.02124 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.734926 w_k1_ADM_s117
                                 + 0.71218 w_k1_AXP_s117
                                 + 0.645787 w_k1_DIS_s117
                                 + 0.75136 w_k1_FDX_s117
                                 + 0.697281 w_k1_IBM_s117
                                 + 0.759303 w_k1_INTC_s117
                                 + 0.678861 w_k1_KO_s117
                                 + 0.403185 w_k1_MAT_s117
                                 + 0.796699 w_k1_MSFT_s117
                                 + 0.805901 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.950188 w_k1_ADM_s117
                                 + 0.847711 w_k1_AXP_s117
                                 + 0.8609 w_k1_DIS_s117
                                 + 0.768417 w_k1_FDX_s117
                                 + 1.06614 w_k1_IBM_s117
                                 + 0.94526 w_k1_INTC_s117 + 1.1312 w_k1_KO_s117
                                 + 1.27413 w_k1_MAT_s117
                                 + 0.935225 w_k1_MSFT_s117
                                 + 0.958333 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.023408879813362 w_k1_ADM_s000
                                 + 0.015673651291012 w_k1_ADM_s001
                                 + 0.0370006686174427 w_k1_ADM_s002
                                 + 0.0108314383723707 w_k1_ADM_s003
                                 + 0.0380010231649804 w_k1_ADM_s004
                                 + 0.0235559702750075 w_k1_ADM_s005
                                 + 0.0414841489906415 w_k1_ADM_s006
                                 + 0.00799283252588988 w_k1_ADM_s007
                                 + 0.0390430861469811 w_k1_ADM_s010
                                 + 0.0412931337776121 w_k1_ADM_s011
                                 + 0.0301814593760411 w_k1_ADM_s012
                                 + 0.0533870658143089 w_k1_ADM_s013
                                 + 0.0795743545769545 w_k1_ADM_s014
                                 + 0.0176743761733582 w_k1_ADM_s015
                                 + 0.0322189000684354 w_k1_ADM_s016
                                 + 0.0163489651051551 w_k1_ADM_s017
                                 + 0.0123887126413284 w_k1_ADM_s100
                                 + 0.0440376689437549 w_k1_ADM_s101
                                 + 0.0207648229627897 w_k1_ADM_s102
                                 + 0.0112324568807865 w_k1_ADM_s103
                                 + 0.0236592281158955 w_k1_ADM_s104
                                 + 0.0232456352906917 w_k1_ADM_s105
                                 + 0.0132484038730853 w_k1_ADM_s106
                                 + 0.0219517877647391 w_k1_ADM_s107
                                 + 0.0614482220524852 w_k1_ADM_s110
                                 + 0.0660906576100009 w_k1_ADM_s111
                                 + 0.0154315155328188 w_k1_ADM_s112
                                 + 0.0391844752079824 w_k1_ADM_s113
                                 + 0.024777727698385 w_k1_ADM_s114
                                 + 0.0473612184392345 w_k1_ADM_s115
                                 + 0.0553315067284275 w_k1_ADM_s116
                                 + 0.0176614087565305 w_k1_ADM_s117
                                 + 0.0229582308552006 w_k1_AXP_s000
                                 + 0.0151515115519048 w_k1_AXP_s001
                                 + 0.0395913222154965 w_k1_AXP_s002
                                 + 0.0103796586071543 w_k1_AXP_s003
                                 + 0.040609859384801 w_k1_AXP_s004
                                 + 0.0227971402461884 w_k1_AXP_s005
                                 + 0.0398372406548751 w_k1_AXP_s006
                                 + 0.010220473679276 w_k1_AXP_s007
                                 + 0.036904549285655 w_k1_AXP_s010
                                 + 0.0465627207912645 w_k1_AXP_s011
                                 + 0.026786323033371 w_k1_AXP_s012
                                 + 0.0492462090101643 w_k1_AXP_s013
                                 + 0.0790018558168597 w_k1_AXP_s014
                                 + 0.0183714561106403 w_k1_AXP_s015
                                 + 0.0344069805791248 w_k1_AXP_s016
                                 + 0.0182007114719793 w_k1_AXP_s017
                                 + 0.013568014686281 w_k1_AXP_s100
                                 + 0.0437401238827058 w_k1_AXP_s101
                                 + 0.0222908760744957 w_k1_AXP_s102
                                 + 0.0109819204132286 w_k1_AXP_s103
                                 + 0.0259393909431738 w_k1_AXP_s104
                                 + 0.0253146663853957 w_k1_AXP_s105
                                 + 0.0124845041771118 w_k1_AXP_s106
                                 + 0.0239849880194274 w_k1_AXP_s107
                                 + 0.0659038004813459 w_k1_AXP_s110
                                 + 0.0684061844619096 w_k1_AXP_s111
                                 + 0.0161693318834931 w_k1_AXP_s112
                                 + 0.0378812258296866 w_k1_AXP_s113
                                 + 0.023326891220812 w_k1_AXP_s114
                                 + 0.0504151922557717 w_k1_AXP_s115
                                 + 0.0584541130923689 w_k1_AXP_s116
                                 + 0.0187220802190114 w_k1_AXP_s117
                                 + 0.0218635147586194 w_k1_DIS_s000
                                 + 0.0152246448356473 w_k1_DIS_s001
                                 + 0.0394903872202348 w_k1_DIS_s002
                                 + 0.0103195854426504 w_k1_DIS_s003
                                 + 0.0408199953310985 w_k1_DIS_s004
                                 + 0.0246401147758714 w_k1_DIS_s005
                                 + 0.0423654823257232 w_k1_DIS_s006
                                 + 0.00915501409629412 w_k1_DIS_s007
                                 + 0.0374021168011257 w_k1_DIS_s010
                                 + 0.045122587343711 w_k1_DIS_s011
                                 + 0.0298769060810273 w_k1_DIS_s012
                                 + 0.0495147216419042 w_k1_DIS_s013
                                 + 0.0793707824538356 w_k1_DIS_s014
                                 + 0.0182533905846993 w_k1_DIS_s015
                                 + 0.0346086968929453 w_k1_DIS_s016
                                 + 0.0190135401757068 w_k1_DIS_s017
                                 + 0.0138328766539428 w_k1_DIS_s100
                                 + 0.0439193890133802 w_k1_DIS_s101
                                 + 0.0212360878731464 w_k1_DIS_s102
                                 + 0.0121610334475766 w_k1_DIS_s103
                                 + 0.0233645541830855 w_k1_DIS_s104
                                 + 0.0242169466034802 w_k1_DIS_s105
                                 + 0.0137497123541047 w_k1_DIS_s106
                                 + 0.0265303939186656 w_k1_DIS_s107
                                 + 0.0644597787862286 w_k1_DIS_s110
                                 + 0.0647188526000034 w_k1_DIS_s111
                                 + 0.0171622820685693 w_k1_DIS_s112
                                 + 0.0429956121318661 w_k1_DIS_s113
                                 + 0.0228517314750999 w_k1_DIS_s114
                                 + 0.0482012839123235 w_k1_DIS_s115
                                 + 0.0558473426721696 w_k1_DIS_s116
                                 + 0.0175434945509916 w_k1_DIS_s117
                                 + 0.0226008539177152 w_k1_FDX_s000
                                 + 0.0153649575509402 w_k1_FDX_s001
                                 + 0.0385832393036475 w_k1_FDX_s002
                                 + 0.010105772403325 w_k1_FDX_s003
                                 + 0.0415697931022245 w_k1_FDX_s004
                                 + 0.0245987136939306 w_k1_FDX_s005
                                 + 0.0427230787386819 w_k1_FDX_s006
                                 + 0.0089480380378011 w_k1_FDX_s007
                                 + 0.0380406785382168 w_k1_FDX_s010
                                 + 0.042202758971972 w_k1_FDX_s011
                                 + 0.0285907133821421 w_k1_FDX_s012
                                 + 0.0479369851994831 w_k1_FDX_s013
                                 + 0.0799682669960502 w_k1_FDX_s014
                                 + 0.0180116298593839 w_k1_FDX_s015
                                 + 0.0346194572195035 w_k1_FDX_s016
                                 + 0.018517224768964 w_k1_FDX_s017
                                 + 0.0134249060958176 w_k1_FDX_s100
                                 + 0.0427464118006133 w_k1_FDX_s101
                                 + 0.0218016454439433 w_k1_FDX_s102
                                 + 0.0108099928288409 w_k1_FDX_s103
                                 + 0.0234421878871223 w_k1_FDX_s104
                                 + 0.0249116523252013 w_k1_FDX_s105
                                 + 0.0128485942919232 w_k1_FDX_s106
                                 + 0.0257042666414767 w_k1_FDX_s107
                                 + 0.0611961437470026 w_k1_FDX_s110
                                 + 0.0661769769211479 w_k1_FDX_s111
                                 + 0.0160690391088764 w_k1_FDX_s112
                                 + 0.040703139341437 w_k1_FDX_s113
                                 + 0.023127169078357 w_k1_FDX_s114
                                 + 0.0497112018098529 w_k1_FDX_s115
                                 + 0.0552305081977988 w_k1_FDX_s116
                                 + 0.0159927759019171 w_k1_FDX_s117
                                 + 0.0232875777994435 w_k1_IBM_s000
                                 + 0.0162117947444866 w_k1_IBM_s001
                                 + 0.0391295098149834 w_k1_IBM_s002
                                 + 0.00973461482805964 w_k1_IBM_s003
                                 + 0.0425935895172014 w_k1_IBM_s004
                                 + 0.023738997807261 w_k1_IBM_s005
                                 + 0.042505353526637 w_k1_IBM_s006
                                 + 0.0090896821695465 w_k1_IBM_s007
                                 + 0.0379947439727816 w_k1_IBM_s010
                                 + 0.0435922700681949 w_k1_IBM_s011
                                 + 0.0285332030767559 w_k1_IBM_s012
                                 + 0.0488633959122549 w_k1_IBM_s013
                                 + 0.0837435340166739 w_k1_IBM_s014
                                 + 0.0187332710976934 w_k1_IBM_s015
                                 + 0.0363027306565831 w_k1_IBM_s016
                                 + 0.018233349186439 w_k1_IBM_s017
                                 + 0.01354737689275 w_k1_IBM_s100
                                 + 0.0451649781008851 w_k1_IBM_s101
                                 + 0.0225600402164691 w_k1_IBM_s102
                                 + 0.0112769640497218 w_k1_IBM_s103
                                 + 0.0225791688327561 w_k1_IBM_s104
                                 + 0.024100625727121 w_k1_IBM_s105
                                 + 0.0127980687747822 w_k1_IBM_s106
                                 + 0.0251872985022443 w_k1_IBM_s107
                                 + 0.064417798760698 w_k1_IBM_s110
                                 + 0.0677241771893651 w_k1_IBM_s111
                                 + 0.0165915536157533 w_k1_IBM_s112
                                 + 0.0419489395075965 w_k1_IBM_s113
                                 + 0.0249550304166658 w_k1_IBM_s114
                                 + 0.0482606527593735 w_k1_IBM_s115
                                 + 0.0585714063308855 w_k1_IBM_s116
                                 + 0.0157200567546371 w_k1_IBM_s117
                                 + 0.0225851302242835 w_k1_INTC_s000
                                 + 0.0122929763110004 w_k1_INTC_s001
                                 + 0.0404930797329825 w_k1_INTC_s002
                                 + 0.00823048470154693 w_k1_INTC_s003
                                 + 0.0388478634997698 w_k1_INTC_s004
                                 + 0.0188902332485708 w_k1_INTC_s005
                                 + 0.043074683043663 w_k1_INTC_s006
                                 + 0.00808006279381438 w_k1_INTC_s007
                                 + 0.0319258342767281 w_k1_INTC_s010
                                 + 0.0391666033921403 w_k1_INTC_s011
                                 + 0.0279016167655634 w_k1_INTC_s012
                                 + 0.0530894186868317 w_k1_INTC_s013
                                 + 0.0784119766864694 w_k1_INTC_s014
                                 + 0.0184423689938741 w_k1_INTC_s015
                                 + 0.0298877416654045 w_k1_INTC_s016
                                 + 0.0165478692989758 w_k1_INTC_s017
                                 + 0.0133308838166339 w_k1_INTC_s100
                                 + 0.0417292868200957 w_k1_INTC_s101
                                 + 0.0213824783645991 w_k1_INTC_s102
                                 + 0.0108734222015434 w_k1_INTC_s103
                                 + 0.0261900025344975 w_k1_INTC_s104
                                 + 0.028072489845343 w_k1_INTC_s105
                                 + 0.0117006923187513 w_k1_INTC_s106
                                 + 0.0257591298584372 w_k1_INTC_s107
                                 + 0.0685583704589349 w_k1_INTC_s110
                                 + 0.0713227212616266 w_k1_INTC_s111
                                 + 0.0177394353446288 w_k1_INTC_s112
                                 + 0.0398431464581027 w_k1_INTC_s113
                                 + 0.0223056708478524 w_k1_INTC_s114
                                 + 0.0531866862459922 w_k1_INTC_s115
                                 + 0.0571270282856354 w_k1_INTC_s116
                                 + 0.0173025938909321 w_k1_INTC_s117
                                 + 0.0237437556775046 w_k1_KO_s000
                                 + 0.0160242480315243 w_k1_KO_s001
                                 + 0.0378177632773714 w_k1_KO_s002
                                 + 0.0097488045827644 w_k1_KO_s003
                                 + 0.0414741846130556 w_k1_KO_s004
                                 + 0.0235831057359707 w_k1_KO_s005
                                 + 0.0415022141338753 w_k1_KO_s006
                                 + 0.00918688135961523 w_k1_KO_s007
                                 + 0.0378140107799737 w_k1_KO_s010
                                 + 0.0411647278838869 w_k1_KO_s011
                                 + 0.0295955870214054 w_k1_KO_s012
                                 + 0.0495116270459951 w_k1_KO_s013
                                 + 0.0754351081768807 w_k1_KO_s014
                                 + 0.0164882211772171 w_k1_KO_s015
                                 + 0.0321235862252911 w_k1_KO_s016
                                 + 0.0175662769748521 w_k1_KO_s017
                                 + 0.0143363645962194 w_k1_KO_s100
                                 + 0.0439587116852599 w_k1_KO_s101
                                 + 0.0231694803587826 w_k1_KO_s102
                                 + 0.0125866502786828 w_k1_KO_s103
                                 + 0.0226494161945162 w_k1_KO_s104
                                 + 0.0254821420905798 w_k1_KO_s105
                                 + 0.0131267709085975 w_k1_KO_s106
                                 + 0.0253081942630829 w_k1_KO_s107
                                 + 0.0660940105742506 w_k1_KO_s110
                                 + 0.0650478477891585 w_k1_KO_s111
                                 + 0.0176802274652112 w_k1_KO_s112
                                 + 0.0456584305716116 w_k1_KO_s113
                                 + 0.0243192013009503 w_k1_KO_s114
                                 + 0.0482269762544332 w_k1_KO_s115
                                 + 0.0561099432829211 w_k1_KO_s116
                                 + 0.0154700832753233 w_k1_KO_s117
                                 + 0.0214999693063151 w_k1_MAT_s000
                                 + 0.0139926234865255 w_k1_MAT_s001
                                 + 0.0342377065500764 w_k1_MAT_s002
                                 + 0.0105888177544845 w_k1_MAT_s003
                                 + 0.043012500677801 w_k1_MAT_s004
                                 + 0.0244717676501502 w_k1_MAT_s005
                                 + 0.0372726489587757 w_k1_MAT_s006
                                 + 0.0110655033682819 w_k1_MAT_s007
                                 + 0.0460124331176209 w_k1_MAT_s010
                                 + 0.0379239678717346 w_k1_MAT_s011
                                 + 0.0279333059111134 w_k1_MAT_s012
                                 + 0.053244584472456 w_k1_MAT_s013
                                 + 0.084879270095558 w_k1_MAT_s014
                                 + 0.0156621347396916 w_k1_MAT_s015
                                 + 0.036881999437733 w_k1_MAT_s016
                                 + 0.0199289520466059 w_k1_MAT_s017
                                 + 0.0113692125871508 w_k1_MAT_s100
                                 + 0.044614947145485 w_k1_MAT_s101
                                 + 0.0213392312703343 w_k1_MAT_s102
                                 + 0.0123437638150761 w_k1_MAT_s103
                                 + 0.0289701358589681 w_k1_MAT_s104
                                 + 0.0218321029612403 w_k1_MAT_s105
                                 + 0.0113388718352788 w_k1_MAT_s106
                                 + 0.0289032994077501 w_k1_MAT_s107
                                 + 0.0740117594744329 w_k1_MAT_s110
                                 + 0.0761451397776044 w_k1_MAT_s111
                                 + 0.0139724505780783 w_k1_MAT_s112
                                 + 0.051753606279223 w_k1_MAT_s113
                                 + 0.0229161057419869 w_k1_MAT_s114
                                 + 0.0512892170805812 w_k1_MAT_s115
                                 + 0.0776953848080715 w_k1_MAT_s116
                                 + 0.0136966805376801 w_k1_MAT_s117
                                 + 0.0225821661653792 w_k1_MSFT_s000
                                 + 0.014568653463268 w_k1_MSFT_s001
                                 + 0.0392917316759424 w_k1_MSFT_s002
                                 + 0.00916829159965636 w_k1_MSFT_s003
                                 + 0.0424658180629298 w_k1_MSFT_s004
                                 + 0.022587561695015 w_k1_MSFT_s005
                                 + 0.0408488533280421 w_k1_MSFT_s006
                                 + 0.008565964186479 w_k1_MSFT_s007
                                 + 0.0386910008655385 w_k1_MSFT_s010
                                 + 0.0427910372400075 w_k1_MSFT_s011
                                 + 0.0283699864814607 w_k1_MSFT_s012
                                 + 0.0485799225095449 w_k1_MSFT_s013
                                 + 0.0732487327010116 w_k1_MSFT_s014
                                 + 0.0174263699329746 w_k1_MSFT_s015
                                 + 0.032650099974198 w_k1_MSFT_s016
                                 + 0.0170099612472696 w_k1_MSFT_s017
                                 + 0.0123678443548264 w_k1_MSFT_s100
                                 + 0.0432822040795895 w_k1_MSFT_s101
                                 + 0.0222574229682082 w_k1_MSFT_s102
                                 + 0.0107685796601935 w_k1_MSFT_s103
                                 + 0.0224129980308142 w_k1_MSFT_s104
                                 + 0.0245665810556386 w_k1_MSFT_s105
                                 + 0.0118779019762347 w_k1_MSFT_s106
                                 + 0.0279010251947109 w_k1_MSFT_s107
                                 + 0.0646986823494855 w_k1_MSFT_s110
                                 + 0.0640941527298123 w_k1_MSFT_s111
                                 + 0.0168648478232232 w_k1_MSFT_s112
                                 + 0.0426292054419115 w_k1_MSFT_s113
                                 + 0.0243402591751967 w_k1_MSFT_s114
                                 + 0.0483963856191011 w_k1_MSFT_s115
                                 + 0.0579312327412154 w_k1_MSFT_s116
                                 + 0.0162107933445618 w_k1_MSFT_s117
                                 + 0.0218831876482965 w_k1_PG_s000
                                 + 0.0152485645455325 w_k1_PG_s001
                                 + 0.0355646421774444 w_k1_PG_s002
                                 + 0.00962301668871331 w_k1_PG_s003
                                 + 0.0430345529190442 w_k1_PG_s004
                                 + 0.0237452367445914 w_k1_PG_s005
                                 + 0.0448795410580228 w_k1_PG_s006
                                 + 0.00986604858373868 w_k1_PG_s007
                                 + 0.0380872508029798 w_k1_PG_s010
                                 + 0.0423853636873673 w_k1_PG_s011
                                 + 0.0267964513403138 w_k1_PG_s012
                                 + 0.0529649538329074 w_k1_PG_s013
                                 + 0.075155481282006 w_k1_PG_s014
                                 + 0.0174590444435881 w_k1_PG_s015
                                 + 0.0340897368565552 w_k1_PG_s016
                                 + 0.017760876589378 w_k1_PG_s017
                                 + 0.0122688200594796 w_k1_PG_s100
                                 + 0.0424269536751366 w_k1_PG_s101
                                 + 0.0207509328448906 w_k1_PG_s102
                                 + 0.0110292359211787 w_k1_PG_s103
                                 + 0.0229232104037633 w_k1_PG_s104
                                 + 0.0245435827888972 w_k1_PG_s105
                                 + 0.0123618877018372 w_k1_PG_s106
                                 + 0.0253904548211041 w_k1_PG_s107
                                 + 0.0620886448545007 w_k1_PG_s110
                                 + 0.0625607341275177 w_k1_PG_s111
                                 + 0.0164016088264591 w_k1_PG_s112
                                 + 0.0392590107200957 w_k1_PG_s113
                                 + 0.0242046493010273 w_k1_PG_s114
                                 + 0.0464938844198192 w_k1_PG_s115
                                 + 0.057605340571628 w_k1_PG_s116
                                 + 0.0160011960974543 w_k1_PG_s117 >= 1.16
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
