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
 w_k1_ADM_s0_return_constraint:  1.0053 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.09535 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.12045 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.869352 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.16291 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.893974 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.823578 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.16531 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.76757 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.977323 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.10746 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.892463 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.941401 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.01235 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.10289 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.23906 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.22307 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.82866 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.898668 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.697141 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.976085 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.841504 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.875466 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.11189 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.19764 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.07842 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.89364 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.878117 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.40416 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.911873 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.2822 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.29312 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.793925 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.836186 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.51213 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.8088 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.866775 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.0914 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.974227 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.17353 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.15471 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.781096 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.21719 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.838645 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.772078 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.47311 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.641784 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.33186 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.20853 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.876085 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.917267 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.0141 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.961463 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.64148 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.48247 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.19677 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.08972 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.674257 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.05168 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.923349 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.30332 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.25541 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.48637 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.27009 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.822705 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.851048 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.61825 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.861335 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.17476 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.41771 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.648697 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.748489 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.4302 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.986391 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.953144 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.09012 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.959137 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.18285 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.25793 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.757924 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.41195 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.803893 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.773273 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.64121 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.559863 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.25155 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.907799 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.819006 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.05383 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.95008 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.958609 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.69915 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.10972 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.26724 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.961859 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.659669 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.30314 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.841789 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.04791 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.41667 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.41249 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.942397 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.9164 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.692203 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.49099 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.776064 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.01666 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.5256 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.559216 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.713982 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.63269 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.04807 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.07155 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.07133 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.988865 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.05496 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.15535 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.818844 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.2532 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.880335 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.78956 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.34357 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.766502 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.20898 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.20278 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.95785 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.988076 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.921276 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.956837 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.39085 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.11325 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.98409 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.901865 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.700966 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.11023 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.982027 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.918171 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.22542 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.14562 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.00612 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.0361 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.769654 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.58687 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.893736 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.14647 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.23851 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.715971 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.803117 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.49803 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.919738 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.05144 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.04655 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.875824 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.34145 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.0548 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.00881 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.11807 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.950787 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.90146 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.62797 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.605592 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.01519 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.759088 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.690486 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.945892 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.38583 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.964858 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.46889 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.08073 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.8433 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.23381 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.890685 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.17204 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.657424 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 2.01234 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.08378 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.46042 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.0572 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.573087 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.12167 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.73821 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.727397 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.778552 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.56886 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.543129 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.914289 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.932499 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.67882 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.714517 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.12915 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.963836 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.07584 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.12482 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.876284 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.10369 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.940708 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.916006 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.26967 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.821846 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.07778 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.965779 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.0158 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.927708 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.881568 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.979627 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.26673 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.12019 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.09078 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.980583 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.79225 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.00036 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.871384 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.30879 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.06025 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.05732 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.991719 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.968308 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.724607 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.45594 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.03201 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.874228 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.957295 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.76597 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.792095 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.01492 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.952357 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.01628 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.941156 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.940495 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.22159 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.19153 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.886894 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.26323 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.807701 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.768525 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.43952 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.611159 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.06222 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.816504 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.690931 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.937441 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.958522 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.05619 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.46348 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.973065 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.42361 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.00857 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.798479 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.16454 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.812317 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.46985 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.12186 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.24394 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.858597 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.702347 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.85112 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.09594 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.8029 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.935432 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.60628 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.700395 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.927882 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.22006 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.37029 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.927967 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.17669 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.00641 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.27092 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.958045 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.82732 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.31811 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.960951 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.691508 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.69158 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.537112 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.08928 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.96421 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.694443 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.900379 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.59442 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.91025 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.9051 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.36918 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.73289 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.35388 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.886711 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.12152 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.588164 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 2.20264 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.19798 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.62207 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.05556 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.721945 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.921129 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.780723 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.716493 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.903102 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.78547 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.465653 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.950567 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.09361 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.57656 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.623228 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.34368 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.992901 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.03681 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.13065 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.83887 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.24468 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.937945 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.862508 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.28215 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.721951 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.07588 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.06415 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.979252 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.973786 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.966194 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.01366 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.2641 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.06956 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.06173 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.936801 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.805461 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.11732 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.927731 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.04722 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.11067 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.00322 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.928055 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.990805 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.75721 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.34106 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.912713 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.960315 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.19292 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.739644 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.833602 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.20424 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.97087 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.07206 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.10326 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.982313 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.10075 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.07918 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.866043 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.16253 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.902112 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.856994 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.15917 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.805668 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.14291 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.964742 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.916815 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.01131 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.972247 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.06167 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.36199 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.08373 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.1456 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.06167 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.83283 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.0691 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.972913 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.0404 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.14565 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.19689 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.07555 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.941236 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.85152 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.16334 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.932536 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.03397 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.29061 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.78372 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.955416 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.19322 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.08762 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.948364 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.990417 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.848001585430278 z_k1_s0000
                                 - 0.270000739832485 z_k1_s0001
                                 - 0.273000711448297 z_k1_s0002
                                 - 1.01700479080145 z_k1_s0003
                                 - 0.218000545489784 z_k1_s0004
                                 - 0.509002733837679 z_k1_s0005
                                 - 0.219000810562509 z_k1_s0006
                                 - 0.76400238018756 z_k1_s0007
                                 - 1.28400202871388 z_k1_s0010
                                 - 0.368001048934535 z_k1_s0011
                                 - 1.02000598115845 z_k1_s0012
                                 - 0.719001621132067 z_k1_s0013
                                 - 1.59100451648893 z_k1_s0014
                                 - 1.03800639722091 z_k1_s0015
                                 - 1.04400150856137 z_k1_s0016
                                 - 0.477001222328628 z_k1_s0017
                                 - 0.649001970350856 z_k1_s0020
                                 - 0.203999723814762 z_k1_s0021
                                 - 0.835998504182191 z_k1_s0022
                                 - 0.584999032998275 z_k1_s0023
                                 - 1.00700144367285 z_k1_s0024
                                 - 0.354999722383459 z_k1_s0025
                                 - 0.372000056367897 z_k1_s0026
                                 - 0.751998813905455 z_k1_s0027
                                 - 1.18200142219305 z_k1_s0030
                                 - 0.573001051664601 z_k1_s0031
                                 - 0.70200126666341 z_k1_s0032
                                 - 0.354000461103547 z_k1_s0033
                                 - 0.743001222214828 z_k1_s0034
                                 - 1.10900099440575 z_k1_s0035
                                 - 0.985003494497115 z_k1_s0036
                                 - 1.28000503341819 z_k1_s0037
                                 - 0.200999355887163 z_k1_s0040
                                 - 0.22099946533372 z_k1_s0041
                                 - 0.278999948828806 z_k1_s0042
                                 - 0.236999885091106 z_k1_s0043
                                 - 0.302999747964534 z_k1_s0044
                                 - 0.108000092711198 z_k1_s0045
                                 - 0.222999310178305 z_k1_s0046
                                 - 0.0890000717870039 z_k1_s0047
                                 - 0.621002825702802 z_k1_s0050
                                 - 0.887004469966179 z_k1_s0051
                                 - 0.436000729942275 z_k1_s0052
                                 - 0.427001079498194 z_k1_s0053
                                 - 0.594003874370559 z_k1_s0054
                                 - 0.739000637752861 z_k1_s0055
                                 - 1.23600229176806 z_k1_s0056
                                 - 0.449001183074791 z_k1_s0057
                                 - 0.35799862405804 z_k1_s0060
                                 - 0.230999358750562 z_k1_s0061
                                 - 0.402998684047768 z_k1_s0062
                                 - 0.314998792733738 z_k1_s0063
                                 - 0.547998380149727 z_k1_s0064
                                 - 0.839998407817661 z_k1_s0065
                                 - 0.583995987349972 z_k1_s0066
                                 - 0.788998068630243 z_k1_s0067
                                 - 0.234000801742389 z_k1_s0070
                                 - 0.598003458624645 z_k1_s0071
                                 - 0.858001918884989 z_k1_s0072
                                 - 0.311001086129918 z_k1_s0073
                                 - 0.872001970591812 z_k1_s0074
                                 - 0.686002533634346 z_k1_s0075
                                 - 0.354001369944194 z_k1_s0076
                                 - 0.462001706325172 z_k1_s0077
                                 - 0.412999744343934 z_k1_s0100
                                 - 0.330000279398394 z_k1_s0101
                                 - 0.374999619043495 z_k1_s0102
                                 - 0.175000078320347 z_k1_s0103
                                 - 0.450999869644373 z_k1_s0104
                                 - 0.289001222521811 z_k1_s0105
                                 - 0.191999999586307 z_k1_s0106
                                 - 0.336000703551172 z_k1_s0107
                                 - 0.18900006868619 z_k1_s0110
                                 - 0.201999941982093 z_k1_s0111
                                 - 0.298999017805363 z_k1_s0112
                                 - 0.363000644492417 z_k1_s0113
                                 - 0.251999815581644 z_k1_s0114
                                 - 0.758999842120821 z_k1_s0115
                                 - 0.536999288299045 z_k1_s0116
                                 - 0.158999978926494 z_k1_s0117
                                 - 0.169999963620285 z_k1_s0120
                                 - 0.244999081134901 z_k1_s0121
                                 - 0.488000173771241 z_k1_s0122
                                 - 0.19200008151 z_k1_s0123
                                 - 0.2880009238652 z_k1_s0124
                                 - 0.247999069883491 z_k1_s0125
                                 - 0.135000094886728 z_k1_s0126
                                 - 0.237999107388189 z_k1_s0127
                                 - 0.413001218679207 z_k1_s0130
                                 - 0.298000347898576 z_k1_s0131
                                 - 0.239000300916001 z_k1_s0132
                                 - 0.427000757277834 z_k1_s0133
                                 - 0.441999839204372 z_k1_s0134
                                 - 0.124000023336159 z_k1_s0135
                                 - 0.589000926497839 z_k1_s0136
                                 - 0.434000526577148 z_k1_s0137
                                 - 0.162999991580913 z_k1_s0140
                                 - 0.0979999572817887 z_k1_s0141
                                 - 0.438999843435756 z_k1_s0142
                                 - 0.192999115565422 z_k1_s0143
                                 - 0.180999329571664 z_k1_s0144
                                 - 0.403000485454482 z_k1_s0145
                                 - 0.137999925927421 z_k1_s0146
                                 - 0.0899999294526735 z_k1_s0147
                                 - 0.816999802029763 z_k1_s0150
                                 - 0.468999609299005 z_k1_s0151
                                 - 0.821997053989876 z_k1_s0152
                                 - 0.250999703058831 z_k1_s0153
                                 - 1.0869968177292 z_k1_s0154
                                 - 0.586001708998673 z_k1_s0155
                                 - 0.505997207685753 z_k1_s0156
                                 - 0.308999815963876 z_k1_s0157
                                 - 0.324999758836592 z_k1_s0160
                                 - 0.31399973589753 z_k1_s0161
                                 - 0.742998825522265 z_k1_s0162
                                 - 0.505998889198433 z_k1_s0163
                                 - 0.489999380923049 z_k1_s0164
                                 - 0.635998503933281 z_k1_s0165
                                 - 0.16999982942225 z_k1_s0166
                                 - 0.426001347821513 z_k1_s0167
                                 - 1.42599447017609 z_k1_s0170
                                 - 0.904000706496626 z_k1_s0171
                                 - 0.669998993197292 z_k1_s0172
                                 - 0.539999541095914 z_k1_s0173
                                 - 1.98299759305639 z_k1_s0174
                                 - 1.00899504013844 z_k1_s0175
                                 - 1.1879983155318 z_k1_s0176
                                 - 0.715998760390132 z_k1_s0177
                                 - 0.0489999613638881 z_k1_s1000
                                 - 0.123000028680597 z_k1_s1001
                                 - 0.0689998886063745 z_k1_s1002
                                 - 0.133999920644064 z_k1_s1003
                                 - 0.0629998889536599 z_k1_s1004
                                 - 0.0709998884906127 z_k1_s1005
                                 - 0.103000029838215 z_k1_s1006
                                 - 0.103999564381013 z_k1_s1007
                                 - 0.153999994673851 z_k1_s1010
                                 - 0.19399996786187 z_k1_s1011
                                 - 0.500999223630034 z_k1_s1012
                                 - 0.231000873110583 z_k1_s1013
                                 - 0.257999729162743 z_k1_s1014
                                 - 0.113000071106121 z_k1_s1015
                                 - 0.134000008079842 z_k1_s1016
                                 - 0.373000875828028 z_k1_s1017
                                 - 0.327998557727947 z_k1_s1020
                                 - 0.186999181195492 z_k1_s1021
                                 - 0.188999156224386 z_k1_s1022
                                 - 0.211998490058171 z_k1_s1023
                                 - 0.132999476416881 z_k1_s1024
                                 - 0.319997899615394 z_k1_s1025
                                 - 0.349999420041243 z_k1_s1026
                                 - 0.175999318536579 z_k1_s1027
                                 - 0.207999654587156 z_k1_s1030
                                 - 0.297000607064816 z_k1_s1031
                                 - 0.114000008904582 z_k1_s1032
                                 - 0.482000996967147 z_k1_s1033
                                 - 0.148000137221791 z_k1_s1034
                                 - 0.117000133267913 z_k1_s1035
                                 - 0.229000350430398 z_k1_s1036
                                 - 0.152000070106134 z_k1_s1037
                                 - 0.101000248322008 z_k1_s1040
                                 - 0.155000229068533 z_k1_s1041
                                 - 0.0569999796768874 z_k1_s1042
                                 - 0.1199999572145 z_k1_s1043
                                 - 0.0419999850250749 z_k1_s1044
                                 - 0.110000245113095 z_k1_s1045
                                 - 0.164999941169937 z_k1_s1046
                                 - 0.102999678942642 z_k1_s1047
                                 - 0.069999949340972 z_k1_s1050
                                 - 0.0939999354435865 z_k1_s1051
                                 - 0.180999535753507 z_k1_s1052
                                 - 0.139999412682559 z_k1_s1053
                                 - 0.306999736166887 z_k1_s1054
                                 - 0.163999277285328 z_k1_s1055
                                 - 0.0729999020413565 z_k1_s1056
                                 - 0.185999254421071 z_k1_s1057
                                 - 0.0739999416761968 z_k1_s1060
                                 - 0.0649998679886815 z_k1_s1061
                                 - 0.181999659528261 z_k1_s1062
                                 - 0.0559999401009311 z_k1_s1063
                                 - 0.0429999227632653 z_k1_s1064
                                 - 0.142999680415146 z_k1_s1065
                                 - 0.0569999077884981 z_k1_s1066
                                 - 0.108999904239279 z_k1_s1067
                                 - 0.356000533615984 z_k1_s1070
                                 - 0.241999675324465 z_k1_s1071
                                 - 0.472997458501259 z_k1_s1072
                                 - 0.370998346553228 z_k1_s1073
                                 - 0.506999493146397 z_k1_s1074
                                 - 0.459999353955647 z_k1_s1075
                                 - 0.814000179179998 z_k1_s1076
                                 - 0.272999350587201 z_k1_s1077
                                 - 0.106000085669547 z_k1_s1100
                                 - 0.337999787228941 z_k1_s1101
                                 - 0.115000101518855 z_k1_s1102
                                 - 0.25800015134659 z_k1_s1103
                                 - 0.151000164916086 z_k1_s1104
                                 - 0.34300080603488 z_k1_s1105
                                 - 0.12100011208506 z_k1_s1106
                                 - 0.0830000956657992 z_k1_s1107
                                 - 0.531996955948309 z_k1_s1110
                                 - 0.272999662389353 z_k1_s1111
                                 - 0.337998695723896 z_k1_s1112
                                 - 0.316998244465193 z_k1_s1113
                                 - 0.131999468611792 z_k1_s1114
                                 - 0.133999422960615 z_k1_s1115
                                 - 0.164999232365177 z_k1_s1116
                                 - 0.176999165257238 z_k1_s1117
                                 - 0.40399977075475 z_k1_s1120
                                 - 0.121000131092522 z_k1_s1121
                                 - 0.106000121039734 z_k1_s1122
                                 - 0.0930000623272845 z_k1_s1123
                                 - 0.162000108570108 z_k1_s1124
                                 - 0.388000760032447 z_k1_s1125
                                 - 0.128000035783756 z_k1_s1126
                                 - 0.0980000156781803 z_k1_s1127
                                 - 0.301000241318677 z_k1_s1130
                                 - 0.503002441063117 z_k1_s1131
                                 - 0.384000651739836 z_k1_s1132
                                 - 0.444000394173773 z_k1_s1133
                                 - 0.584001709988492 z_k1_s1134
                                 - 0.757002692461657 z_k1_s1135
                                 - 0.829002575062601 z_k1_s1136
                                 - 0.98999958376436 z_k1_s1137
                                 - 0.137000064466261 z_k1_s1140
                                 - 0.103000067576841 z_k1_s1141
                                 - 0.0880001459492262 z_k1_s1142
                                 - 0.165999830812909 z_k1_s1143
                                 - 0.253000669854253 z_k1_s1144
                                 - 0.408000270673316 z_k1_s1145
                                 - 0.140000141191868 z_k1_s1146
                                 - 0.245000362585874 z_k1_s1147
                                 - 0.257999384166406 z_k1_s1150
                                 - 0.152000033108889 z_k1_s1151
                                 - 0.478999294452357 z_k1_s1152
                                 - 0.153999956002427 z_k1_s1153
                                 - 0.129000019483202 z_k1_s1154
                                 - 0.269000024130863 z_k1_s1155
                                 - 0.0969999435865939 z_k1_s1156
                                 - 0.0990000301801316 z_k1_s1157
                                 - 0.260000787886557 z_k1_s1160
                                 - 0.454001931230193 z_k1_s1161
                                 - 0.238000676038353 z_k1_s1162
                                 - 0.317000826401743 z_k1_s1163
                                 - 0.360001516151658 z_k1_s1164
                                 - 0.749000918279504 z_k1_s1165
                                 - 0.379000607654465 z_k1_s1166
                                 - 0.69800153845173 z_k1_s1167
                                 - 0.515001013247718 z_k1_s1170
                                 - 0.539001140107945 z_k1_s1171
                                 - 0.238000364112356 z_k1_s1172
                                 - 0.510000329525691 z_k1_s1173
                                 - 0.214000237252129 z_k1_s1174
                                 - 0.189000289148048 z_k1_s1175
                                 - 0.642999586498593 z_k1_s1176
                                 - 0.307000649960371 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.888438 w_k1_ADM_s000
                                 + 0.727555 w_k1_AXP_s000
                                 + 0.868277 w_k1_DIS_s000
                                 + 0.959773 w_k1_FDX_s000
                                 + 0.710975 w_k1_IBM_s000
                                 + 1.01261 w_k1_INTC_s000
                                 + 0.804929 w_k1_KO_s000
                                 + 0.648163 w_k1_MAT_s000
                                 + 0.970741 w_k1_MSFT_s000
                                 + 0.870429 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.30297 w_k1_ADM_s000
                                 + 1.12703 w_k1_AXP_s000
                                 + 1.25773 w_k1_DIS_s000
                                 + 1.16568 w_k1_FDX_s000
                                 + 1.59529 w_k1_IBM_s000
                                 + 1.10442 w_k1_INTC_s000
                                 + 1.43695 w_k1_KO_s000 + 1.93007 w_k1_MAT_s000
                                 + 1.1403 w_k1_MSFT_s000 + 1.21437 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.219 w_k1_ADM_s000
                                 + 1.25774 w_k1_AXP_s000
                                 + 1.23282 w_k1_DIS_s000
                                 + 1.16095 w_k1_FDX_s000
                                 + 0.912225 w_k1_IBM_s000
                                 + 1.03653 w_k1_INTC_s000
                                 + 1.06879 w_k1_KO_s000
                                 + 0.874972 w_k1_MAT_s000
                                 + 0.971219 w_k1_MSFT_s000
                                 + 1.09759 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.03273 w_k1_ADM_s000
                                 + 1.00496 w_k1_AXP_s000 + 1.3023 w_k1_DIS_s000
                                 + 1.15208 w_k1_FDX_s000
                                 + 1.02916 w_k1_IBM_s000
                                 + 1.05974 w_k1_INTC_s000
                                 + 1.27728 w_k1_KO_s000 + 1.18177 w_k1_MAT_s000
                                 + 1.2121 w_k1_MSFT_s000 + 1.15989 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.712722 w_k1_ADM_s000
                                 + 0.500682 w_k1_AXP_s000
                                 + 0.556302 w_k1_DIS_s000
                                 + 0.679778 w_k1_FDX_s000
                                 + 0.735909 w_k1_IBM_s000
                                 + 0.658993 w_k1_INTC_s000
                                 + 0.704917 w_k1_KO_s000
                                 + 0.86072 w_k1_MAT_s000
                                 + 0.697133 w_k1_MSFT_s000
                                 + 0.753144 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.889848 w_k1_ADM_s000
                                 + 0.952853 w_k1_AXP_s000
                                 + 0.762724 w_k1_DIS_s000
                                 + 0.836863 w_k1_FDX_s000
                                 + 1.36762 w_k1_IBM_s000
                                 + 0.828908 w_k1_INTC_s000
                                 + 1.11961 w_k1_KO_s000 + 1.53231 w_k1_MAT_s000
                                 + 0.858532 w_k1_MSFT_s000
                                 + 1.08802 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.21231 w_k1_ADM_s000
                                 + 1.32589 w_k1_AXP_s000
                                 + 1.60638 w_k1_DIS_s000
                                 + 1.38923 w_k1_FDX_s000
                                 + 1.27736 w_k1_IBM_s000
                                 + 1.1266 w_k1_INTC_s000 + 1.26793 w_k1_KO_s000
                                 + 1.3516 w_k1_MAT_s000 + 1.4041 w_k1_MSFT_s000
                                 + 1.22233 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.912336 w_k1_ADM_s000
                                 + 0.682264 w_k1_AXP_s000
                                 + 0.866972 w_k1_DIS_s000
                                 + 0.872286 w_k1_FDX_s000
                                 + 1.01142 w_k1_IBM_s000
                                 + 0.833803 w_k1_INTC_s000
                                 + 1.01105 w_k1_KO_s000 + 1.03958 w_k1_MAT_s000
                                 + 0.864488 w_k1_MSFT_s000
                                 + 0.968901 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.02734 w_k1_ADM_s001
                                 + 0.760604 w_k1_AXP_s001
                                 + 0.8187 w_k1_DIS_s001
                                 + 0.808205 w_k1_FDX_s001
                                 + 0.914824 w_k1_IBM_s001
                                 + 0.882222 w_k1_INTC_s001
                                 + 0.920655 w_k1_KO_s001
                                 + 0.815853 w_k1_MAT_s001
                                 + 0.957538 w_k1_MSFT_s001
                                 + 1.02396 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.819537 w_k1_ADM_s001
                                 + 0.532213 w_k1_AXP_s001
                                 + 0.570774 w_k1_DIS_s001
                                 + 0.663511 w_k1_FDX_s001
                                 + 0.719093 w_k1_IBM_s001
                                 + 0.673531 w_k1_INTC_s001
                                 + 0.699601 w_k1_KO_s001
                                 + 0.68005 w_k1_MAT_s001
                                 + 0.710153 w_k1_MSFT_s001
                                 + 0.854681 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.909125 w_k1_ADM_s001
                                 + 0.981173 w_k1_AXP_s001
                                 + 1.26823 w_k1_DIS_s001
                                 + 0.91502 w_k1_FDX_s001 + 1.2614 w_k1_IBM_s001
                                 + 1.12796 w_k1_INTC_s001
                                 + 1.22163 w_k1_KO_s001 + 1.12562 w_k1_MAT_s001
                                 + 0.988683 w_k1_MSFT_s001
                                 + 1.12659 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.94655 w_k1_ADM_s001
                                 + 0.899048 w_k1_AXP_s001
                                 + 0.752073 w_k1_DIS_s001
                                 + 0.78497 w_k1_FDX_s001
                                 + 0.775196 w_k1_IBM_s001
                                 + 0.825845 w_k1_INTC_s001
                                 + 0.721768 w_k1_KO_s001
                                 + 0.849445 w_k1_MAT_s001
                                 + 0.887861 w_k1_MSFT_s001
                                 + 0.970847 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.37652 w_k1_ADM_s001
                                 + 1.38467 w_k1_AXP_s001 + 1.5953 w_k1_DIS_s001
                                 + 1.3774 w_k1_FDX_s001
                                 + 0.861411 w_k1_IBM_s001
                                 + 1.38004 w_k1_INTC_s001
                                 + 1.17044 w_k1_KO_s001
                                 + 0.993392 w_k1_MAT_s001
                                 + 1.43058 w_k1_MSFT_s001
                                 + 1.27944 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.15759 w_k1_ADM_s001
                                 + 0.985126 w_k1_AXP_s001
                                 + 0.921915 w_k1_DIS_s001
                                 + 1.02971 w_k1_FDX_s001
                                 + 0.670426 w_k1_IBM_s001
                                 + 1.12406 w_k1_INTC_s001
                                 + 0.717478 w_k1_KO_s001
                                 + 0.519977 w_k1_MAT_s001
                                 + 1.06068 w_k1_MSFT_s001
                                 + 0.976718 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.11988 w_k1_ADM_s001
                                 + 1.29872 w_k1_AXP_s001
                                 + 1.01231 w_k1_DIS_s001
                                 + 0.997718 w_k1_FDX_s001
                                 + 1.11253 w_k1_IBM_s001
                                 + 0.963212 w_k1_INTC_s001
                                 + 0.852293 w_k1_KO_s001 + 1.3789 w_k1_MAT_s001
                                 + 0.88989 w_k1_MSFT_s001
                                 + 1.09746 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.938173 w_k1_ADM_s001
                                 + 1.4835 w_k1_AXP_s001 + 1.36156 w_k1_DIS_s001
                                 + 1.03963 w_k1_FDX_s001
                                 + 1.92192 w_k1_IBM_s001
                                 + 0.924072 w_k1_INTC_s001
                                 + 1.26436 w_k1_KO_s001 + 2.18278 w_k1_MAT_s001
                                 + 0.904358 w_k1_MSFT_s001
                                 + 1.20923 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.943817 w_k1_ADM_s002
                                 + 1.13884 w_k1_AXP_s002
                                 + 0.858238 w_k1_DIS_s002
                                 + 0.872193 w_k1_FDX_s002
                                 + 1.06822 w_k1_IBM_s002
                                 + 0.727042 w_k1_INTC_s002
                                 + 0.906721 w_k1_KO_s002
                                 + 1.23424 w_k1_MAT_s002
                                 + 0.857447 w_k1_MSFT_s002
                                 + 0.96174 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.27286 w_k1_ADM_s002
                                 + 0.914384 w_k1_AXP_s002
                                 + 0.752091 w_k1_DIS_s002
                                 + 1.03523 w_k1_FDX_s002
                                 + 0.462851 w_k1_IBM_s002
                                 + 0.874953 w_k1_INTC_s002
                                 + 0.706109 w_k1_KO_s002
                                 + 0.418961 w_k1_MAT_s002
                                 + 1.0389 w_k1_MSFT_s002
                                 + 0.865083 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.13926 w_k1_ADM_s002
                                 + 1.09956 w_k1_AXP_s002
                                 + 1.07975 w_k1_DIS_s002
                                 + 1.10832 w_k1_FDX_s002
                                 + 0.789556 w_k1_IBM_s002
                                 + 0.980762 w_k1_INTC_s002
                                 + 0.92755 w_k1_KO_s002
                                 + 0.764308 w_k1_MAT_s002
                                 + 1.03983 w_k1_MSFT_s002
                                 + 0.89868 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.09747 w_k1_ADM_s002
                                 + 1.3228 w_k1_AXP_s002 + 1.26104 w_k1_DIS_s002
                                 + 1.12821 w_k1_FDX_s002
                                 + 1.14054 w_k1_IBM_s002
                                 + 1.07697 w_k1_INTC_s002
                                 + 1.08798 w_k1_KO_s002 + 1.08071 w_k1_MAT_s002
                                 + 1.14396 w_k1_MSFT_s002
                                 + 1.14935 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.74833 w_k1_ADM_s002
                                 + 0.6721 w_k1_AXP_s002
                                 + 0.724841 w_k1_DIS_s002
                                 + 0.777821 w_k1_FDX_s002
                                 + 0.836305 w_k1_IBM_s002
                                 + 0.801429 w_k1_INTC_s002
                                 + 0.769592 w_k1_KO_s002
                                 + 0.724985 w_k1_MAT_s002
                                 + 0.88868 w_k1_MSFT_s002
                                 + 0.857936 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.3183 w_k1_ADM_s002
                                 + 1.38924 w_k1_AXP_s002
                                 + 1.46812 w_k1_DIS_s002
                                 + 1.11829 w_k1_FDX_s002
                                 + 1.71298 w_k1_IBM_s002
                                 + 1.10042 w_k1_INTC_s002
                                 + 1.67621 w_k1_KO_s002 + 1.93489 w_k1_MAT_s002
                                 + 1.29268 w_k1_MSFT_s002
                                 + 1.25175 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.54443 w_k1_ADM_s002
                                 + 1.46646 w_k1_AXP_s002
                                 + 1.56072 w_k1_DIS_s002
                                 + 1.35391 w_k1_FDX_s002
                                 + 1.07326 w_k1_IBM_s002
                                 + 1.07447 w_k1_INTC_s002
                                 + 1.13752 w_k1_KO_s002 + 1.35823 w_k1_MAT_s002
                                 + 1.28913 w_k1_MSFT_s002
                                 + 1.17369 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.997524 w_k1_ADM_s002
                                 + 1.0089 w_k1_AXP_s002 + 1.04473 w_k1_DIS_s002
                                 + 0.902699 w_k1_FDX_s002
                                 + 1.40503 w_k1_IBM_s002
                                 + 0.938996 w_k1_INTC_s002
                                 + 1.05997 w_k1_KO_s002 + 1.35227 w_k1_MAT_s002
                                 + 1.08557 w_k1_MSFT_s002
                                 + 1.04989 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.03888 w_k1_ADM_s003
                                 + 0.70165 w_k1_AXP_s003
                                 + 0.827845 w_k1_DIS_s003
                                 + 0.872287 w_k1_FDX_s003
                                 + 0.946088 w_k1_IBM_s003
                                 + 1.04135 w_k1_INTC_s003
                                 + 0.900131 w_k1_KO_s003
                                 + 0.815212 w_k1_MAT_s003
                                 + 0.93259 w_k1_MSFT_s003
                                 + 0.92716 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.48358 w_k1_ADM_s003
                                 + 0.940096 w_k1_AXP_s003
                                 + 1.47419 w_k1_DIS_s003
                                 + 1.14595 w_k1_FDX_s003
                                 + 1.27047 w_k1_IBM_s003
                                 + 1.1296 w_k1_INTC_s003 + 1.19867 w_k1_KO_s003
                                 + 1.03492 w_k1_MAT_s003
                                 + 1.28176 w_k1_MSFT_s003
                                 + 1.12453 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.01384 w_k1_ADM_s003
                                 + 1.04331 w_k1_AXP_s003
                                 + 0.800842 w_k1_DIS_s003
                                 + 1.00323 w_k1_FDX_s003
                                 + 0.825961 w_k1_IBM_s003
                                 + 0.940523 w_k1_INTC_s003
                                 + 0.826938 w_k1_KO_s003
                                 + 0.978769 w_k1_MAT_s003
                                 + 0.910775 w_k1_MSFT_s003
                                 + 0.873261 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.03822 w_k1_ADM_s003
                                 + 1.00563 w_k1_AXP_s003
                                 + 1.04714 w_k1_DIS_s003
                                 + 1.03703 w_k1_FDX_s003
                                 + 0.753175 w_k1_IBM_s003
                                 + 1.0662 w_k1_INTC_s003
                                 + 0.857937 w_k1_KO_s003
                                 + 0.935607 w_k1_MAT_s003
                                 + 1.04757 w_k1_MSFT_s003
                                 + 0.892036 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.819197 w_k1_ADM_s003
                                 + 0.595431 w_k1_AXP_s003
                                 + 0.564834 w_k1_DIS_s003
                                 + 0.73061 w_k1_FDX_s003
                                 + 0.766267 w_k1_IBM_s003
                                 + 0.737961 w_k1_INTC_s003
                                 + 0.637816 w_k1_KO_s003
                                 + 0.694921 w_k1_MAT_s003
                                 + 0.752488 w_k1_MSFT_s003
                                 + 0.789131 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.7149 w_k1_ADM_s003
                                 + 1.11514 w_k1_AXP_s003
                                 + 1.39483 w_k1_DIS_s003 + 1.3219 w_k1_FDX_s003
                                 + 0.922943 w_k1_IBM_s003
                                 + 1.20008 w_k1_INTC_s003
                                 + 1.17164 w_k1_KO_s003
                                 + 0.911438 w_k1_MAT_s003
                                 + 1.36615 w_k1_MSFT_s003
                                 + 1.11885 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.2711 w_k1_ADM_s003
                                 + 1.30358 w_k1_AXP_s003
                                 + 1.11943 w_k1_DIS_s003
                                 + 1.22675 w_k1_FDX_s003
                                 + 1.19212 w_k1_IBM_s003
                                 + 1.23025 w_k1_INTC_s003
                                 + 1.03927 w_k1_KO_s003 + 1.67602 w_k1_MAT_s003
                                 + 1.17264 w_k1_MSFT_s003
                                 + 1.07711 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.11626 w_k1_ADM_s003
                                 + 0.929088 w_k1_AXP_s003
                                 + 0.914244 w_k1_DIS_s003
                                 + 0.903883 w_k1_FDX_s003
                                 + 1.30513 w_k1_IBM_s003
                                 + 0.961919 w_k1_INTC_s003
                                 + 1.06387 w_k1_KO_s003 + 1.2647 w_k1_MAT_s003
                                 + 1.1327 w_k1_MSFT_s003
                                 + 0.957101 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.16291 w_k1_ADM_s004
                                 + 1.07235 w_k1_AXP_s004
                                 + 0.812333 w_k1_DIS_s004
                                 + 0.803183 w_k1_FDX_s004
                                 + 1.27255 w_k1_IBM_s004
                                 + 0.964681 w_k1_INTC_s004
                                 + 0.974865 w_k1_KO_s004
                                 + 1.25915 w_k1_MAT_s004
                                 + 0.814244 w_k1_MSFT_s004
                                 + 0.971288 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.58098 w_k1_ADM_s004
                                 + 1.29507 w_k1_AXP_s004
                                 + 1.52464 w_k1_DIS_s004
                                 + 1.38744 w_k1_FDX_s004
                                 + 0.981792 w_k1_IBM_s004
                                 + 1.18456 w_k1_INTC_s004
                                 + 1.23443 w_k1_KO_s004 + 1.10882 w_k1_MAT_s004
                                 + 1.35494 w_k1_MSFT_s004
                                 + 1.18069 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.882963 w_k1_ADM_s004
                                 + 0.849248 w_k1_AXP_s004
                                 + 0.694974 w_k1_DIS_s004
                                 + 0.760309 w_k1_FDX_s004
                                 + 0.811931 w_k1_IBM_s004
                                 + 0.824411 w_k1_INTC_s004
                                 + 0.650694 w_k1_KO_s004
                                 + 0.707263 w_k1_MAT_s004
                                 + 0.691733 w_k1_MSFT_s004
                                 + 0.792334 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.18882 w_k1_ADM_s004
                                 + 0.973525 w_k1_AXP_s004
                                 + 1.23617 w_k1_DIS_s004
                                 + 1.09961 w_k1_FDX_s004
                                 + 0.891736 w_k1_IBM_s004
                                 + 1.07702 w_k1_INTC_s004 + 0.9965 w_k1_KO_s004
                                 + 0.931476 w_k1_MAT_s004
                                 + 1.10733 w_k1_MSFT_s004
                                 + 1.17025 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.14784 w_k1_ADM_s004
                                 + 1.17346 w_k1_AXP_s004
                                 + 0.942439 w_k1_DIS_s004
                                 + 0.966123 w_k1_FDX_s004
                                 + 0.907405 w_k1_IBM_s004
                                 + 1.14683 w_k1_INTC_s004
                                 + 0.964385 w_k1_KO_s004
                                 + 1.29718 w_k1_MAT_s004
                                 + 0.961605 w_k1_MSFT_s004
                                 + 1.04651 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.00828 w_k1_ADM_s004
                                 + 0.762585 w_k1_AXP_s004
                                 + 0.842907 w_k1_DIS_s004
                                 + 0.960982 w_k1_FDX_s004
                                 + 0.708812 w_k1_IBM_s004
                                 + 1.0014 w_k1_INTC_s004
                                 + 0.791302 w_k1_KO_s004
                                 + 0.545896 w_k1_MAT_s004
                                 + 0.897585 w_k1_MSFT_s004
                                 + 0.910748 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.17698 w_k1_ADM_s004
                                 + 1.44722 w_k1_AXP_s004
                                 + 1.16934 w_k1_DIS_s004
                                 + 1.06373 w_k1_FDX_s004 + 1.332 w_k1_IBM_s004
                                 + 1.29036 w_k1_INTC_s004
                                 + 1.31267 w_k1_KO_s004 + 1.46582 w_k1_MAT_s004
                                 + 1.15032 w_k1_MSFT_s004 + 1.2144 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.18137 w_k1_ADM_s004
                                 + 1.11312 w_k1_AXP_s004
                                 + 1.31716 w_k1_DIS_s004
                                 + 1.02358 w_k1_FDX_s004
                                 + 0.898904 w_k1_IBM_s004
                                 + 1.02197 w_k1_INTC_s004
                                 + 0.992017 w_k1_KO_s004
                                 + 0.964316 w_k1_MAT_s004
                                 + 1.14918 w_k1_MSFT_s004
                                 + 1.20925 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.858125 w_k1_ADM_s005
                                 + 0.732885 w_k1_AXP_s005
                                 + 0.636701 w_k1_DIS_s005
                                 + 0.709627 w_k1_FDX_s005
                                 + 0.761634 w_k1_IBM_s005
                                 + 0.606078 w_k1_INTC_s005
                                 + 0.825166 w_k1_KO_s005
                                 + 0.992836 w_k1_MAT_s005
                                 + 0.706897 w_k1_MSFT_s005
                                 + 0.787052 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.29886 w_k1_ADM_s005
                                 + 0.988516 w_k1_AXP_s005
                                 + 1.27467 w_k1_DIS_s005
                                 + 1.22244 w_k1_FDX_s005
                                 + 0.928106 w_k1_IBM_s005
                                 + 1.17983 w_k1_INTC_s005
                                 + 1.15674 w_k1_KO_s005
                                 + 0.575023 w_k1_MAT_s005
                                 + 1.12486 w_k1_MSFT_s005
                                 + 0.981189 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06267 w_k1_ADM_s005
                                 + 1.27715 w_k1_AXP_s005
                                 + 1.00012 w_k1_DIS_s005
                                 + 0.924651 w_k1_FDX_s005
                                 + 1.71413 w_k1_IBM_s005
                                 + 1.01762 w_k1_INTC_s005 + 1.3882 w_k1_KO_s005
                                 + 1.98184 w_k1_MAT_s005
                                 + 0.854959 w_k1_MSFT_s005
                                 + 1.00706 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.862736 w_k1_ADM_s005
                                 + 0.763811 w_k1_AXP_s005
                                 + 0.799673 w_k1_DIS_s005
                                 + 0.821678 w_k1_FDX_s005
                                 + 1.20716 w_k1_IBM_s005
                                 + 0.867356 w_k1_INTC_s005
                                 + 1.05321 w_k1_KO_s005 + 1.15722 w_k1_MAT_s005
                                 + 0.878668 w_k1_MSFT_s005
                                 + 0.981054 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.31332 w_k1_ADM_s005
                                 + 1.44879 w_k1_AXP_s005 + 1.6845 w_k1_DIS_s005
                                 + 1.36194 w_k1_FDX_s005
                                 + 1.21865 w_k1_IBM_s005
                                 + 1.19935 w_k1_INTC_s005
                                 + 1.42373 w_k1_KO_s005 + 1.43036 w_k1_MAT_s005
                                 + 1.37234 w_k1_MSFT_s005
                                 + 1.29436 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.02766 w_k1_ADM_s005
                                 + 0.782111 w_k1_AXP_s005
                                 + 0.936326 w_k1_DIS_s005
                                 + 0.975076 w_k1_FDX_s005
                                 + 0.724488 w_k1_IBM_s005
                                 + 0.881445 w_k1_INTC_s005
                                 + 0.866985 w_k1_KO_s005
                                 + 0.801246 w_k1_MAT_s005
                                 + 1.02353 w_k1_MSFT_s005
                                 + 0.875868 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.944096 w_k1_ADM_s005
                                 + 1.0938 w_k1_AXP_s005 + 1.01397 w_k1_DIS_s005
                                 + 1.03644 w_k1_FDX_s005
                                 + 1.18729 w_k1_IBM_s005
                                 + 0.864514 w_k1_INTC_s005
                                 + 1.12153 w_k1_KO_s005 + 1.38765 w_k1_MAT_s005
                                 + 0.923578 w_k1_MSFT_s005
                                 + 1.07981 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.923979 w_k1_ADM_s005
                                 + 0.823021 w_k1_AXP_s005
                                 + 0.763153 w_k1_DIS_s005
                                 + 0.896631 w_k1_FDX_s005
                                 + 0.88705 w_k1_IBM_s005
                                 + 0.998821 w_k1_INTC_s005
                                 + 0.862954 w_k1_KO_s005
                                 + 0.631335 w_k1_MAT_s005
                                 + 0.846346 w_k1_MSFT_s005
                                 + 0.901592 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.926254 w_k1_ADM_s006
                                 + 0.768472 w_k1_AXP_s006
                                 + 0.823965 w_k1_DIS_s006
                                 + 0.791682 w_k1_FDX_s006
                                 + 1.32349 w_k1_IBM_s006
                                 + 1.04446 w_k1_INTC_s006
                                 + 1.08276 w_k1_KO_s006 + 1.27089 w_k1_MAT_s006
                                 + 0.932893 w_k1_MSFT_s006
                                 + 0.969089 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.14174 w_k1_ADM_s006
                                 + 1.34952 w_k1_AXP_s006
                                 + 1.80053 w_k1_DIS_s006
                                 + 1.38775 w_k1_FDX_s006 + 1.1304 w_k1_IBM_s006
                                 + 1.44193 w_k1_INTC_s006
                                 + 1.43861 w_k1_KO_s006 + 1.17045 w_k1_MAT_s006
                                 + 1.51978 w_k1_MSFT_s006
                                 + 1.23716 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.15434 w_k1_ADM_s006
                                 + 0.970489 w_k1_AXP_s006
                                 + 0.722857 w_k1_DIS_s006
                                 + 0.912844 w_k1_FDX_s006
                                 + 0.988587 w_k1_IBM_s006
                                 + 1.00947 w_k1_INTC_s006
                                 + 0.908675 w_k1_KO_s006
                                 + 1.10369 w_k1_MAT_s006
                                 + 0.91217 w_k1_MSFT_s006
                                 + 0.900662 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.20256 w_k1_ADM_s006
                                 + 1.23359 w_k1_AXP_s006
                                 + 1.15628 w_k1_DIS_s006
                                 + 1.03037 w_k1_FDX_s006
                                 + 1.58393 w_k1_IBM_s006
                                 + 1.10224 w_k1_INTC_s006
                                 + 1.29776 w_k1_KO_s006 + 1.77679 w_k1_MAT_s006
                                 + 1.06364 w_k1_MSFT_s006
                                 + 1.21961 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.25525 w_k1_ADM_s006
                                 + 1.12763 w_k1_AXP_s006
                                 + 1.23347 w_k1_DIS_s006
                                 + 1.19281 w_k1_FDX_s006
                                 + 0.748889 w_k1_IBM_s006
                                 + 1.18852 w_k1_INTC_s006
                                 + 1.01874 w_k1_KO_s006
                                 + 0.669205 w_k1_MAT_s006
                                 + 1.19311 w_k1_MSFT_s006
                                 + 1.08589 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.08895 w_k1_ADM_s006
                                 + 1.19082 w_k1_AXP_s006
                                 + 1.10534 w_k1_DIS_s006 + 1.0612 w_k1_FDX_s006
                                 + 1.17148 w_k1_IBM_s006
                                 + 1.20728 w_k1_INTC_s006
                                 + 1.09959 w_k1_KO_s006
                                 + 0.871974 w_k1_MAT_s006
                                 + 1.13801 w_k1_MSFT_s006
                                 + 0.999246 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.980807 w_k1_ADM_s006
                                 + 0.764968 w_k1_AXP_s006
                                 + 0.978437 w_k1_DIS_s006
                                 + 0.996443 w_k1_FDX_s006
                                 + 0.905868 w_k1_IBM_s006
                                 + 1.0079 w_k1_INTC_s006
                                 + 0.951153 w_k1_KO_s006
                                 + 0.909001 w_k1_MAT_s006
                                 + 1.12366 w_k1_MSFT_s006
                                 + 1.00943 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.890254 w_k1_ADM_s006
                                 + 0.713611 w_k1_AXP_s006
                                 + 0.679344 w_k1_DIS_s006
                                 + 0.755691 w_k1_FDX_s006
                                 + 0.80936 w_k1_IBM_s006
                                 + 0.870338 w_k1_INTC_s006
                                 + 0.76502 w_k1_KO_s006
                                 + 0.642863 w_k1_MAT_s006
                                 + 0.879629 w_k1_MSFT_s006
                                 + 0.82792 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.96646 w_k1_ADM_s007
                                 + 0.958847 w_k1_AXP_s007
                                 + 0.864459 w_k1_DIS_s007
                                 + 1.01837 w_k1_FDX_s007
                                 + 0.638879 w_k1_IBM_s007
                                 + 0.982726 w_k1_INTC_s007
                                 + 0.834137 w_k1_KO_s007
                                 + 0.556063 w_k1_MAT_s007
                                 + 0.801078 w_k1_MSFT_s007
                                 + 0.905425 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.919789 w_k1_ADM_s007
                                 + 1.01449 w_k1_AXP_s007
                                 + 1.24606 w_k1_DIS_s007
                                 + 1.11964 w_k1_FDX_s007
                                 + 0.914587 w_k1_IBM_s007
                                 + 1.24893 w_k1_INTC_s007
                                 + 1.31044 w_k1_KO_s007 + 0.68994 w_k1_MAT_s007
                                 + 1.11375 w_k1_MSFT_s007
                                 + 1.12392 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.77731 w_k1_ADM_s007
                                 + 0.999808 w_k1_AXP_s007
                                 + 1.0295 w_k1_DIS_s007
                                 + 0.913001 w_k1_FDX_s007
                                 + 1.43885 w_k1_IBM_s007
                                 + 1.15974 w_k1_INTC_s007
                                 + 1.24069 w_k1_KO_s007 + 1.19244 w_k1_MAT_s007
                                 + 0.936973 w_k1_MSFT_s007
                                 + 1.03235 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.30604 w_k1_ADM_s007
                                 + 1.633 w_k1_AXP_s007 + 1.12499 w_k1_DIS_s007
                                 + 1.22512 w_k1_FDX_s007
                                 + 1.00625 w_k1_IBM_s007
                                 + 1.35266 w_k1_INTC_s007
                                 + 1.15497 w_k1_KO_s007 + 1.19546 w_k1_MAT_s007
                                 + 0.983844 w_k1_MSFT_s007
                                 + 1.09188 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.756955 w_k1_ADM_s007
                                 + 0.772968 w_k1_AXP_s007
                                 + 0.620457 w_k1_DIS_s007
                                 + 0.788283 w_k1_FDX_s007
                                 + 0.789022 w_k1_IBM_s007
                                 + 0.834495 w_k1_INTC_s007
                                 + 0.799074 w_k1_KO_s007
                                 + 0.752632 w_k1_MAT_s007
                                 + 0.730056 w_k1_MSFT_s007
                                 + 0.858054 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.911346 w_k1_ADM_s007
                                 + 0.933291 w_k1_AXP_s007
                                 + 0.932338 w_k1_DIS_s007
                                 + 0.962881 w_k1_FDX_s007
                                 + 0.940043 w_k1_IBM_s007
                                 + 1.08917 w_k1_INTC_s007
                                 + 0.941337 w_k1_KO_s007
                                 + 0.545665 w_k1_MAT_s007
                                 + 0.925592 w_k1_MSFT_s007
                                 + 1.04603 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.981986 w_k1_ADM_s007
                                 + 1.06283 w_k1_AXP_s007
                                 + 0.760111 w_k1_DIS_s007
                                 + 0.931271 w_k1_FDX_s007
                                 + 1.08655 w_k1_IBM_s007
                                 + 1.08957 w_k1_INTC_s007
                                 + 0.970395 w_k1_KO_s007
                                 + 1.09183 w_k1_MAT_s007
                                 + 0.935111 w_k1_MSFT_s007
                                 + 1.01607 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.915856 w_k1_ADM_s007
                                 + 1.30189 w_k1_AXP_s007
                                 + 1.67733 w_k1_DIS_s007
                                 + 1.41546 w_k1_FDX_s007
                                 + 1.11072 w_k1_IBM_s007
                                 + 1.64095 w_k1_INTC_s007
                                 + 1.43559 w_k1_KO_s007
                                 + 0.851459 w_k1_MAT_s007
                                 + 1.30003 w_k1_MSFT_s007
                                 + 1.27847 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.8788 w_k1_ADM_s010
                                 + 0.863298 w_k1_AXP_s010
                                 + 1.07187 w_k1_DIS_s010
                                 + 0.987075 w_k1_FDX_s010 + 1.169 w_k1_IBM_s010
                                 + 0.861385 w_k1_INTC_s010
                                 + 1.01598 w_k1_KO_s010 + 1.03791 w_k1_MAT_s010
                                 + 0.89789 w_k1_MSFT_s010
                                 + 0.931329 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.911804 w_k1_ADM_s010
                                 + 1.06075 w_k1_AXP_s010
                                 + 0.935857 w_k1_DIS_s010
                                 + 0.952946 w_k1_FDX_s010
                                 + 0.959547 w_k1_IBM_s010
                                 + 0.959476 w_k1_INTC_s010
                                 + 0.797279 w_k1_KO_s010 + 1.1089 w_k1_MAT_s010
                                 + 0.874138 w_k1_MSFT_s010
                                 + 0.972965 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.718597 w_k1_ADM_s010
                                 + 0.840552 w_k1_AXP_s010
                                 + 0.742739 w_k1_DIS_s010
                                 + 0.806331 w_k1_FDX_s010
                                 + 0.955348 w_k1_IBM_s010
                                 + 0.781466 w_k1_INTC_s010
                                 + 0.862829 w_k1_KO_s010
                                 + 0.94067 w_k1_MAT_s010
                                 + 0.804942 w_k1_MSFT_s010
                                 + 0.922585 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.718234 w_k1_ADM_s010
                                 + 0.720168 w_k1_AXP_s010
                                 + 0.766807 w_k1_DIS_s010
                                 + 0.838806 w_k1_FDX_s010
                                 + 1.11057 w_k1_IBM_s010
                                 + 0.951886 w_k1_INTC_s010
                                 + 0.832472 w_k1_KO_s010
                                 + 0.735407 w_k1_MAT_s010
                                 + 0.769808 w_k1_MSFT_s010
                                 + 0.812102 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.05125 w_k1_ADM_s010
                                 + 1.125 w_k1_AXP_s010 + 1.10958 w_k1_DIS_s010
                                 + 1.19085 w_k1_FDX_s010
                                 + 0.85002 w_k1_IBM_s010
                                 + 1.12234 w_k1_INTC_s010
                                 + 0.805302 w_k1_KO_s010
                                 + 0.68108 w_k1_MAT_s010
                                 + 1.0508 w_k1_MSFT_s010
                                 + 0.986872 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.775281 w_k1_ADM_s010
                                 + 0.592263 w_k1_AXP_s010
                                 + 0.565338 w_k1_DIS_s010
                                 + 0.77494 w_k1_FDX_s010
                                 + 0.737481 w_k1_IBM_s010
                                 + 0.684621 w_k1_INTC_s010
                                 + 0.583894 w_k1_KO_s010
                                 + 0.614905 w_k1_MAT_s010
                                 + 0.708639 w_k1_MSFT_s010
                                 + 0.813606 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.02836 w_k1_ADM_s010
                                 + 0.999106 w_k1_AXP_s010
                                 + 1.46946 w_k1_DIS_s010
                                 + 1.20497 w_k1_FDX_s010
                                 + 1.10566 w_k1_IBM_s010 + 1.078 w_k1_INTC_s010
                                 + 1.11406 w_k1_KO_s010 + 1.18511 w_k1_MAT_s010
                                 + 1.12192 w_k1_MSFT_s010 + 1.02 w_k1_PG_s010
                                 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.03146 w_k1_ADM_s010
                                 + 1.15011 w_k1_AXP_s010
                                 + 1.48429 w_k1_DIS_s010
                                 + 1.19385 w_k1_FDX_s010
                                 + 1.53419 w_k1_IBM_s010
                                 + 1.0973 w_k1_INTC_s010 + 1.22374 w_k1_KO_s010
                                 + 1.26315 w_k1_MAT_s010
                                 + 1.14156 w_k1_MSFT_s010
                                 + 1.13522 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.976376 w_k1_ADM_s011
                                 + 0.789029 w_k1_AXP_s011
                                 + 0.992319 w_k1_DIS_s011
                                 + 0.984598 w_k1_FDX_s011
                                 + 0.76815 w_k1_IBM_s011
                                 + 1.25281 w_k1_INTC_s011
                                 + 0.967508 w_k1_KO_s011
                                 + 0.559539 w_k1_MAT_s011
                                 + 1.09173 w_k1_MSFT_s011 + 0.908 w_k1_PG_s011
                                 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.713527 w_k1_ADM_s011
                                 + 0.557275 w_k1_AXP_s011
                                 + 0.620993 w_k1_DIS_s011
                                 + 0.780139 w_k1_FDX_s011
                                 + 0.569184 w_k1_IBM_s011
                                 + 0.826691 w_k1_INTC_s011
                                 + 0.680367 w_k1_KO_s011
                                 + 0.527323 w_k1_MAT_s011
                                 + 0.819884 w_k1_MSFT_s011
                                 + 0.841819 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.2298 w_k1_ADM_s011
                                 + 1.21036 w_k1_AXP_s011
                                 + 1.76884 w_k1_DIS_s011
                                 + 1.29522 w_k1_FDX_s011
                                 + 1.21342 w_k1_IBM_s011
                                 + 1.13002 w_k1_INTC_s011 + 1.285 w_k1_KO_s011
                                 + 0.946144 w_k1_MAT_s011
                                 + 1.39142 w_k1_MSFT_s011
                                 + 1.07281 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.846451 w_k1_ADM_s011
                                 + 0.79518 w_k1_AXP_s011
                                 + 0.773655 w_k1_DIS_s011
                                 + 0.817286 w_k1_FDX_s011
                                 + 0.873767 w_k1_IBM_s011
                                 + 0.870952 w_k1_INTC_s011
                                 + 0.818473 w_k1_KO_s011
                                 + 0.73652 w_k1_MAT_s011
                                 + 0.963214 w_k1_MSFT_s011
                                 + 0.878201 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.974849 w_k1_ADM_s011
                                 + 0.867689 w_k1_AXP_s011
                                 + 0.961104 w_k1_DIS_s011
                                 + 1.06226 w_k1_FDX_s011
                                 + 0.642022 w_k1_IBM_s011
                                 + 1.05489 w_k1_INTC_s011 + 0.7795 w_k1_KO_s011
                                 + 0.538756 w_k1_MAT_s011
                                 + 1.02771 w_k1_MSFT_s011
                                 + 0.87166 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.898442 w_k1_ADM_s011
                                 + 0.808846 w_k1_AXP_s011
                                 + 1.06118 w_k1_DIS_s011
                                 + 0.876234 w_k1_FDX_s011
                                 + 1.26804 w_k1_IBM_s011
                                 + 1.10363 w_k1_INTC_s011
                                 + 1.06205 w_k1_KO_s011
                                 + 0.795844 w_k1_MAT_s011
                                 + 1.02842 w_k1_MSFT_s011
                                 + 0.953283 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.24557 w_k1_ADM_s011
                                 + 1.5111 w_k1_AXP_s011 + 1.35129 w_k1_DIS_s011
                                 + 1.17485 w_k1_FDX_s011
                                 + 1.27624 w_k1_IBM_s011
                                 + 1.18953 w_k1_INTC_s011
                                 + 1.04776 w_k1_KO_s011 + 1.14386 w_k1_MAT_s011
                                 + 1.1607 w_k1_MSFT_s011 + 1.11461 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.13742 w_k1_ADM_s011
                                 + 0.929513 w_k1_AXP_s011
                                 + 1.19321 w_k1_DIS_s011
                                 + 1.19496 w_k1_FDX_s011
                                 + 0.682222 w_k1_IBM_s011
                                 + 1.20624 w_k1_INTC_s011
                                 + 0.834341 w_k1_KO_s011
                                 + 0.44453 w_k1_MAT_s011
                                 + 1.32037 w_k1_MSFT_s011
                                 + 0.937386 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.852335 w_k1_ADM_s012
                                 + 0.917855 w_k1_AXP_s012
                                 + 0.874261 w_k1_DIS_s012
                                 + 1.04824 w_k1_FDX_s012
                                 + 0.625375 w_k1_IBM_s012
                                 + 1.08624 w_k1_INTC_s012
                                 + 0.756299 w_k1_KO_s012
                                 + 0.422197 w_k1_MAT_s012
                                 + 0.888569 w_k1_MSFT_s012
                                 + 0.860868 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.0942 w_k1_ADM_s012
                                 + 1.25208 w_k1_AXP_s012
                                 + 1.40891 w_k1_DIS_s012
                                 + 1.17366 w_k1_FDX_s012
                                 + 1.59775 w_k1_IBM_s012
                                 + 0.866346 w_k1_INTC_s012
                                 + 1.43887 w_k1_KO_s012 + 1.46708 w_k1_MAT_s012
                                 + 0.960759 w_k1_MSFT_s012
                                 + 1.15348 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.995319 w_k1_ADM_s012
                                 + 1.17169 w_k1_AXP_s012
                                 + 1.32908 w_k1_DIS_s012
                                 + 1.28862 w_k1_FDX_s012
                                 + 0.930285 w_k1_IBM_s012
                                 + 1.09574 w_k1_INTC_s012
                                 + 1.16344 w_k1_KO_s012
                                 + 0.747919 w_k1_MAT_s012
                                 + 1.00221 w_k1_MSFT_s012
                                 + 1.12746 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.812013 w_k1_ADM_s012
                                 + 0.912402 w_k1_AXP_s012
                                 + 0.763324 w_k1_DIS_s012
                                 + 0.873968 w_k1_FDX_s012
                                 + 0.893458 w_k1_IBM_s012
                                 + 0.818438 w_k1_INTC_s012
                                 + 0.933384 w_k1_KO_s012
                                 + 1.05842 w_k1_MAT_s012
                                 + 0.760263 w_k1_MSFT_s012
                                 + 0.974514 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.703517 w_k1_ADM_s012
                                 + 0.623365 w_k1_AXP_s012
                                 + 0.67116 w_k1_DIS_s012
                                 + 0.781913 w_k1_FDX_s012
                                 + 0.715879 w_k1_IBM_s012
                                 + 0.851959 w_k1_INTC_s012
                                 + 0.662665 w_k1_KO_s012
                                 + 0.467677 w_k1_MAT_s012
                                 + 0.689075 w_k1_MSFT_s012
                                 + 0.816027 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.856863 w_k1_ADM_s012
                                 + 0.777759 w_k1_AXP_s012
                                 + 1.04532 w_k1_DIS_s012
                                 + 1.01777 w_k1_FDX_s012
                                 + 0.938969 w_k1_IBM_s012
                                 + 0.915244 w_k1_INTC_s012
                                 + 1.05765 w_k1_KO_s012
                                 + 0.757258 w_k1_MAT_s012
                                 + 0.937795 w_k1_MSFT_s012
                                 + 0.988334 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.696832 w_k1_ADM_s012
                                 + 0.904478 w_k1_AXP_s012
                                 + 0.805578 w_k1_DIS_s012
                                 + 0.807098 w_k1_FDX_s012
                                 + 1.05215 w_k1_IBM_s012
                                 + 0.805758 w_k1_INTC_s012
                                 + 0.919781 w_k1_KO_s012
                                 + 0.913205 w_k1_MAT_s012
                                 + 0.812635 w_k1_MSFT_s012
                                 + 0.957562 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.956107 w_k1_ADM_s012
                                 + 1.06539 w_k1_AXP_s012
                                 + 1.13111 w_k1_DIS_s012
                                 + 1.04732 w_k1_FDX_s012
                                 + 1.27481 w_k1_IBM_s012
                                 + 0.910973 w_k1_INTC_s012
                                 + 1.23182 w_k1_KO_s012 + 1.18295 w_k1_MAT_s012
                                 + 1.07404 w_k1_MSFT_s012
                                 + 1.04645 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.15649 w_k1_ADM_s013
                                 + 1.16836 w_k1_AXP_s013
                                 + 1.30564 w_k1_DIS_s013
                                 + 1.11721 w_k1_FDX_s013
                                 + 0.977178 w_k1_IBM_s013
                                 + 1.10932 w_k1_INTC_s013
                                 + 0.983522 w_k1_KO_s013
                                 + 0.956863 w_k1_MAT_s013
                                 + 1.03204 w_k1_MSFT_s013
                                 + 1.06171 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.2447 w_k1_ADM_s013
                                 + 1.16926 w_k1_AXP_s013
                                 + 0.898915 w_k1_DIS_s013
                                 + 1.13861 w_k1_FDX_s013
                                 + 0.73572 w_k1_IBM_s013
                                 + 0.93482 w_k1_INTC_s013
                                 + 0.794432 w_k1_KO_s013
                                 + 0.926839 w_k1_MAT_s013
                                 + 0.980304 w_k1_MSFT_s013
                                 + 0.934937 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.36229 w_k1_ADM_s013
                                 + 1.27248 w_k1_AXP_s013
                                 + 1.22908 w_k1_DIS_s013
                                 + 1.08608 w_k1_FDX_s013 + 1.708 w_k1_IBM_s013
                                 + 1.04452 w_k1_INTC_s013
                                 + 1.51265 w_k1_KO_s013 + 1.75385 w_k1_MAT_s013
                                 + 1.01896 w_k1_MSFT_s013 + 1.2126 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.950139 w_k1_ADM_s013
                                 + 0.914505 w_k1_AXP_s013
                                 + 0.644141 w_k1_DIS_s013
                                 + 0.801547 w_k1_FDX_s013
                                 + 1.00473 w_k1_IBM_s013
                                 + 0.789164 w_k1_INTC_s013
                                 + 0.843923 w_k1_KO_s013
                                 + 1.21108 w_k1_MAT_s013
                                 + 0.745219 w_k1_MSFT_s013
                                 + 0.944547 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.18868 w_k1_ADM_s013
                                 + 0.963545 w_k1_AXP_s013
                                 + 0.85109 w_k1_DIS_s013
                                 + 0.873192 w_k1_FDX_s013
                                 + 0.960335 w_k1_IBM_s013
                                 + 0.861048 w_k1_INTC_s013
                                 + 0.861374 w_k1_KO_s013
                                 + 0.847338 w_k1_MAT_s013
                                 + 0.876804 w_k1_MSFT_s013
                                 + 0.883034 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.86933 w_k1_ADM_s013
                                 + 1.16102 w_k1_AXP_s013
                                 + 1.17959 w_k1_DIS_s013
                                 + 1.23964 w_k1_FDX_s013 + 1.2083 w_k1_IBM_s013
                                 + 1.00289 w_k1_INTC_s013
                                 + 1.06075 w_k1_KO_s013 + 1.03095 w_k1_MAT_s013
                                 + 0.996574 w_k1_MSFT_s013
                                 + 1.07039 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.745825 w_k1_ADM_s013
                                 + 0.724537 w_k1_AXP_s013
                                 + 0.651006 w_k1_DIS_s013
                                 + 0.810177 w_k1_FDX_s013
                                 + 0.684257 w_k1_IBM_s013
                                 + 0.850714 w_k1_INTC_s013
                                 + 0.605463 w_k1_KO_s013
                                 + 0.659569 w_k1_MAT_s013
                                 + 0.779817 w_k1_MSFT_s013
                                 + 0.771536 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.39227 w_k1_ADM_s013
                                 + 1.55471 w_k1_AXP_s013
                                 + 1.46707 w_k1_DIS_s013
                                 + 1.18652 w_k1_FDX_s013
                                 + 1.29073 w_k1_IBM_s013
                                 + 1.1606 w_k1_INTC_s013 + 1.07062 w_k1_KO_s013
                                 + 1.34423 w_k1_MAT_s013
                                 + 1.15934 w_k1_MSFT_s013
                                 + 1.21705 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.944807 w_k1_ADM_s014
                                 + 0.905999 w_k1_AXP_s014
                                 + 0.974327 w_k1_DIS_s014
                                 + 1.00778 w_k1_FDX_s014
                                 + 0.851576 w_k1_IBM_s014
                                 + 0.963193 w_k1_INTC_s014
                                 + 1.01617 w_k1_KO_s014 + 1.13387 w_k1_MAT_s014
                                 + 0.932906 w_k1_MSFT_s014
                                 + 0.965726 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.02353 w_k1_ADM_s014
                                 + 1.1899 w_k1_AXP_s014 + 1.0461 w_k1_DIS_s014
                                 + 1.33323 w_k1_FDX_s014
                                 + 0.799619 w_k1_IBM_s014
                                 + 1.01203 w_k1_INTC_s014
                                 + 0.938221 w_k1_KO_s014
                                 + 1.10783 w_k1_MAT_s014
                                 + 1.14381 w_k1_MSFT_s014
                                 + 0.978156 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.42726 w_k1_ADM_s014
                                 + 1.32147 w_k1_AXP_s014
                                 + 1.40309 w_k1_DIS_s014
                                 + 1.18669 w_k1_FDX_s014
                                 + 1.34229 w_k1_IBM_s014
                                 + 1.18847 w_k1_INTC_s014
                                 + 1.31127 w_k1_KO_s014 + 1.19743 w_k1_MAT_s014
                                 + 1.26168 w_k1_MSFT_s014
                                 + 1.13846 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.833839 w_k1_ADM_s014
                                 + 0.736917 w_k1_AXP_s014
                                 + 0.647068 w_k1_DIS_s014
                                 + 0.757506 w_k1_FDX_s014
                                 + 0.823698 w_k1_IBM_s014
                                 + 0.755182 w_k1_INTC_s014
                                 + 0.844374 w_k1_KO_s014
                                 + 0.782845 w_k1_MAT_s014
                                 + 0.74635 w_k1_MSFT_s014
                                 + 0.860054 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.615103 w_k1_ADM_s014
                                 + 0.806055 w_k1_AXP_s014
                                 + 0.773705 w_k1_DIS_s014
                                 + 0.85594 w_k1_FDX_s014
                                 + 1.06286 w_k1_IBM_s014
                                 + 0.877002 w_k1_INTC_s014
                                 + 0.868416 w_k1_KO_s014
                                 + 1.15726 w_k1_MAT_s014
                                 + 0.932645 w_k1_MSFT_s014
                                 + 0.949548 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.05747 w_k1_ADM_s014
                                 + 1.18373 w_k1_AXP_s014
                                 + 1.07402 w_k1_DIS_s014
                                 + 1.03403 w_k1_FDX_s014
                                 + 1.07231 w_k1_IBM_s014
                                 + 0.886958 w_k1_INTC_s014
                                 + 1.23049 w_k1_KO_s014 + 1.07179 w_k1_MAT_s014
                                 + 0.927102 w_k1_MSFT_s014
                                 + 1.01946 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.04043 w_k1_ADM_s014
                                 + 0.789998 w_k1_AXP_s014
                                 + 0.72563 w_k1_DIS_s014
                                 + 0.903734 w_k1_FDX_s014
                                 + 0.648596 w_k1_IBM_s014
                                 + 0.713984 w_k1_INTC_s014
                                 + 0.835503 w_k1_KO_s014
                                 + 0.688864 w_k1_MAT_s014
                                 + 0.881893 w_k1_MSFT_s014
                                 + 0.912183 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.951691 w_k1_ADM_s014
                                 + 1.00422 w_k1_AXP_s014
                                 + 0.787291 w_k1_DIS_s014
                                 + 1.01555 w_k1_FDX_s014
                                 + 0.925179 w_k1_IBM_s014
                                 + 1.03225 w_k1_INTC_s014
                                 + 0.923892 w_k1_KO_s014
                                 + 0.879749 w_k1_MAT_s014
                                 + 0.861606 w_k1_MSFT_s014
                                 + 0.933007 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.735685 w_k1_ADM_s015
                                 + 0.918325 w_k1_AXP_s015
                                 + 0.747155 w_k1_DIS_s015
                                 + 0.831947 w_k1_FDX_s015
                                 + 1.08764 w_k1_IBM_s015
                                 + 0.894421 w_k1_INTC_s015
                                 + 0.922109 w_k1_KO_s015
                                 + 1.01287 w_k1_MAT_s015
                                 + 0.729645 w_k1_MSFT_s015
                                 + 0.942951 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.991617 w_k1_ADM_s015
                                 + 1.16173 w_k1_AXP_s015
                                 + 1.43116 w_k1_DIS_s015
                                 + 1.18358 w_k1_FDX_s015
                                 + 1.52999 w_k1_IBM_s015
                                 + 1.07191 w_k1_INTC_s015
                                 + 1.54586 w_k1_KO_s015 + 1.28015 w_k1_MAT_s015
                                 + 1.04991 w_k1_MSFT_s015
                                 + 1.25869 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.38547 w_k1_ADM_s015
                                 + 1.36722 w_k1_AXP_s015
                                 + 1.33698 w_k1_DIS_s015 + 1.2328 w_k1_FDX_s015
                                 + 1.21931 w_k1_IBM_s015
                                 + 1.09337 w_k1_INTC_s015
                                 + 1.33135 w_k1_KO_s015
                                 + 0.981979 w_k1_MAT_s015
                                 + 1.04002 w_k1_MSFT_s015
                                 + 1.14905 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.962449 w_k1_ADM_s015
                                 + 1.36584 w_k1_AXP_s015 + 1.2375 w_k1_DIS_s015
                                 + 0.983526 w_k1_FDX_s015
                                 + 1.86231 w_k1_IBM_s015
                                 + 1.2884 w_k1_INTC_s015 + 1.43168 w_k1_KO_s015
                                 + 1.50048 w_k1_MAT_s015
                                 + 1.09791 w_k1_MSFT_s015
                                 + 1.15077 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.939072 w_k1_ADM_s015
                                 + 0.826322 w_k1_AXP_s015
                                 + 0.743771 w_k1_DIS_s015
                                 + 0.895588 w_k1_FDX_s015
                                 + 0.735311 w_k1_IBM_s015
                                 + 0.861768 w_k1_INTC_s015
                                 + 0.777646 w_k1_KO_s015
                                 + 0.735739 w_k1_MAT_s015
                                 + 0.836839 w_k1_MSFT_s015
                                 + 0.946639 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.0898 w_k1_ADM_s015
                                 + 1.36603 w_k1_AXP_s015
                                 + 1.19553 w_k1_DIS_s015
                                 + 1.30355 w_k1_FDX_s015
                                 + 0.881825 w_k1_IBM_s015
                                 + 1.15635 w_k1_INTC_s015
                                 + 1.08603 w_k1_KO_s015
                                 + 0.826788 w_k1_MAT_s015
                                 + 1.11227 w_k1_MSFT_s015
                                 + 1.12463 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.06407 w_k1_ADM_s015
                                 + 0.885681 w_k1_AXP_s015
                                 + 1.15482 w_k1_DIS_s015
                                 + 1.07813 w_k1_FDX_s015
                                 + 0.98543 w_k1_IBM_s015
                                 + 0.958991 w_k1_INTC_s015
                                 + 0.972382 w_k1_KO_s015
                                 + 0.569088 w_k1_MAT_s015
                                 + 1.00639 w_k1_MSFT_s015
                                 + 1.03679 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.886889 w_k1_ADM_s015
                                 + 0.634135 w_k1_AXP_s015
                                 + 0.555001 w_k1_DIS_s015
                                 + 0.88913 w_k1_FDX_s015
                                 + 0.488166 w_k1_IBM_s015
                                 + 0.849421 w_k1_INTC_s015
                                 + 0.559558 w_k1_KO_s015
                                 + 0.396345 w_k1_MAT_s015
                                 + 0.790845 w_k1_MSFT_s015
                                 + 0.801074 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.879818 w_k1_ADM_s016
                                 + 0.792268 w_k1_AXP_s016
                                 + 0.669168 w_k1_DIS_s016
                                 + 0.709325 w_k1_FDX_s016
                                 + 0.986763 w_k1_IBM_s016
                                 + 0.880272 w_k1_INTC_s016
                                 + 0.845681 w_k1_KO_s016
                                 + 0.890641 w_k1_MAT_s016
                                 + 0.74751 w_k1_MSFT_s016
                                 + 0.943723 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.05613 w_k1_ADM_s016
                                 + 1.06847 w_k1_AXP_s016
                                 + 0.907046 w_k1_DIS_s016
                                 + 0.922236 w_k1_FDX_s016
                                 + 1.4446 w_k1_IBM_s016
                                 + 1.23392 w_k1_INTC_s016
                                 + 1.12415 w_k1_KO_s016 + 1.57914 w_k1_MAT_s016
                                 + 0.969153 w_k1_MSFT_s016
                                 + 1.17419 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.89289 w_k1_ADM_s016
                                 + 1.17321 w_k1_AXP_s016
                                 + 0.985199 w_k1_DIS_s016
                                 + 1.05542 w_k1_FDX_s016 + 0.9324 w_k1_IBM_s016
                                 + 1.13798 w_k1_INTC_s016 + 1.0382 w_k1_KO_s016
                                 + 1.1066 w_k1_MAT_s016
                                 + 1.12077 w_k1_MSFT_s016
                                 + 1.05294 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.11185 w_k1_ADM_s016
                                 + 1.48578 w_k1_AXP_s016
                                 + 1.19044 w_k1_DIS_s016
                                 + 1.13261 w_k1_FDX_s016
                                 + 1.36881 w_k1_IBM_s016
                                 + 1.17861 w_k1_INTC_s016
                                 + 1.37886 w_k1_KO_s016 + 1.89573 w_k1_MAT_s016
                                 + 1.05046 w_k1_MSFT_s016
                                 + 1.14214 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.14816 w_k1_ADM_s016
                                 + 1.2764 w_k1_AXP_s016 + 1.1477 w_k1_DIS_s016
                                 + 1.21062 w_k1_FDX_s016
                                 + 0.857876 w_k1_IBM_s016
                                 + 1.32123 w_k1_INTC_s016
                                 + 1.12572 w_k1_KO_s016 + 1.11306 w_k1_MAT_s016
                                 + 1.27782 w_k1_MSFT_s016
                                 + 1.10392 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.800465 w_k1_ADM_s016
                                 + 0.812879 w_k1_AXP_s016
                                 + 0.737981 w_k1_DIS_s016
                                 + 0.852264 w_k1_FDX_s016
                                 + 0.701154 w_k1_IBM_s016
                                 + 1.19243 w_k1_INTC_s016
                                 + 0.804654 w_k1_KO_s016
                                 + 0.746134 w_k1_MAT_s016
                                 + 0.954024 w_k1_MSFT_s016
                                 + 0.8747 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.16609 w_k1_ADM_s016
                                 + 1.28185 w_k1_AXP_s016
                                 + 1.86004 w_k1_DIS_s016
                                 + 1.23636 w_k1_FDX_s016
                                 + 1.29168 w_k1_IBM_s016
                                 + 1.2608 w_k1_INTC_s016 + 1.24255 w_k1_KO_s016
                                 + 1.11468 w_k1_MAT_s016
                                 + 1.27466 w_k1_MSFT_s016
                                 + 1.18633 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.947789 w_k1_ADM_s016
                                 + 0.970837 w_k1_AXP_s016
                                 + 0.947177 w_k1_DIS_s016
                                 + 0.883763 w_k1_FDX_s016
                                 + 1.16434 w_k1_IBM_s016
                                 + 1.12942 w_k1_INTC_s016
                                 + 1.07686 w_k1_KO_s016
                                 + 0.747512 w_k1_MAT_s016
                                 + 0.87129 w_k1_MSFT_s016
                                 + 1.05651 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.03878 w_k1_ADM_s017
                                 + 0.825963 w_k1_AXP_s017
                                 + 0.813349 w_k1_DIS_s017
                                 + 0.817828 w_k1_FDX_s017
                                 + 0.883905 w_k1_IBM_s017
                                 + 0.959025 w_k1_INTC_s017
                                 + 0.861497 w_k1_KO_s017
                                 + 0.90454 w_k1_MAT_s017
                                 + 0.941742 w_k1_MSFT_s017
                                 + 0.910447 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.22373 w_k1_ADM_s017
                                 + 1.05825 w_k1_AXP_s017
                                 + 1.28257 w_k1_DIS_s017
                                 + 1.21892 w_k1_FDX_s017
                                 + 0.762609 w_k1_IBM_s017
                                 + 1.28079 w_k1_INTC_s017
                                 + 0.980044 w_k1_KO_s017
                                 + 1.20182 w_k1_MAT_s017
                                 + 1.31561 w_k1_MSFT_s017
                                 + 0.989445 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.911877 w_k1_ADM_s017
                                 + 0.756873 w_k1_AXP_s017
                                 + 0.819207 w_k1_DIS_s017
                                 + 0.794977 w_k1_FDX_s017
                                 + 1.31471 w_k1_IBM_s017
                                 + 1.06677 w_k1_INTC_s017
                                 + 0.91244 w_k1_KO_s017 + 1.17803 w_k1_MAT_s017
                                 + 0.919294 w_k1_MSFT_s017
                                 + 0.861689 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.833523 w_k1_ADM_s017
                                 + 0.735607 w_k1_AXP_s017
                                 + 1.06159 w_k1_DIS_s017
                                 + 0.925904 w_k1_FDX_s017
                                 + 0.817279 w_k1_IBM_s017
                                 + 1.09073 w_k1_INTC_s017
                                 + 0.835286 w_k1_KO_s017
                                 + 0.851794 w_k1_MAT_s017
                                 + 0.951934 w_k1_MSFT_s017
                                 + 0.78169 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.50531 w_k1_ADM_s017
                                 + 1.47705 w_k1_AXP_s017
                                 + 1.49849 w_k1_DIS_s017
                                 + 1.32991 w_k1_FDX_s017
                                 + 1.21589 w_k1_IBM_s017
                                 + 1.39555 w_k1_INTC_s017
                                 + 1.28734 w_k1_KO_s017 + 1.2173 w_k1_MAT_s017
                                 + 1.32604 w_k1_MSFT_s017
                                 + 1.20044 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.778128 w_k1_ADM_s017
                                 + 0.501483 w_k1_AXP_s017
                                 + 0.601437 w_k1_DIS_s017
                                 + 0.646334 w_k1_FDX_s017
                                 + 0.642651 w_k1_IBM_s017
                                 + 0.868941 w_k1_INTC_s017
                                 + 0.608205 w_k1_KO_s017
                                 + 0.509794 w_k1_MAT_s017
                                 + 0.705439 w_k1_MSFT_s017
                                 + 0.735434 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.06108 w_k1_ADM_s017
                                 + 1.13771 w_k1_AXP_s017
                                 + 0.86428 w_k1_DIS_s017
                                 + 1.02588 w_k1_FDX_s017
                                 + 1.24039 w_k1_IBM_s017
                                 + 1.09712 w_k1_INTC_s017
                                 + 1.01372 w_k1_KO_s017 + 1.76489 w_k1_MAT_s017
                                 + 1.00654 w_k1_MSFT_s017
                                 + 0.985317 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.19334 w_k1_ADM_s017
                                 + 1.08815 w_k1_AXP_s017 + 1.2772 w_k1_DIS_s017
                                 + 0.997761 w_k1_FDX_s017
                                 + 1.23834 w_k1_IBM_s017
                                 + 1.25574 w_k1_INTC_s017
                                 + 1.21927 w_k1_KO_s017 + 1.2612 w_k1_MAT_s017
                                 + 1.1094 w_k1_MSFT_s017
                                 + 0.935708 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.06656 w_k1_ADM_s100
                                 + 1.47976 w_k1_AXP_s100
                                 + 1.70827 w_k1_DIS_s100
                                 + 1.13679 w_k1_FDX_s100
                                 + 1.82386 w_k1_IBM_s100
                                 + 0.766004 w_k1_INTC_s100
                                 + 1.30077 w_k1_KO_s100 + 1.96261 w_k1_MAT_s100
                                 + 0.975307 w_k1_MSFT_s100
                                 + 1.30842 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.717412 w_k1_ADM_s100
                                 + 0.707486 w_k1_AXP_s100
                                 + 0.666977 w_k1_DIS_s100
                                 + 0.833368 w_k1_FDX_s100
                                 + 0.767661 w_k1_IBM_s100
                                 + 0.827252 w_k1_INTC_s100
                                 + 0.681547 w_k1_KO_s100
                                 + 0.695543 w_k1_MAT_s100
                                 + 0.771662 w_k1_MSFT_s100
                                 + 0.923395 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.13604 w_k1_ADM_s100
                                 + 1.18945 w_k1_AXP_s100
                                 + 1.23387 w_k1_DIS_s100
                                 + 1.14615 w_k1_FDX_s100
                                 + 0.889773 w_k1_IBM_s100
                                 + 0.923609 w_k1_INTC_s100
                                 + 0.804159 w_k1_KO_s100
                                 + 0.97723 w_k1_MAT_s100
                                 + 1.0047 w_k1_MSFT_s100 + 1.10084 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.23373 w_k1_ADM_s100
                                 + 0.700281 w_k1_AXP_s100
                                 + 0.866821 w_k1_DIS_s100
                                 + 0.959968 w_k1_FDX_s100
                                 + 0.710691 w_k1_IBM_s100
                                 + 0.822631 w_k1_INTC_s100
                                 + 0.690645 w_k1_KO_s100
                                 + 0.480746 w_k1_MAT_s100
                                 + 0.932931 w_k1_MSFT_s100
                                 + 0.948579 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.28745 w_k1_ADM_s100
                                 + 1.12268 w_k1_AXP_s100
                                 + 1.36307 w_k1_DIS_s100
                                 + 1.10861 w_k1_FDX_s100
                                 + 0.987824 w_k1_IBM_s100
                                 + 0.887871 w_k1_INTC_s100
                                 + 0.872518 w_k1_KO_s100
                                 + 0.901285 w_k1_MAT_s100
                                 + 1.0259 w_k1_MSFT_s100 + 1.12939 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.821971 w_k1_ADM_s100
                                 + 0.639117 w_k1_AXP_s100
                                 + 0.861628 w_k1_DIS_s100
                                 + 0.781716 w_k1_FDX_s100
                                 + 0.965761 w_k1_IBM_s100
                                 + 0.74989 w_k1_INTC_s100
                                 + 0.796142 w_k1_KO_s100
                                 + 0.591284 w_k1_MAT_s100
                                 + 0.716945 w_k1_MSFT_s100
                                 + 0.825162 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.10329 w_k1_ADM_s100
                                 + 0.947882 w_k1_AXP_s100
                                 + 0.928575 w_k1_DIS_s100
                                 + 0.940168 w_k1_FDX_s100
                                 + 1.16548 w_k1_IBM_s100
                                 + 0.574889 w_k1_INTC_s100
                                 + 0.845147 w_k1_KO_s100
                                 + 1.11366 w_k1_MAT_s100
                                 + 0.816232 w_k1_MSFT_s100
                                 + 0.968772 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.33701 w_k1_ADM_s100
                                 + 1.47441 w_k1_AXP_s100
                                 + 1.67729 w_k1_DIS_s100
                                 + 1.19615 w_k1_FDX_s100
                                 + 1.33434 w_k1_IBM_s100
                                 + 0.79443 w_k1_INTC_s100
                                 + 1.17748 w_k1_KO_s100 + 1.32177 w_k1_MAT_s100
                                 + 1.03488 w_k1_MSFT_s100
                                 + 1.34573 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.07156 w_k1_ADM_s101
                                 + 1.67845 w_k1_AXP_s101
                                 + 1.80728 w_k1_DIS_s101
                                 + 1.54925 w_k1_FDX_s101
                                 + 1.26804 w_k1_IBM_s101
                                 + 1.25803 w_k1_INTC_s101
                                 + 1.30242 w_k1_KO_s101 + 1.39868 w_k1_MAT_s101
                                 + 1.39137 w_k1_MSFT_s101
                                 + 1.11088 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.13326 w_k1_ADM_s101
                                 + 1.07202 w_k1_AXP_s101
                                 + 1.05093 w_k1_DIS_s101
                                 + 0.99405 w_k1_FDX_s101
                                 + 1.22835 w_k1_IBM_s101
                                 + 1.16824 w_k1_INTC_s101
                                 + 0.999313 w_k1_KO_s101
                                 + 0.838882 w_k1_MAT_s101
                                 + 0.98367 w_k1_MSFT_s101
                                 + 0.95949 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.915552 w_k1_ADM_s101
                                 + 0.924344 w_k1_AXP_s101
                                 + 1.02795 w_k1_DIS_s101
                                 + 1.01568 w_k1_FDX_s101
                                 + 0.861573 w_k1_IBM_s101
                                 + 1.28017 w_k1_INTC_s101
                                 + 0.945966 w_k1_KO_s101
                                 + 0.765773 w_k1_MAT_s101
                                 + 1.17471 w_k1_MSFT_s101
                                 + 1.02097 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.32259 w_k1_ADM_s101
                                 + 1.78114 w_k1_AXP_s101
                                 + 1.15657 w_k1_DIS_s101
                                 + 1.29275 w_k1_FDX_s101
                                 + 1.11206 w_k1_IBM_s101
                                 + 1.34684 w_k1_INTC_s101
                                 + 1.09443 w_k1_KO_s101 + 1.20257 w_k1_MAT_s101
                                 + 1.05957 w_k1_MSFT_s101
                                 + 1.03951 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.823573 w_k1_ADM_s101
                                 + 0.857308 w_k1_AXP_s101
                                 + 0.7649 w_k1_DIS_s101
                                 + 0.792716 w_k1_FDX_s101
                                 + 1.1072 w_k1_IBM_s101
                                 + 1.04598 w_k1_INTC_s101 + 0.9976 w_k1_KO_s101
                                 + 0.943812 w_k1_MAT_s101
                                 + 0.913539 w_k1_MSFT_s101
                                 + 0.94054 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.709404 w_k1_ADM_s101
                                 + 0.657472 w_k1_AXP_s101
                                 + 0.571265 w_k1_DIS_s101
                                 + 0.72635 w_k1_FDX_s101
                                 + 0.691788 w_k1_IBM_s101
                                 + 0.848554 w_k1_INTC_s101
                                 + 0.663752 w_k1_KO_s101
                                 + 0.500358 w_k1_MAT_s101
                                 + 0.684209 w_k1_MSFT_s101
                                 + 0.77767 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.936728 w_k1_ADM_s101
                                 + 1.28107 w_k1_AXP_s101
                                 + 1.19213 w_k1_DIS_s101
                                 + 1.08256 w_k1_FDX_s101
                                 + 1.50979 w_k1_IBM_s101
                                 + 1.38469 w_k1_INTC_s101
                                 + 1.27358 w_k1_KO_s101 + 1.26776 w_k1_MAT_s101
                                 + 1.27862 w_k1_MSFT_s101
                                 + 1.16924 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.980969 w_k1_ADM_s101
                                 + 0.841822 w_k1_AXP_s101
                                 + 0.72008 w_k1_DIS_s101
                                 + 0.865621 w_k1_FDX_s101
                                 + 0.693261 w_k1_IBM_s101
                                 + 1.13486 w_k1_INTC_s101
                                 + 0.750342 w_k1_KO_s101
                                 + 0.665275 w_k1_MAT_s101
                                 + 0.964138 w_k1_MSFT_s101
                                 + 0.845549 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.894321 w_k1_ADM_s102
                                 + 0.824947 w_k1_AXP_s102
                                 + 0.961768 w_k1_DIS_s102
                                 + 0.99226 w_k1_FDX_s102
                                 + 0.94387 w_k1_IBM_s102
                                 + 1.28011 w_k1_INTC_s102
                                 + 1.07947 w_k1_KO_s102
                                 + 0.624096 w_k1_MAT_s102
                                 + 1.11029 w_k1_MSFT_s102
                                 + 0.919687 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.943753 w_k1_ADM_s102
                                 + 1.35162 w_k1_AXP_s102
                                 + 0.929912 w_k1_DIS_s102
                                 + 1.1589 w_k1_FDX_s102
                                 + 0.960375 w_k1_IBM_s102
                                 + 1.31709 w_k1_INTC_s102
                                 + 0.921956 w_k1_KO_s102
                                 + 1.36521 w_k1_MAT_s102
                                 + 1.2163 w_k1_MSFT_s102 + 1.00045 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.23642 w_k1_ADM_s102
                                 + 1.17049 w_k1_AXP_s102
                                 + 1.00545 w_k1_DIS_s102
                                 + 1.01055 w_k1_FDX_s102
                                 + 1.32434 w_k1_IBM_s102
                                 + 1.19443 w_k1_INTC_s102
                                 + 1.12584 w_k1_KO_s102 + 1.09473 w_k1_MAT_s102
                                 + 1.11971 w_k1_MSFT_s102
                                 + 0.888573 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.21576 w_k1_ADM_s102
                                 + 1.11826 w_k1_AXP_s102 + 1.317 w_k1_DIS_s102
                                 + 1.12114 w_k1_FDX_s102 + 1.507 w_k1_IBM_s102
                                 + 1.1177 w_k1_INTC_s102 + 1.34212 w_k1_KO_s102
                                 + 1.19406 w_k1_MAT_s102
                                 + 1.18272 w_k1_MSFT_s102
                                 + 1.06538 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.22817 w_k1_ADM_s102
                                 + 1.10053 w_k1_AXP_s102
                                 + 1.40099 w_k1_DIS_s102
                                 + 1.40745 w_k1_FDX_s102
                                 + 1.05774 w_k1_IBM_s102
                                 + 1.45772 w_k1_INTC_s102
                                 + 1.06514 w_k1_KO_s102
                                 + 0.773414 w_k1_MAT_s102
                                 + 1.51428 w_k1_MSFT_s102
                                 + 1.06763 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.556243 w_k1_ADM_s102
                                 + 0.696284 w_k1_AXP_s102
                                 + 0.61758 w_k1_DIS_s102
                                 + 0.78132 w_k1_FDX_s102
                                 + 0.669976 w_k1_IBM_s102
                                 + 0.996476 w_k1_INTC_s102
                                 + 0.721673 w_k1_KO_s102
                                 + 0.674229 w_k1_MAT_s102
                                 + 0.961556 w_k1_MSFT_s102
                                 + 0.807658 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.764424 w_k1_ADM_s102
                                 + 0.783017 w_k1_AXP_s102
                                 + 0.657365 w_k1_DIS_s102
                                 + 0.785963 w_k1_FDX_s102
                                 + 1.02604 w_k1_IBM_s102
                                 + 0.965371 w_k1_INTC_s102
                                 + 0.845756 w_k1_KO_s102
                                 + 1.02841 w_k1_MAT_s102
                                 + 0.892735 w_k1_MSFT_s102
                                 + 0.839384 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.35897 w_k1_ADM_s102
                                 + 1.62058 w_k1_AXP_s102
                                 + 1.63171 w_k1_DIS_s102
                                 + 1.51008 w_k1_FDX_s102
                                 + 1.19965 w_k1_IBM_s102
                                 + 1.39873 w_k1_INTC_s102
                                 + 1.48541 w_k1_KO_s102 + 1.69101 w_k1_MAT_s102
                                 + 1.67566 w_k1_MSFT_s102
                                 + 1.30322 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.38918 w_k1_ADM_s103
                                 + 0.98674 w_k1_AXP_s103
                                 + 1.27857 w_k1_DIS_s103
                                 + 1.23773 w_k1_FDX_s103
                                 + 0.889661 w_k1_IBM_s103
                                 + 1.24005 w_k1_INTC_s103
                                 + 1.27112 w_k1_KO_s103 + 0.98588 w_k1_MAT_s103
                                 + 1.30801 w_k1_MSFT_s103
                                 + 1.03359 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.720036 w_k1_ADM_s103
                                 + 0.784476 w_k1_AXP_s103
                                 + 0.822651 w_k1_DIS_s103
                                 + 0.917856 w_k1_FDX_s103
                                 + 0.88772 w_k1_IBM_s103
                                 + 0.914981 w_k1_INTC_s103
                                 + 0.945035 w_k1_KO_s103
                                 + 0.895156 w_k1_MAT_s103
                                 + 0.9251 w_k1_MSFT_s103
                                 + 0.788046 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.885392 w_k1_ADM_s103
                                 + 0.659731 w_k1_AXP_s103
                                 + 0.64935 w_k1_DIS_s103
                                 + 0.767115 w_k1_FDX_s103
                                 + 0.879653 w_k1_IBM_s103
                                 + 0.668511 w_k1_INTC_s103
                                 + 0.787978 w_k1_KO_s103
                                 + 0.976345 w_k1_MAT_s103
                                 + 0.743811 w_k1_MSFT_s103
                                 + 0.721222 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.16325 w_k1_ADM_s103
                                 + 0.898135 w_k1_AXP_s103
                                 + 0.963998 w_k1_DIS_s103
                                 + 1.01352 w_k1_FDX_s103
                                 + 1.15537 w_k1_IBM_s103
                                 + 0.906248 w_k1_INTC_s103
                                 + 1.05866 w_k1_KO_s103
                                 + 0.987911 w_k1_MAT_s103
                                 + 0.969596 w_k1_MSFT_s103
                                 + 0.936221 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.11547 w_k1_ADM_s103
                                 + 1.20992 w_k1_AXP_s103 + 1.3925 w_k1_DIS_s103
                                 + 1.11108 w_k1_FDX_s103
                                 + 1.33973 w_k1_IBM_s103
                                 + 0.935575 w_k1_INTC_s103
                                 + 1.48097 w_k1_KO_s103 + 1.61429 w_k1_MAT_s103
                                 + 0.956199 w_k1_MSFT_s103
                                 + 1.10313 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.733998 w_k1_ADM_s103
                                 + 0.664515 w_k1_AXP_s103
                                 + 0.641805 w_k1_DIS_s103
                                 + 0.729692 w_k1_FDX_s103
                                 + 0.760312 w_k1_IBM_s103
                                 + 0.861811 w_k1_INTC_s103
                                 + 0.815152 w_k1_KO_s103
                                 + 0.697834 w_k1_MAT_s103
                                 + 0.82411 w_k1_MSFT_s103
                                 + 0.662576 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.24253 w_k1_ADM_s103
                                 + 1.53855 w_k1_AXP_s103
                                 + 1.19631 w_k1_DIS_s103
                                 + 1.31151 w_k1_FDX_s103
                                 + 1.29058 w_k1_IBM_s103
                                 + 1.20998 w_k1_INTC_s103
                                 + 1.12933 w_k1_KO_s103 + 1.63301 w_k1_MAT_s103
                                 + 1.2216 w_k1_MSFT_s103 + 1.092 w_k1_PG_s103
                                 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.859461 w_k1_ADM_s103
                                 + 0.906426 w_k1_AXP_s103
                                 + 0.932309 w_k1_DIS_s103
                                 + 1.06917 w_k1_FDX_s103
                                 + 0.69125 w_k1_IBM_s103
                                 + 1.1939 w_k1_INTC_s103
                                 + 0.934367 w_k1_KO_s103
                                 + 0.878886 w_k1_MAT_s103
                                 + 1.10273 w_k1_MSFT_s103
                                 + 0.904728 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.669662 w_k1_ADM_s104
                                 + 0.475051 w_k1_AXP_s104
                                 + 0.543158 w_k1_DIS_s104
                                 + 0.669091 w_k1_FDX_s104
                                 + 0.744287 w_k1_IBM_s104
                                 + 0.960506 w_k1_INTC_s104
                                 + 0.80045 w_k1_KO_s104
                                 + 0.490568 w_k1_MAT_s104
                                 + 0.791994 w_k1_MSFT_s104
                                 + 0.734137 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.48426 w_k1_ADM_s104
                                 + 1.44075 w_k1_AXP_s104
                                 + 1.67183 w_k1_DIS_s104
                                 + 1.31209 w_k1_FDX_s104
                                 + 1.26457 w_k1_IBM_s104
                                 + 1.69312 w_k1_INTC_s104
                                 + 1.50278 w_k1_KO_s104 + 0.76163 w_k1_MAT_s104
                                 + 1.51854 w_k1_MSFT_s104
                                 + 1.13114 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.756496 w_k1_ADM_s104
                                 + 0.675799 w_k1_AXP_s104
                                 + 0.953823 w_k1_DIS_s104
                                 + 0.817629 w_k1_FDX_s104
                                 + 1.16118 w_k1_IBM_s104
                                 + 0.902809 w_k1_INTC_s104
                                 + 0.996123 w_k1_KO_s104
                                 + 0.94812 w_k1_MAT_s104
                                 + 0.955127 w_k1_MSFT_s104
                                 + 0.877805 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.879448 w_k1_ADM_s104
                                 + 0.923713 w_k1_AXP_s104
                                 + 0.924727 w_k1_DIS_s104
                                 + 0.955235 w_k1_FDX_s104
                                 + 0.774209 w_k1_IBM_s104
                                 + 1.17391 w_k1_INTC_s104
                                 + 0.877524 w_k1_KO_s104
                                 + 0.615959 w_k1_MAT_s104
                                 + 1.17622 w_k1_MSFT_s104
                                 + 0.940971 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.857242 w_k1_ADM_s104
                                 + 0.843167 w_k1_AXP_s104
                                 + 1.02268 w_k1_DIS_s104
                                 + 0.841952 w_k1_FDX_s104
                                 + 1.45753 w_k1_IBM_s104
                                 + 1.25536 w_k1_INTC_s104
                                 + 1.29502 w_k1_KO_s104
                                 + 0.947118 w_k1_MAT_s104
                                 + 0.974677 w_k1_MSFT_s104
                                 + 0.999028 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.13015 w_k1_ADM_s104
                                 + 0.748036 w_k1_AXP_s104
                                 + 0.790972 w_k1_DIS_s104
                                 + 0.87152 w_k1_FDX_s104
                                 + 0.916609 w_k1_IBM_s104
                                 + 1.07379 w_k1_INTC_s104
                                 + 0.87404 w_k1_KO_s104
                                 + 0.493798 w_k1_MAT_s104
                                 + 0.919696 w_k1_MSFT_s104
                                 + 0.807328 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.04909 w_k1_ADM_s104
                                 + 1.06693 w_k1_AXP_s104
                                 + 0.946885 w_k1_DIS_s104
                                 + 1.13498 w_k1_FDX_s104
                                 + 0.844744 w_k1_IBM_s104
                                 + 1.33451 w_k1_INTC_s104
                                 + 0.939053 w_k1_KO_s104
                                 + 0.577421 w_k1_MAT_s104
                                 + 1.16503 w_k1_MSFT_s104
                                 + 1.02154 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.40628 w_k1_ADM_s104
                                 + 1.06682 w_k1_AXP_s104
                                 + 1.23377 w_k1_DIS_s104
                                 + 1.06768 w_k1_FDX_s104
                                 + 1.29402 w_k1_IBM_s104
                                 + 1.05462 w_k1_INTC_s104
                                 + 1.22178 w_k1_KO_s104
                                 + 0.731396 w_k1_MAT_s104
                                 + 1.12271 w_k1_MSFT_s104
                                 + 0.971419 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.751746 w_k1_ADM_s105
                                 + 1.10897 w_k1_AXP_s105
                                 + 0.927517 w_k1_DIS_s105
                                 + 0.91214 w_k1_FDX_s105
                                 + 1.25022 w_k1_IBM_s105
                                 + 1.22388 w_k1_INTC_s105 + 1.0802 w_k1_KO_s105
                                 + 1.96734 w_k1_MAT_s105
                                 + 1.05055 w_k1_MSFT_s105
                                 + 1.14147 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.715879 w_k1_ADM_s105
                                 + 0.863756 w_k1_AXP_s105
                                 + 0.83264 w_k1_DIS_s105
                                 + 0.82556 w_k1_FDX_s105
                                 + 1.17083 w_k1_IBM_s105
                                 + 1.21274 w_k1_INTC_s105
                                 + 0.920359 w_k1_KO_s105
                                 + 0.954913 w_k1_MAT_s105
                                 + 0.859379 w_k1_MSFT_s105
                                 + 0.896428 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.967318 w_k1_ADM_s105
                                 + 1.11306 w_k1_AXP_s105
                                 + 1.25402 w_k1_DIS_s105
                                 + 1.16495 w_k1_FDX_s105 + 1.2704 w_k1_IBM_s105
                                 + 1.43728 w_k1_INTC_s105
                                 + 1.08384 w_k1_KO_s105 + 1.34343 w_k1_MAT_s105
                                 + 1.19694 w_k1_MSFT_s105
                                 + 1.04554 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.864478 w_k1_ADM_s105
                                 + 1.28573 w_k1_AXP_s105
                                 + 1.11165 w_k1_DIS_s105
                                 + 1.00226 w_k1_FDX_s105
                                 + 1.51968 w_k1_IBM_s105
                                 + 1.24519 w_k1_INTC_s105
                                 + 1.16993 w_k1_KO_s105 + 1.89148 w_k1_MAT_s105
                                 + 1.01585 w_k1_MSFT_s105
                                 + 1.17624 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.743737 w_k1_ADM_s105
                                 + 0.671776 w_k1_AXP_s105
                                 + 0.714064 w_k1_DIS_s105
                                 + 0.711779 w_k1_FDX_s105
                                 + 0.799709 w_k1_IBM_s105
                                 + 0.8733 w_k1_INTC_s105
                                 + 0.856567 w_k1_KO_s105
                                 + 0.829359 w_k1_MAT_s105
                                 + 0.854657 w_k1_MSFT_s105
                                 + 0.938198 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.17552 w_k1_ADM_s105
                                 + 1.39463 w_k1_AXP_s105
                                 + 1.07039 w_k1_DIS_s105
                                 + 1.08139 w_k1_FDX_s105
                                 + 0.992488 w_k1_IBM_s105
                                 + 1.25816 w_k1_INTC_s105
                                 + 0.934265 w_k1_KO_s105
                                 + 1.25361 w_k1_MAT_s105
                                 + 1.11686 w_k1_MSFT_s105
                                 + 1.11199 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.978638 w_k1_ADM_s105
                                 + 0.932148 w_k1_AXP_s105
                                 + 1.37913 w_k1_DIS_s105
                                 + 1.32812 w_k1_FDX_s105
                                 + 0.813858 w_k1_IBM_s105
                                 + 1.37237 w_k1_INTC_s105
                                 + 1.00972 w_k1_KO_s105
                                 + 0.731463 w_k1_MAT_s105
                                 + 1.36913 w_k1_MSFT_s105
                                 + 1.07346 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.726668 w_k1_ADM_s105
                                 + 0.983168 w_k1_AXP_s105
                                 + 1.07961 w_k1_DIS_s105
                                 + 0.983626 w_k1_FDX_s105
                                 + 0.816627 w_k1_IBM_s105
                                 + 1.12544 w_k1_INTC_s105
                                 + 0.871881 w_k1_KO_s105
                                 + 0.898195 w_k1_MAT_s105
                                 + 1.08871 w_k1_MSFT_s105
                                 + 0.973932 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.742956 w_k1_ADM_s106
                                 + 0.581904 w_k1_AXP_s106
                                 + 0.748947 w_k1_DIS_s106
                                 + 0.800682 w_k1_FDX_s106
                                 + 0.68672 w_k1_IBM_s106
                                 + 0.691319 w_k1_INTC_s106
                                 + 0.799032 w_k1_KO_s106
                                 + 0.834405 w_k1_MAT_s106
                                 + 0.899337 w_k1_MSFT_s106
                                 + 0.850294 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.912162 w_k1_ADM_s106
                                 + 0.724173 w_k1_AXP_s106
                                 + 0.836898 w_k1_DIS_s106
                                 + 0.809553 w_k1_FDX_s106
                                 + 1.02407 w_k1_IBM_s106
                                 + 0.693144 w_k1_INTC_s106
                                 + 0.859859 w_k1_KO_s106
                                 + 1.37726 w_k1_MAT_s106
                                 + 0.85529 w_k1_MSFT_s106
                                 + 1.00981 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.938728 w_k1_ADM_s106
                                 + 0.941303 w_k1_AXP_s106
                                 + 1.0582 w_k1_DIS_s106 + 1.02143 w_k1_FDX_s106
                                 + 1.01742 w_k1_IBM_s106
                                 + 0.881788 w_k1_INTC_s106
                                 + 0.972327 w_k1_KO_s106
                                 + 0.946987 w_k1_MAT_s106
                                 + 0.916698 w_k1_MSFT_s106
                                 + 1.00914 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.633696 w_k1_ADM_s106
                                 + 0.445402 w_k1_AXP_s106
                                 + 0.611467 w_k1_DIS_s106
                                 + 0.663491 w_k1_FDX_s106
                                 + 0.82921 w_k1_IBM_s106
                                 + 0.807309 w_k1_INTC_s106
                                 + 0.603141 w_k1_KO_s106
                                 + 0.376225 w_k1_MAT_s106
                                 + 0.714486 w_k1_MSFT_s106
                                 + 0.742639 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.938851 w_k1_ADM_s106
                                 + 0.858552 w_k1_AXP_s106
                                 + 0.562137 w_k1_DIS_s106
                                 + 0.750126 w_k1_FDX_s106
                                 + 0.979966 w_k1_IBM_s106
                                 + 0.970222 w_k1_INTC_s106
                                 + 0.742755 w_k1_KO_s106
                                 + 1.01493 w_k1_MAT_s106
                                 + 0.897888 w_k1_MSFT_s106
                                 + 0.856616 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.08168 w_k1_ADM_s106
                                 + 1.13513 w_k1_AXP_s106 + 1.0407 w_k1_DIS_s106
                                 + 0.972725 w_k1_FDX_s106
                                 + 1.42553 w_k1_IBM_s106
                                 + 1.04902 w_k1_INTC_s106
                                 + 1.18146 w_k1_KO_s106 + 1.75772 w_k1_MAT_s106
                                 + 1.07936 w_k1_MSFT_s106
                                 + 1.26013 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.09322 w_k1_ADM_s106
                                 + 0.621864 w_k1_AXP_s106
                                 + 0.961087 w_k1_DIS_s106
                                 + 0.863919 w_k1_FDX_s106
                                 + 0.878126 w_k1_IBM_s106
                                 + 1.00878 w_k1_INTC_s106
                                 + 1.06717 w_k1_KO_s106
                                 + 0.712288 w_k1_MAT_s106
                                 + 1.08086 w_k1_MSFT_s106
                                 + 1.06273 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.33762 w_k1_ADM_s106
                                 + 1.45542 w_k1_AXP_s106
                                 + 1.91699 w_k1_DIS_s106
                                 + 1.29534 w_k1_FDX_s106
                                 + 1.61098 w_k1_IBM_s106
                                 + 1.28752 w_k1_INTC_s106
                                 + 1.53023 w_k1_KO_s106 + 1.73955 w_k1_MAT_s106
                                 + 1.45123 w_k1_MSFT_s106
                                 + 1.34242 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.851305 w_k1_ADM_s107
                                 + 0.912146 w_k1_AXP_s107
                                 + 0.821368 w_k1_DIS_s107
                                 + 0.81431 w_k1_FDX_s107 + 1.0296 w_k1_IBM_s107
                                 + 0.719894 w_k1_INTC_s107
                                 + 0.877886 w_k1_KO_s107
                                 + 0.983491 w_k1_MAT_s107
                                 + 0.786739 w_k1_MSFT_s107
                                 + 0.893202 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.67605 w_k1_ADM_s107
                                 + 1.85787 w_k1_AXP_s107
                                 + 1.47623 w_k1_DIS_s107
                                 + 1.49542 w_k1_FDX_s107
                                 + 0.936197 w_k1_IBM_s107
                                 + 1.06351 w_k1_INTC_s107
                                 + 1.24069 w_k1_KO_s107 + 1.36343 w_k1_MAT_s107
                                 + 1.33015 w_k1_MSFT_s107
                                 + 1.22322 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.997171 w_k1_ADM_s107
                                 + 0.930758 w_k1_AXP_s107
                                 + 0.88217 w_k1_DIS_s107
                                 + 1.04627 w_k1_FDX_s107
                                 + 0.848211 w_k1_IBM_s107
                                 + 0.847067 w_k1_INTC_s107
                                 + 0.815792 w_k1_KO_s107 + 0.7952 w_k1_MAT_s107
                                 + 0.986534 w_k1_MSFT_s107
                                 + 0.867828 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.973053 w_k1_ADM_s107
                                 + 0.765227 w_k1_AXP_s107
                                 + 0.833492 w_k1_DIS_s107
                                 + 0.827687 w_k1_FDX_s107
                                 + 1.30231 w_k1_IBM_s107
                                 + 0.953351 w_k1_INTC_s107
                                 + 0.920188 w_k1_KO_s107
                                 + 0.858213 w_k1_MAT_s107
                                 + 0.952584 w_k1_MSFT_s107
                                 + 0.878345 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.983859 w_k1_ADM_s107
                                 + 1.44841 w_k1_AXP_s107
                                 + 1.39548 w_k1_DIS_s107
                                 + 1.21635 w_k1_FDX_s107
                                 + 1.50948 w_k1_IBM_s107
                                 + 1.04741 w_k1_INTC_s107
                                 + 1.19613 w_k1_KO_s107 + 1.42646 w_k1_MAT_s107
                                 + 1.17551 w_k1_MSFT_s107
                                 + 1.08895 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.803546 w_k1_ADM_s107
                                 + 0.703503 w_k1_AXP_s107
                                 + 0.745891 w_k1_DIS_s107
                                 + 0.812643 w_k1_FDX_s107
                                 + 0.747694 w_k1_IBM_s107
                                 + 0.856071 w_k1_INTC_s107
                                 + 0.637584 w_k1_KO_s107
                                 + 0.491596 w_k1_MAT_s107
                                 + 0.775613 w_k1_MSFT_s107
                                 + 0.804474 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.25036 w_k1_ADM_s107
                                 + 1.24281 w_k1_AXP_s107
                                 + 1.16803 w_k1_DIS_s107
                                 + 1.12085 w_k1_FDX_s107
                                 + 1.07548 w_k1_IBM_s107
                                 + 1.15582 w_k1_INTC_s107
                                 + 0.972403 w_k1_KO_s107
                                 + 1.02521 w_k1_MAT_s107
                                 + 1.16328 w_k1_MSFT_s107
                                 + 1.02247 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.713216 w_k1_ADM_s107
                                 + 0.974847 w_k1_AXP_s107
                                 + 0.543061 w_k1_DIS_s107
                                 + 0.796231 w_k1_FDX_s107
                                 + 0.817779 w_k1_IBM_s107
                                 + 0.750754 w_k1_INTC_s107
                                 + 0.611822 w_k1_KO_s107
                                 + 0.969496 w_k1_MAT_s107
                                 + 0.718539 w_k1_MSFT_s107
                                 + 0.706579 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.858318 w_k1_ADM_s110
                                 + 1.27324 w_k1_AXP_s110 + 1.2469 w_k1_DIS_s110
                                 + 1.18934 w_k1_FDX_s110
                                 + 1.24732 w_k1_IBM_s110
                                 + 1.38159 w_k1_INTC_s110 + 1.2775 w_k1_KO_s110
                                 + 1.44417 w_k1_MAT_s110
                                 + 1.18352 w_k1_MSFT_s110
                                 + 1.08202 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.940621 w_k1_ADM_s110
                                 + 0.981457 w_k1_AXP_s110
                                 + 0.96724 w_k1_DIS_s110
                                 + 1.05949 w_k1_FDX_s110
                                 + 0.908772 w_k1_IBM_s110
                                 + 1.14859 w_k1_INTC_s110
                                 + 0.957862 w_k1_KO_s110
                                 + 0.952635 w_k1_MAT_s110
                                 + 1.06452 w_k1_MSFT_s110
                                 + 1.03048 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.06818 w_k1_ADM_s110
                                 + 1.5333 w_k1_AXP_s110 + 1.63185 w_k1_DIS_s110
                                 + 1.3441 w_k1_FDX_s110 + 1.31354 w_k1_IBM_s110
                                 + 1.54358 w_k1_INTC_s110
                                 + 1.48794 w_k1_KO_s110 + 1.77111 w_k1_MAT_s110
                                 + 1.43042 w_k1_MSFT_s110
                                 + 1.17717 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.653653 w_k1_ADM_s110
                                 + 0.914236 w_k1_AXP_s110
                                 + 0.706495 w_k1_DIS_s110
                                 + 0.824057 w_k1_FDX_s110
                                 + 0.977772 w_k1_IBM_s110
                                 + 0.974419 w_k1_INTC_s110
                                 + 0.821404 w_k1_KO_s110
                                 + 1.07459 w_k1_MAT_s110
                                 + 0.834641 w_k1_MSFT_s110
                                 + 0.918659 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.613767 w_k1_ADM_s110
                                 + 0.843332 w_k1_AXP_s110
                                 + 0.639084 w_k1_DIS_s110
                                 + 0.888033 w_k1_FDX_s110
                                 + 0.595574 w_k1_IBM_s110
                                 + 1.07051 w_k1_INTC_s110
                                 + 0.693966 w_k1_KO_s110
                                 + 0.622045 w_k1_MAT_s110
                                 + 0.875481 w_k1_MSFT_s110
                                 + 0.822617 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.81887 w_k1_ADM_s110
                                 + 1.17345 w_k1_AXP_s110
                                 + 1.02121 w_k1_DIS_s110
                                 + 0.97655 w_k1_FDX_s110
                                 + 1.33369 w_k1_IBM_s110
                                 + 1.16853 w_k1_INTC_s110
                                 + 1.14411 w_k1_KO_s110 + 1.64767 w_k1_MAT_s110
                                 + 1.01608 w_k1_MSFT_s110
                                 + 1.08803 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.879648 w_k1_ADM_s110
                                 + 0.773104 w_k1_AXP_s110
                                 + 0.756966 w_k1_DIS_s110
                                 + 0.809097 w_k1_FDX_s110
                                 + 0.849523 w_k1_IBM_s110
                                 + 0.788374 w_k1_INTC_s110
                                 + 0.85894 w_k1_KO_s110 + 0.68808 w_k1_MAT_s110
                                 + 0.849603 w_k1_MSFT_s110
                                 + 0.793651 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.08981 w_k1_ADM_s110
                                 + 1.58594 w_k1_AXP_s110
                                 + 1.37868 w_k1_DIS_s110
                                 + 1.28038 w_k1_FDX_s110
                                 + 1.00137 w_k1_IBM_s110
                                 + 1.06154 w_k1_INTC_s110
                                 + 1.13027 w_k1_KO_s110 + 1.49073 w_k1_MAT_s110
                                 + 1.15276 w_k1_MSFT_s110 + 1.0492 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.737813 w_k1_ADM_s111
                                 + 0.77951 w_k1_AXP_s111
                                 + 0.780724 w_k1_DIS_s111
                                 + 0.762984 w_k1_FDX_s111
                                 + 0.867489 w_k1_IBM_s111
                                 + 1.10494 w_k1_INTC_s111
                                 + 0.905397 w_k1_KO_s111
                                 + 0.828493 w_k1_MAT_s111
                                 + 0.969455 w_k1_MSFT_s111
                                 + 0.864436 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.930112 w_k1_ADM_s111
                                 + 1.28527 w_k1_AXP_s111
                                 + 1.17098 w_k1_DIS_s111
                                 + 1.09818 w_k1_FDX_s111
                                 + 0.859796 w_k1_IBM_s111
                                 + 1.29077 w_k1_INTC_s111
                                 + 0.942003 w_k1_KO_s111
                                 + 0.894731 w_k1_MAT_s111
                                 + 1.13522 w_k1_MSFT_s111
                                 + 0.994076 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.844398 w_k1_ADM_s111
                                 + 0.965157 w_k1_AXP_s111
                                 + 1.00207 w_k1_DIS_s111 + 1.0143 w_k1_FDX_s111
                                 + 1.16753 w_k1_IBM_s111
                                 + 1.20188 w_k1_INTC_s111
                                 + 1.04355 w_k1_KO_s111 + 1.19208 w_k1_MAT_s111
                                 + 1.11992 w_k1_MSFT_s111
                                 + 1.05546 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.81786 w_k1_ADM_s111
                                 + 0.97605 w_k1_AXP_s111
                                 + 1.11887 w_k1_DIS_s111
                                 + 1.10911 w_k1_FDX_s111
                                 + 0.712848 w_k1_IBM_s111
                                 + 1.30606 w_k1_INTC_s111
                                 + 0.915849 w_k1_KO_s111
                                 + 0.61629 w_k1_MAT_s111
                                 + 1.11623 w_k1_MSFT_s111
                                 + 0.978442 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.06094 w_k1_ADM_s111
                                 + 1.42942 w_k1_AXP_s111
                                 + 1.78195 w_k1_DIS_s111 + 1.3716 w_k1_FDX_s111
                                 + 1.0938 w_k1_IBM_s111
                                 + 1.48895 w_k1_INTC_s111
                                 + 1.27949 w_k1_KO_s111
                                 + 0.900634 w_k1_MAT_s111
                                 + 1.3474 w_k1_MSFT_s111 + 1.20566 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.44306 w_k1_ADM_s111
                                 + 1.71235 w_k1_AXP_s111
                                 + 1.54202 w_k1_DIS_s111
                                 + 1.26584 w_k1_FDX_s111
                                 + 1.91269 w_k1_IBM_s111
                                 + 1.52548 w_k1_INTC_s111
                                 + 1.37241 w_k1_KO_s111 + 1.65913 w_k1_MAT_s111
                                 + 1.42136 w_k1_MSFT_s111
                                 + 1.20497 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.02224 w_k1_ADM_s111
                                 + 1.05098 w_k1_AXP_s111
                                 + 1.25811 w_k1_DIS_s111
                                 + 0.974377 w_k1_FDX_s111
                                 + 1.5203 w_k1_IBM_s111
                                 + 1.04575 w_k1_INTC_s111 + 1.2073 w_k1_KO_s111
                                 + 1.46542 w_k1_MAT_s111
                                 + 0.82116 w_k1_MSFT_s111
                                 + 1.05697 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.944515 w_k1_ADM_s111
                                 + 1.1303 w_k1_AXP_s111 + 1.35609 w_k1_DIS_s111
                                 + 1.0558 w_k1_FDX_s111 + 1.10692 w_k1_IBM_s111
                                 + 1.26195 w_k1_INTC_s111
                                 + 1.28831 w_k1_KO_s111 + 1.28661 w_k1_MAT_s111
                                 + 1.16118 w_k1_MSFT_s111 + 1.16 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.772572 w_k1_ADM_s112
                                 + 0.727468 w_k1_AXP_s112
                                 + 0.808428 w_k1_DIS_s112
                                 + 0.782579 w_k1_FDX_s112
                                 + 0.839499 w_k1_IBM_s112
                                 + 0.80794 w_k1_INTC_s112
                                 + 0.749394 w_k1_KO_s112
                                 + 0.810417 w_k1_MAT_s112
                                 + 0.781189 w_k1_MSFT_s112
                                 + 0.882673 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.21456 w_k1_ADM_s112
                                 + 0.99704 w_k1_AXP_s112
                                 + 1.18243 w_k1_DIS_s112
                                 + 1.18646 w_k1_FDX_s112
                                 + 0.724588 w_k1_IBM_s112
                                 + 1.08987 w_k1_INTC_s112
                                 + 0.773981 w_k1_KO_s112
                                 + 0.803589 w_k1_MAT_s112
                                 + 1.16893 w_k1_MSFT_s112
                                 + 1.10288 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.10792 w_k1_ADM_s112
                                 + 1.50858 w_k1_AXP_s112
                                 + 1.45283 w_k1_DIS_s112
                                 + 1.33125 w_k1_FDX_s112
                                 + 1.54167 w_k1_IBM_s112
                                 + 1.1913 w_k1_INTC_s112 + 1.23677 w_k1_KO_s112
                                 + 2.08663 w_k1_MAT_s112
                                 + 1.27165 w_k1_MSFT_s112
                                 + 1.06842 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.19209 w_k1_ADM_s112
                                 + 1.35543 w_k1_AXP_s112 + 1.5738 w_k1_DIS_s112
                                 + 1.56096 w_k1_FDX_s112
                                 + 1.07179 w_k1_IBM_s112 + 1.073 w_k1_INTC_s112
                                 + 1.24885 w_k1_KO_s112 + 1.69938 w_k1_MAT_s112
                                 + 1.29083 w_k1_MSFT_s112
                                 + 1.19569 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.813316 w_k1_ADM_s112
                                 + 0.762224 w_k1_AXP_s112
                                 + 0.762997 w_k1_DIS_s112
                                 + 0.809706 w_k1_FDX_s112
                                 + 1.36439 w_k1_IBM_s112
                                 + 0.777271 w_k1_INTC_s112
                                 + 0.930907 w_k1_KO_s112
                                 + 1.27319 w_k1_MAT_s112
                                 + 0.851201 w_k1_MSFT_s112
                                 + 0.932496 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.935816 w_k1_ADM_s112
                                 + 0.938614 w_k1_AXP_s112
                                 + 1.07348 w_k1_DIS_s112
                                 + 1.02085 w_k1_FDX_s112
                                 + 0.963574 w_k1_IBM_s112
                                 + 0.826746 w_k1_INTC_s112
                                 + 0.970385 w_k1_KO_s112
                                 + 1.41143 w_k1_MAT_s112
                                 + 0.991779 w_k1_MSFT_s112
                                 + 0.965967 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.755758 w_k1_ADM_s112
                                 + 0.910737 w_k1_AXP_s112
                                 + 1.02663 w_k1_DIS_s112
                                 + 0.914455 w_k1_FDX_s112
                                 + 1.76472 w_k1_IBM_s112
                                 + 0.87008 w_k1_INTC_s112
                                 + 1.03587 w_k1_KO_s112 + 1.64301 w_k1_MAT_s112
                                 + 0.798513 w_k1_MSFT_s112
                                 + 0.996548 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.40195 w_k1_ADM_s112
                                 + 1.69236 w_k1_AXP_s112
                                 + 1.09582 w_k1_DIS_s112
                                 + 1.27707 w_k1_FDX_s112
                                 + 1.02528 w_k1_IBM_s112
                                 + 1.15835 w_k1_INTC_s112
                                 + 1.09418 w_k1_KO_s112 + 1.58829 w_k1_MAT_s112
                                 + 1.18028 w_k1_MSFT_s112
                                 + 1.18546 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.12555 w_k1_ADM_s113
                                 + 1.54061 w_k1_AXP_s113
                                 + 1.71414 w_k1_DIS_s113
                                 + 1.35529 w_k1_FDX_s113
                                 + 1.66311 w_k1_IBM_s113
                                 + 1.4001 w_k1_INTC_s113 + 1.58528 w_k1_KO_s113
                                 + 1.57528 w_k1_MAT_s113
                                 + 1.3379 w_k1_MSFT_s113 + 1.18967 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.922072 w_k1_ADM_s113
                                 + 0.739691 w_k1_AXP_s113
                                 + 0.708449 w_k1_DIS_s113
                                 + 0.703455 w_k1_FDX_s113
                                 + 0.92057 w_k1_IBM_s113
                                 + 0.901402 w_k1_INTC_s113
                                 + 0.74084 w_k1_KO_s113
                                 + 0.704681 w_k1_MAT_s113
                                 + 0.72896 w_k1_MSFT_s113
                                 + 0.753446 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.96487 w_k1_ADM_s113
                                 + 1.1614 w_k1_AXP_s113 + 1.06502 w_k1_DIS_s113
                                 + 0.896155 w_k1_FDX_s113
                                 + 1.48179 w_k1_IBM_s113
                                 + 0.879681 w_k1_INTC_s113
                                 + 1.25585 w_k1_KO_s113 + 1.49478 w_k1_MAT_s113
                                 + 0.763262 w_k1_MSFT_s113
                                 + 0.985269 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.906881 w_k1_ADM_s113
                                 + 0.986035 w_k1_AXP_s113
                                 + 1.14014 w_k1_DIS_s113
                                 + 1.11805 w_k1_FDX_s113
                                 + 0.791921 w_k1_IBM_s113
                                 + 1.09169 w_k1_INTC_s113
                                 + 0.965513 w_k1_KO_s113
                                 + 0.723999 w_k1_MAT_s113
                                 + 1.0737 w_k1_MSFT_s113 + 1.01151 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.30692 w_k1_ADM_s113
                                 + 1.44697 w_k1_AXP_s113 + 1.6983 w_k1_DIS_s113
                                 + 1.3576 w_k1_FDX_s113 + 1.04023 w_k1_IBM_s113
                                 + 1.47455 w_k1_INTC_s113
                                 + 1.30464 w_k1_KO_s113
                                 + 0.955658 w_k1_MAT_s113
                                 + 1.25582 w_k1_MSFT_s113
                                 + 1.01057 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.969565 w_k1_ADM_s113
                                 + 1.00369 w_k1_AXP_s113 + 1.045 w_k1_DIS_s113
                                 + 1.00566 w_k1_FDX_s113
                                 + 1.16497 w_k1_IBM_s113
                                 + 1.02855 w_k1_INTC_s113
                                 + 1.09509 w_k1_KO_s113 + 1.01849 w_k1_MAT_s113
                                 + 1.03819 w_k1_MSFT_s113
                                 + 0.961576 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.865023 w_k1_ADM_s113
                                 + 0.834381 w_k1_AXP_s113
                                 + 0.84538 w_k1_DIS_s113
                                 + 0.914299 w_k1_FDX_s113
                                 + 0.649765 w_k1_IBM_s113
                                 + 0.825837 w_k1_INTC_s113
                                 + 0.890529 w_k1_KO_s113
                                 + 0.685163 w_k1_MAT_s113
                                 + 0.787789 w_k1_MSFT_s113
                                 + 0.885285 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.939688 w_k1_ADM_s113
                                 + 1.17528 w_k1_AXP_s113
                                 + 1.06756 w_k1_DIS_s113
                                 + 1.04615 w_k1_FDX_s113
                                 + 0.963344 w_k1_IBM_s113
                                 + 1.13865 w_k1_INTC_s113
                                 + 1.10629 w_k1_KO_s113 + 1.0379 w_k1_MAT_s113
                                 + 1.04332 w_k1_MSFT_s113
                                 + 1.01987 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.85476 w_k1_ADM_s114
                                 + 1.23415 w_k1_AXP_s114
                                 + 1.41354 w_k1_DIS_s114
                                 + 1.09623 w_k1_FDX_s114
                                 + 1.49794 w_k1_IBM_s114
                                 + 1.40613 w_k1_INTC_s114
                                 + 1.27253 w_k1_KO_s114 + 1.18264 w_k1_MAT_s114
                                 + 1.15224 w_k1_MSFT_s114
                                 + 1.14997 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.13259 w_k1_ADM_s114
                                 + 1.55288 w_k1_AXP_s114 + 1.475 w_k1_DIS_s114
                                 + 1.30352 w_k1_FDX_s114
                                 + 1.40207 w_k1_IBM_s114
                                 + 1.47793 w_k1_INTC_s114
                                 + 1.48064 w_k1_KO_s114 + 1.13217 w_k1_MAT_s114
                                 + 1.37913 w_k1_MSFT_s114
                                 + 1.22367 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.12908 w_k1_ADM_s114
                                 + 1.15945 w_k1_AXP_s114 + 1.3717 w_k1_DIS_s114
                                 + 1.274 w_k1_FDX_s114 + 0.765616 w_k1_IBM_s114
                                 + 1.19768 w_k1_INTC_s114
                                 + 0.938693 w_k1_KO_s114
                                 + 0.77461 w_k1_MAT_s114
                                 + 1.48379 w_k1_MSFT_s114
                                 + 1.02406 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.945116 w_k1_ADM_s114
                                 + 1.56792 w_k1_AXP_s114
                                 + 1.11767 w_k1_DIS_s114
                                 + 1.12686 w_k1_FDX_s114 + 1.0165 w_k1_IBM_s114
                                 + 1.37832 w_k1_INTC_s114
                                 + 1.14777 w_k1_KO_s114 + 1.06551 w_k1_MAT_s114
                                 + 1.25062 w_k1_MSFT_s114
                                 + 1.11263 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.858769 w_k1_ADM_s114
                                 + 0.968503 w_k1_AXP_s114
                                 + 0.86001 w_k1_DIS_s114
                                 + 0.926929 w_k1_FDX_s114
                                 + 0.714811 w_k1_IBM_s114
                                 + 1.20943 w_k1_INTC_s114
                                 + 0.95747 w_k1_KO_s114
                                 + 0.688713 w_k1_MAT_s114
                                 + 1.06187 w_k1_MSFT_s114
                                 + 0.981964 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.962108 w_k1_ADM_s114
                                 + 1.00517 w_k1_AXP_s114
                                 + 1.00636 w_k1_DIS_s114
                                 + 0.859978 w_k1_FDX_s114
                                 + 1.16894 w_k1_IBM_s114
                                 + 1.1566 w_k1_INTC_s114 + 1.12104 w_k1_KO_s114
                                 + 1.23658 w_k1_MAT_s114
                                 + 1.06436 w_k1_MSFT_s114
                                 + 0.990389 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.939156 w_k1_ADM_s114
                                 + 0.8228 w_k1_AXP_s114 + 1.22279 w_k1_DIS_s114
                                 + 1.08295 w_k1_FDX_s114
                                 + 0.916876 w_k1_IBM_s114
                                 + 1.17602 w_k1_INTC_s114
                                 + 1.15567 w_k1_KO_s114
                                 + 0.684633 w_k1_MAT_s114
                                 + 1.31591 w_k1_MSFT_s114
                                 + 1.07172 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.825731 w_k1_ADM_s114
                                 + 0.783603 w_k1_AXP_s114
                                 + 0.676286 w_k1_DIS_s114
                                 + 0.76592 w_k1_FDX_s114
                                 + 0.772623 w_k1_IBM_s114
                                 + 1.00106 w_k1_INTC_s114
                                 + 0.853112 w_k1_KO_s114
                                 + 0.523006 w_k1_MAT_s114
                                 + 0.792845 w_k1_MSFT_s114
                                 + 0.771942 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.10221 w_k1_ADM_s115
                                 + 0.956172 w_k1_AXP_s115
                                 + 0.739695 w_k1_DIS_s115
                                 + 0.897396 w_k1_FDX_s115
                                 + 0.70391 w_k1_IBM_s115
                                 + 0.933812 w_k1_INTC_s115
                                 + 0.773146 w_k1_KO_s115
                                 + 0.736885 w_k1_MAT_s115
                                 + 0.870775 w_k1_MSFT_s115
                                 + 0.898933 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.901127 w_k1_ADM_s115
                                 + 1.08991 w_k1_AXP_s115
                                 + 1.00826 w_k1_DIS_s115
                                 + 0.980673 w_k1_FDX_s115
                                 + 0.994865 w_k1_IBM_s115
                                 + 1.01091 w_k1_INTC_s115
                                 + 1.04052 w_k1_KO_s115 + 1.07736 w_k1_MAT_s115
                                 + 0.957803 w_k1_MSFT_s115
                                 + 0.876024 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.34027 w_k1_ADM_s115
                                 + 1.31122 w_k1_AXP_s115
                                 + 1.25342 w_k1_DIS_s115
                                 + 1.14174 w_k1_FDX_s115
                                 + 1.33964 w_k1_IBM_s115
                                 + 1.11121 w_k1_INTC_s115
                                 + 1.32455 w_k1_KO_s115 + 1.50255 w_k1_MAT_s115
                                 + 1.07933 w_k1_MSFT_s115 + 1.0275 w_k1_PG_s115
                                 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.44885 w_k1_ADM_s115
                                 + 1.02777 w_k1_AXP_s115
                                 + 1.16679 w_k1_DIS_s115
                                 + 1.10737 w_k1_FDX_s115
                                 + 0.854191 w_k1_IBM_s115
                                 + 0.812984 w_k1_INTC_s115
                                 + 0.959813 w_k1_KO_s115
                                 + 0.636884 w_k1_MAT_s115
                                 + 0.993201 w_k1_MSFT_s115
                                 + 0.984216 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.25956 w_k1_ADM_s115
                                 + 1.48478 w_k1_AXP_s115
                                 + 1.22215 w_k1_DIS_s115
                                 + 1.27158 w_k1_FDX_s115
                                 + 0.81176 w_k1_IBM_s115
                                 + 1.11849 w_k1_INTC_s115
                                 + 0.957683 w_k1_KO_s115
                                 + 0.899749 w_k1_MAT_s115
                                 + 1.00704 w_k1_MSFT_s115
                                 + 0.963486 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.942882 w_k1_ADM_s115
                                 + 0.93543 w_k1_AXP_s115
                                 + 0.724353 w_k1_DIS_s115
                                 + 0.786627 w_k1_FDX_s115
                                 + 1.03829 w_k1_IBM_s115
                                 + 1.01663 w_k1_INTC_s115
                                 + 1.00557 w_k1_KO_s115
                                 + 0.936094 w_k1_MAT_s115
                                 + 0.850828 w_k1_MSFT_s115
                                 + 0.938131 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.15128 w_k1_ADM_s115
                                 + 1.27306 w_k1_AXP_s115
                                 + 0.805157 w_k1_DIS_s115
                                 + 0.987468 w_k1_FDX_s115
                                 + 0.936376 w_k1_IBM_s115
                                 + 0.873727 w_k1_INTC_s115
                                 + 0.887484 w_k1_KO_s115
                                 + 0.988669 w_k1_MAT_s115
                                 + 0.960668 w_k1_MSFT_s115
                                 + 0.931185 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.02848 w_k1_ADM_s115
                                 + 0.739036 w_k1_AXP_s115
                                 + 0.741112 w_k1_DIS_s115
                                 + 0.868508 w_k1_FDX_s115
                                 + 0.586844 w_k1_IBM_s115
                                 + 0.913143 w_k1_INTC_s115
                                 + 0.637367 w_k1_KO_s115
                                 + 0.577723 w_k1_MAT_s115
                                 + 0.996782 w_k1_MSFT_s115
                                 + 0.817526 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.03424 w_k1_ADM_s116
                                 + 0.85632 w_k1_AXP_s116
                                 + 0.813785 w_k1_DIS_s116
                                 + 0.955456 w_k1_FDX_s116
                                 + 0.526818 w_k1_IBM_s116
                                 + 0.898105 w_k1_INTC_s116
                                 + 0.812522 w_k1_KO_s116
                                 + 0.542703 w_k1_MAT_s116
                                 + 0.925874 w_k1_MSFT_s116
                                 + 0.881916 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.26833 w_k1_ADM_s116
                                 + 0.991516 w_k1_AXP_s116
                                 + 1.14086 w_k1_DIS_s116
                                 + 0.941711 w_k1_FDX_s116
                                 + 1.50564 w_k1_IBM_s116
                                 + 1.01796 w_k1_INTC_s116
                                 + 1.40456 w_k1_KO_s116 + 1.31737 w_k1_MAT_s116
                                 + 1.01843 w_k1_MSFT_s116 + 1.0876 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.871394 w_k1_ADM_s116
                                 + 0.750185 w_k1_AXP_s116
                                 + 0.754451 w_k1_DIS_s116
                                 + 0.781879 w_k1_FDX_s116
                                 + 1.11497 w_k1_IBM_s116
                                 + 0.780313 w_k1_INTC_s116
                                 + 1.16493 w_k1_KO_s116 + 1.37759 w_k1_MAT_s116
                                 + 0.832254 w_k1_MSFT_s116
                                 + 0.954148 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.950981 w_k1_ADM_s116
                                 + 0.628821 w_k1_AXP_s116
                                 + 0.605342 w_k1_DIS_s116
                                 + 0.735612 w_k1_FDX_s116
                                 + 0.704304 w_k1_IBM_s116
                                 + 0.660032 w_k1_INTC_s116
                                 + 0.817793 w_k1_KO_s116
                                 + 0.810406 w_k1_MAT_s116
                                 + 0.751299 w_k1_MSFT_s116
                                 + 0.891292 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.04951 w_k1_ADM_s116
                                 + 0.766975 w_k1_AXP_s116
                                 + 0.739464 w_k1_DIS_s116
                                 + 0.885881 w_k1_FDX_s116
                                 + 0.805364 w_k1_IBM_s116
                                 + 0.959564 w_k1_INTC_s116
                                 + 0.90965 w_k1_KO_s116
                                 + 0.886952 w_k1_MAT_s116
                                 + 0.889269 w_k1_MSFT_s116
                                 + 0.981481 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.35235 w_k1_ADM_s116
                                 + 1.0999 w_k1_AXP_s116 + 1.25537 w_k1_DIS_s116
                                 + 1.2436 w_k1_FDX_s116
                                 + 0.895751 w_k1_IBM_s116
                                 + 1.17005 w_k1_INTC_s116
                                 + 1.09456 w_k1_KO_s116
                                 + 0.918967 w_k1_MAT_s116
                                 + 1.16716 w_k1_MSFT_s116 + 1.1519 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.09707 w_k1_ADM_s116
                                 + 1.03329 w_k1_AXP_s116
                                 + 0.890156 w_k1_DIS_s116
                                 + 0.965542 w_k1_FDX_s116
                                 + 1.02382 w_k1_IBM_s116
                                 + 0.96022 w_k1_INTC_s116
                                 + 0.954585 w_k1_KO_s116
                                 + 1.08309 w_k1_MAT_s116
                                 + 0.881659 w_k1_MSFT_s116
                                 + 1.06585 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.29352 w_k1_ADM_s116
                                 + 1.4597 w_k1_AXP_s116 + 1.30597 w_k1_DIS_s116
                                 + 1.18808 w_k1_FDX_s116
                                 + 1.37988 w_k1_IBM_s116
                                 + 1.16774 w_k1_INTC_s116
                                 + 1.28764 w_k1_KO_s116 + 1.61162 w_k1_MAT_s116
                                 + 1.19626 w_k1_MSFT_s116
                                 + 1.17808 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.16723 w_k1_ADM_s117
                                 + 1.2922 w_k1_AXP_s117 + 1.26155 w_k1_DIS_s117
                                 + 1.13998 w_k1_FDX_s117
                                 + 1.32191 w_k1_IBM_s117
                                 + 1.22845 w_k1_INTC_s117
                                 + 1.37553 w_k1_KO_s117 + 1.40122 w_k1_MAT_s117
                                 + 1.1235 w_k1_MSFT_s117 + 1.20423 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.856286 w_k1_ADM_s117
                                 + 0.995877 w_k1_AXP_s117
                                 + 0.872511 w_k1_DIS_s117
                                 + 0.966184 w_k1_FDX_s117
                                 + 0.750029 w_k1_IBM_s117
                                 + 1.0508 w_k1_INTC_s117
                                 + 0.885339 w_k1_KO_s117
                                 + 0.652796 w_k1_MAT_s117
                                 + 0.843988 w_k1_MSFT_s117
                                 + 0.985621 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.712996 w_k1_ADM_s117
                                 + 0.730879 w_k1_AXP_s117
                                 + 0.708687 w_k1_DIS_s117
                                 + 0.745436 w_k1_FDX_s117
                                 + 0.804493 w_k1_IBM_s117
                                 + 0.745702 w_k1_INTC_s117
                                 + 0.807672 w_k1_KO_s117
                                 + 0.638166 w_k1_MAT_s117
                                 + 0.752587 w_k1_MSFT_s117
                                 + 1.00665 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.0998 w_k1_ADM_s117
                                 + 1.39287 w_k1_AXP_s117
                                 + 1.37405 w_k1_DIS_s117 + 1.3051 w_k1_FDX_s117
                                 + 0.982285 w_k1_IBM_s117
                                 + 1.16872 w_k1_INTC_s117
                                 + 1.22883 w_k1_KO_s117 + 1.27844 w_k1_MAT_s117
                                 + 1.11071 w_k1_MSFT_s117
                                 + 1.25845 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.664113 w_k1_ADM_s117
                                 + 0.957315 w_k1_AXP_s117
                                 + 0.708932 w_k1_DIS_s117
                                 + 0.84393 w_k1_FDX_s117
                                 + 0.863691 w_k1_IBM_s117
                                 + 0.950062 w_k1_INTC_s117
                                 + 0.893865 w_k1_KO_s117
                                 + 0.896111 w_k1_MAT_s117
                                 + 0.768262 w_k1_MSFT_s117
                                 + 0.98469 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.07637 w_k1_ADM_s117
                                 + 1.56448 w_k1_AXP_s117
                                 + 1.26159 w_k1_DIS_s117
                                 + 1.13757 w_k1_FDX_s117
                                 + 1.76725 w_k1_IBM_s117
                                 + 1.31586 w_k1_INTC_s117
                                 + 1.43891 w_k1_KO_s117 + 1.67216 w_k1_MAT_s117
                                 + 0.9809 w_k1_MSFT_s117 + 1.31552 w_k1_PG_s117
                                 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.820036 w_k1_ADM_s117
                                 + 0.969456 w_k1_AXP_s117
                                 + 1.00166 w_k1_DIS_s117
                                 + 0.936725 w_k1_FDX_s117
                                 + 1.12996 w_k1_IBM_s117
                                 + 1.01606 w_k1_INTC_s117
                                 + 0.872289 w_k1_KO_s117
                                 + 1.04159 w_k1_MAT_s117
                                 + 0.831806 w_k1_MSFT_s117
                                 + 1.0494 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.752625 w_k1_ADM_s117
                                 + 0.860442 w_k1_AXP_s117
                                 + 0.807185 w_k1_DIS_s117
                                 + 0.748679 w_k1_FDX_s117
                                 + 1.03311 w_k1_IBM_s117
                                 + 0.894867 w_k1_INTC_s117
                                 + 0.9113 w_k1_KO_s117 + 0.844312 w_k1_MAT_s117
                                 + 0.796839 w_k1_MSFT_s117
                                 + 0.953752 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0405911151906229 w_k1_ADM_s000
                                 + 0.0823688248148253 w_k1_ADM_s001
                                 + 0.0501287116198503 w_k1_ADM_s002
                                 + 0.0834864439863725 w_k1_ADM_s003
                                 + 0.0193553403795292 w_k1_ADM_s004
                                 + 0.0563803782214297 w_k1_ADM_s005
                                 + 0.0431712858635281 w_k1_ADM_s006
                                 + 0.0390530608883548 w_k1_ADM_s007
                                 + 0.0230119153260521 w_k1_ADM_s010
                                 + 0.0278093785822637 w_k1_ADM_s011
                                 + 0.0179133709237526 w_k1_ADM_s012
                                 + 0.0338058146650499 w_k1_ADM_s013
                                 + 0.0180853207515277 w_k1_ADM_s014
                                 + 0.0491840708117906 w_k1_ADM_s015
                                 + 0.0351726298936236 w_k1_ADM_s016
                                 + 0.095337506424683 w_k1_ADM_s017
                                 + 0.00776365867785829 w_k1_ADM_s100
                                 + 0.0193315109294958 w_k1_ADM_s101
                                 + 0.0180930696661263 w_k1_ADM_s102
                                 + 0.0183056795943293 w_k1_ADM_s103
                                 + 0.00924617752249955 w_k1_ADM_s104
                                 + 0.010437369604633 w_k1_ADM_s105
                                 + 0.00723768530582874 w_k1_ADM_s106
                                 + 0.0362227641696641 w_k1_ADM_s107
                                 + 0.0127083868987413 w_k1_ADM_s110
                                 + 0.01860360044326 w_k1_ADM_s111
                                 + 0.0141636770108956 w_k1_ADM_s112
                                 + 0.047203653724566 w_k1_ADM_s113
                                 + 0.0143360314619358 w_k1_ADM_s114
                                 + 0.01916064247072 w_k1_ADM_s115
                                 + 0.0406298733799829 w_k1_ADM_s116
                                 + 0.0289714918685751 w_k1_ADM_s117
                                 + 0.036924558123773 w_k1_AXP_s000
                                 + 0.0810877158576992 w_k1_AXP_s001
                                 + 0.050929110141107 w_k1_AXP_s002
                                 + 0.0660879122126082 w_k1_AXP_s003
                                 + 0.0182913105506205 w_k1_AXP_s004
                                 + 0.0537497233070081 w_k1_AXP_s005
                                 + 0.0399453758446953 w_k1_AXP_s006
                                 + 0.0448872973894313 w_k1_AXP_s007
                                 + 0.024046314910068 w_k1_AXP_s010
                                 + 0.0270406698278472 w_k1_AXP_s011
                                 + 0.01957839637434 w_k1_AXP_s012
                                 + 0.0319694258620053 w_k1_AXP_s013
                                 + 0.0180897600477479 w_k1_AXP_s014
                                 + 0.051046035160515 w_k1_AXP_s015
                                 + 0.0399040007798148 w_k1_AXP_s016
                                 + 0.0860450107784516 w_k1_AXP_s017
                                 + 0.00702514320115412 w_k1_AXP_s100
                                 + 0.0212213700630835 w_k1_AXP_s101
                                 + 0.019100801161843 w_k1_AXP_s102
                                 + 0.0169326470696746 w_k1_AXP_s103
                                 + 0.00824285548189333 w_k1_AXP_s104
                                 + 0.012261542346962 w_k1_AXP_s105
                                 + 0.00679719050498689 w_k1_AXP_s106
                                 + 0.0383420674173096 w_k1_AXP_s107
                                 + 0.0163391489757188 w_k1_AXP_s110
                                 + 0.0219281280894056 w_k1_AXP_s111
                                 + 0.0147059260426885 w_k1_AXP_s112
                                 + 0.0517962834109895 w_k1_AXP_s113
                                 + 0.0165364827357337 w_k1_AXP_s114
                                 + 0.0183852680888286 w_k1_AXP_s115
                                 + 0.0356110402376857 w_k1_AXP_s116
                                 + 0.0347464564470765 w_k1_AXP_s117
                                 + 0.0426056431349862 w_k1_DIS_s000
                                 + 0.0829700390756108 w_k1_DIS_s001
                                 + 0.0496812284946125 w_k1_DIS_s002
                                 + 0.0699552612193882 w_k1_DIS_s003
                                 + 0.0174167505452045 w_k1_DIS_s004
                                 + 0.0559201627330794 w_k1_DIS_s005
                                 + 0.0407826462337077 w_k1_DIS_s006
                                 + 0.0440525399035576 w_k1_DIS_s007
                                 + 0.0260889584671632 w_k1_DIS_s010
                                 + 0.0308570321175657 w_k1_DIS_s011
                                 + 0.0211944489935932 w_k1_DIS_s012
                                 + 0.0291851019888785 w_k1_DIS_s013
                                 + 0.0174603657853 w_k1_DIS_s014
                                 + 0.0495613495722648 w_k1_DIS_s015
                                 + 0.0358808692339184 w_k1_DIS_s016
                                 + 0.0896098715459974 w_k1_DIS_s017
                                 + 0.00784163624499848 w_k1_DIS_s100
                                 + 0.0195460520709872 w_k1_DIS_s101
                                 + 0.0185979496859967 w_k1_DIS_s102
                                 + 0.0174579177182183 w_k1_DIS_s103
                                 + 0.0089260175963897 w_k1_DIS_s104
                                 + 0.0122204529018236 w_k1_DIS_s105
                                 + 0.00773379660719852 w_k1_DIS_s106
                                 + 0.0352579190926485 w_k1_DIS_s107
                                 + 0.0148183833887305 w_k1_DIS_s110
                                 + 0.0231785834044044 w_k1_DIS_s111
                                 + 0.0154895827225573 w_k1_DIS_s112
                                 + 0.0532808318379776 w_k1_DIS_s113
                                 + 0.0161688234824061 w_k1_DIS_s114
                                 + 0.0162814795440575 w_k1_DIS_s115
                                 + 0.0355641132472622 w_k1_DIS_s116
                                 + 0.0327144344527821 w_k1_DIS_s117
                                 + 0.0416206314932514 w_k1_FDX_s000
                                 + 0.0757744924519848 w_k1_FDX_s001
                                 + 0.0472653991387914 w_k1_FDX_s002
                                 + 0.0713321322261131 w_k1_FDX_s003
                                 + 0.0166562729987349 w_k1_FDX_s004
                                 + 0.054922060379816 w_k1_FDX_s005
                                 + 0.040196523749143 w_k1_FDX_s006
                                 + 0.0440375752946502 w_k1_FDX_s007
                                 + 0.0256481958400154 w_k1_FDX_s010
                                 + 0.0278126499537954 w_k1_FDX_s011
                                 + 0.0209821230606256 w_k1_FDX_s012
                                 + 0.0293436131139856 w_k1_FDX_s013
                                 + 0.0174983313834631 w_k1_FDX_s014
                                 + 0.0505267740363462 w_k1_FDX_s015
                                 + 0.0359929443444824 w_k1_FDX_s016
                                 + 0.0852323975178976 w_k1_FDX_s017
                                 + 0.00712507278838912 w_k1_FDX_s100
                                 + 0.0199344998480477 w_k1_FDX_s101
                                 + 0.0194891680156499 w_k1_FDX_s102
                                 + 0.0181868418720449 w_k1_FDX_s103
                                 + 0.00860657318676911 w_k1_FDX_s104
                                 + 0.0116839303722901 w_k1_FDX_s105
                                 + 0.0069671659140906 w_k1_FDX_s106
                                 + 0.0357398624360876 w_k1_FDX_s107
                                 + 0.0152457906039167 w_k1_FDX_s110
                                 + 0.0209844541109282 w_k1_FDX_s111
                                 + 0.0151547177909182 w_k1_FDX_s112
                                 + 0.0495009328311436 w_k1_FDX_s113
                                 + 0.015082657293427 w_k1_FDX_s114
                                 + 0.0165542071481052 w_k1_FDX_s115
                                 + 0.0354083398556486 w_k1_FDX_s116
                                 + 0.0317864160854754 w_k1_FDX_s117
                                 + 0.0423835717698216 w_k1_IBM_s000
                                 + 0.0742792203148007 w_k1_IBM_s001
                                 + 0.0502108223379775 w_k1_IBM_s002
                                 + 0.0713040533058205 w_k1_IBM_s003
                                 + 0.0163916012409095 w_k1_IBM_s004
                                 + 0.0568409116708252 w_k1_IBM_s005
                                 + 0.041943026195252 w_k1_IBM_s006
                                 + 0.044746006336633 w_k1_IBM_s007
                                 + 0.0267631991097363 w_k1_IBM_s010
                                 + 0.0285818975078423 w_k1_IBM_s011
                                 + 0.020077655035646 w_k1_IBM_s012
                                 + 0.0299755615605771 w_k1_IBM_s013
                                 + 0.0176269823668476 w_k1_IBM_s014
                                 + 0.0504137456842355 w_k1_IBM_s015
                                 + 0.0374157759287332 w_k1_IBM_s016
                                 + 0.0869179712874077 w_k1_IBM_s017
                                 + 0.00730035149163874 w_k1_IBM_s100
                                 + 0.0194684047451694 w_k1_IBM_s101
                                 + 0.0198427897079461 w_k1_IBM_s102
                                 + 0.0179372265118679 w_k1_IBM_s103
                                 + 0.00864983902467808 w_k1_IBM_s104
                                 + 0.0125985073290751 w_k1_IBM_s105
                                 + 0.0082062602217829 w_k1_IBM_s106
                                 + 0.036853924239228 w_k1_IBM_s107
                                 + 0.0157599802806458 w_k1_IBM_s110
                                 + 0.0216427333087284 w_k1_IBM_s111
                                 + 0.0161118684164431 w_k1_IBM_s112
                                 + 0.0486601557303088 w_k1_IBM_s113
                                 + 0.0156117551189412 w_k1_IBM_s114
                                 + 0.0163900293444266 w_k1_IBM_s115
                                 + 0.036212024742221 w_k1_IBM_s116
                                 + 0.0334005687030023 w_k1_IBM_s117
                                 + 0.039669542483109 w_k1_INTC_s000
                                 + 0.0793375439095683 w_k1_INTC_s001
                                 + 0.0440379956773969 w_k1_INTC_s002
                                 + 0.0723808034707474 w_k1_INTC_s003
                                 + 0.0177529691639203 w_k1_INTC_s004
                                 + 0.0511775570532551 w_k1_INTC_s005
                                 + 0.044017443684066 w_k1_INTC_s006
                                 + 0.0501124664269727 w_k1_INTC_s007
                                 + 0.0241170978092675 w_k1_INTC_s010
                                 + 0.0299185969671962 w_k1_INTC_s011
                                 + 0.0188670901466898 w_k1_INTC_s012
                                 + 0.0283305522333792 w_k1_INTC_s013
                                 + 0.0163128039457934 w_k1_INTC_s014
                                 + 0.0481768193816996 w_k1_INTC_s015
                                 + 0.0421668770138955 w_k1_INTC_s016
                                 + 0.096757423825384 w_k1_INTC_s017
                                 + 0.00564259431836127 w_k1_INTC_s100
                                 + 0.0234746199198652 w_k1_INTC_s101
                                 + 0.0222566700052187 w_k1_INTC_s102
                                 + 0.0174055854152239 w_k1_INTC_s103
                                 + 0.0105143643407933 w_k1_INTC_s104
                                 + 0.0141809597789928 w_k1_INTC_s105
                                 + 0.00691475125093198 w_k1_INTC_s106
                                 + 0.0333862539332362 w_k1_INTC_s107
                                 + 0.0170953912248922 w_k1_INTC_s110
                                 + 0.0255732318268042 w_k1_INTC_s111
                                 + 0.0135593365076694 w_k1_INTC_s112
                                 + 0.0514898557831841 w_k1_INTC_s113
                                 + 0.0186684643434557 w_k1_INTC_s114
                                 + 0.0164496096120917 w_k1_INTC_s115
                                 + 0.0349143082376413 w_k1_INTC_s116
                                 + 0.033526226497606 w_k1_INTC_s117
                                 + 0.0443501808642698 w_k1_KO_s000
                                 + 0.0730441758176936 w_k1_KO_s001
                                 + 0.0473469544360771 w_k1_KO_s002
                                 + 0.0679370843907111 w_k1_KO_s003
                                 + 0.0164515277527436 w_k1_KO_s004
                                 + 0.0585352660971223 w_k1_KO_s005
                                 + 0.0413592279702783 w_k1_KO_s006
                                 + 0.0475186756289475 w_k1_KO_s007
                                 + 0.0230895918186654 w_k1_KO_s010
                                 + 0.0269944935165653 w_k1_KO_s011
                                 + 0.0209854828782451 w_k1_KO_s012
                                 + 0.0269834446821719 w_k1_KO_s013
                                 + 0.0184771242734432 w_k1_KO_s014
                                 + 0.0507872769835419 w_k1_KO_s015
                                 + 0.0383024987591958 w_k1_KO_s016
                                 + 0.0842059977534224 w_k1_KO_s017
                                 + 0.00616603340523309 w_k1_KO_s100
                                 + 0.0190410438655635 w_k1_KO_s101
                                 + 0.0195382200749303 w_k1_KO_s102
                                 + 0.0186037063521356 w_k1_KO_s103
                                 + 0.00907180798559155 w_k1_KO_s104
                                 + 0.0117415498640917 w_k1_KO_s105
                                 + 0.00754268410822877 w_k1_KO_s106
                                 + 0.0319831898755588 w_k1_KO_s107
                                 + 0.0153717185604944 w_k1_KO_s110
                                 + 0.0216190352899887 w_k1_KO_s111
                                 + 0.0141078591101868 w_k1_KO_s112
                                 + 0.0518512490516007 w_k1_KO_s113
                                 + 0.0167040934620441 w_k1_KO_s114
                                 + 0.0168312467165244 w_k1_KO_s115
                                 + 0.0379328931737596 w_k1_KO_s116
                                 + 0.0330842089589462 w_k1_KO_s117
                                 + 0.0456932780822095 w_k1_MAT_s000
                                 + 0.076577005043158 w_k1_MAT_s001
                                 + 0.0509678019698524 w_k1_MAT_s002
                                 + 0.0737171510225314 w_k1_MAT_s003
                                 + 0.0178092608563651 w_k1_MAT_s004
                                 + 0.059251892374686 w_k1_MAT_s005
                                 + 0.0386707302285423 w_k1_MAT_s006
                                 + 0.0374811984690099 w_k1_MAT_s007
                                 + 0.0241287674487243 w_k1_MAT_s010
                                 + 0.0218726963576467 w_k1_MAT_s011
                                 + 0.0172672254829884 w_k1_MAT_s012
                                 + 0.030819310481808 w_k1_MAT_s013
                                 + 0.0178578377163492 w_k1_MAT_s014
                                 + 0.0430638092249618 w_k1_MAT_s015
                                 + 0.040946227757354 w_k1_MAT_s016
                                 + 0.0955354902142215 w_k1_MAT_s017
                                 + 0.00664500780537129 w_k1_MAT_s100
                                 + 0.0175765792991261 w_k1_MAT_s101
                                 + 0.0189621273549249 w_k1_MAT_s102
                                 + 0.0188651440887715 w_k1_MAT_s103
                                 + 0.00560263015724988 w_k1_MAT_s104
                                 + 0.0141610621196576 w_k1_MAT_s105
                                 + 0.00869893972170304 w_k1_MAT_s106
                                 + 0.0342313736261965 w_k1_MAT_s107
                                 + 0.0182206430886491 w_k1_MAT_s110
                                 + 0.0209402909136832 w_k1_MAT_s111
                                 + 0.0192371874970068 w_k1_MAT_s112
                                 + 0.0464869629664048 w_k1_MAT_s113
                                 + 0.0142643081687779 w_k1_MAT_s114
                                 + 0.0169264750851001 w_k1_MAT_s115
                                 + 0.0386697465392408 w_k1_MAT_s116
                                 + 0.0331412903186028 w_k1_MAT_s117
                                 + 0.0418586717269616 w_k1_MSFT_s000
                                 + 0.0787513209576361 w_k1_MSFT_s001
                                 + 0.049566387931058 w_k1_MSFT_s002
                                 + 0.0752605541233754 w_k1_MSFT_s003
                                 + 0.016656370942935 w_k1_MSFT_s004
                                 + 0.0527781529478828 w_k1_MSFT_s005
                                 + 0.0434767869004431 w_k1_MSFT_s006
                                 + 0.0416659581190508 w_k1_MSFT_s007
                                 + 0.0237354533135616 w_k1_MSFT_s010
                                 + 0.0301042130576686 w_k1_MSFT_s011
                                 + 0.018178438697518 w_k1_MSFT_s012
                                 + 0.0275369506698193 w_k1_MSFT_s013
                                 + 0.0170375553733864 w_k1_MSFT_s014
                                 + 0.0453403060054546 w_k1_MSFT_s015
                                 + 0.0373226828516252 w_k1_MSFT_s016
                                 + 0.0899361666112142 w_k1_MSFT_s017
                                 + 0.00644274912171859 w_k1_MSFT_s100
                                 + 0.0208236120306944 w_k1_MSFT_s101
                                 + 0.021704458199861 w_k1_MSFT_s102
                                 + 0.0178426201750356 w_k1_MSFT_s103
                                 + 0.0096092606362368 w_k1_MSFT_s104
                                 + 0.0126117331550483 w_k1_MSFT_s105
                                 + 0.00741744620796877 w_k1_MSFT_s106
                                 + 0.035178457975437 w_k1_MSFT_s107
                                 + 0.0154429198731158 w_k1_MSFT_s110
                                 + 0.0226737239959021 w_k1_MSFT_s111
                                 + 0.0145246599580586 w_k1_MSFT_s112
                                 + 0.047444723390338 w_k1_MSFT_s113
                                 + 0.0171947177010927 w_k1_MSFT_s114
                                 + 0.0159084384228216 w_k1_MSFT_s115
                                 + 0.0350281929199494 w_k1_MSFT_s116
                                 + 0.0290837221751288 w_k1_MSFT_s117
                                 + 0.0427118739635951 w_k1_PG_s000
                                 + 0.082484477615304 w_k1_PG_s001
                                 + 0.0475875409631915 w_k1_PG_s002
                                 + 0.0678225506620348 w_k1_PG_s003
                                 + 0.0174845721585686 w_k1_PG_s004
                                 + 0.0537265593174081 w_k1_PG_s005
                                 + 0.0405701833416944 w_k1_PG_s006
                                 + 0.0452546187259593 w_k1_PG_s007
                                 + 0.0245129128815446 w_k1_PG_s010
                                 + 0.0267200505353549 w_k1_PG_s011
                                 + 0.0202470408156159 w_k1_PG_s012
                                 + 0.0291589815501034 w_k1_PG_s013
                                 + 0.0171160817608941 w_k1_PG_s014
                                 + 0.050542477877879 w_k1_PG_s015
                                 + 0.03784638441561 w_k1_PG_s016
                                 + 0.0815523396346618 w_k1_PG_s017
                                 + 0.00750234222282278 w_k1_PG_s100
                                 + 0.0191145395577042 w_k1_PG_s101
                                 + 0.0180613067867978 w_k1_PG_s102
                                 + 0.0161088724695407 w_k1_PG_s103
                                 + 0.00811801649259125 w_k1_PG_s104
                                 + 0.0124798737561247 w_k1_PG_s105
                                 + 0.00777741871606515 w_k1_PG_s106
                                 + 0.032976852022774 w_k1_PG_s107
                                 + 0.0151591109396258 w_k1_PG_s110
                                 + 0.0209849826704401 w_k1_PG_s111
                                 + 0.0148409386972965 w_k1_PG_s112
                                 + 0.0462619556037624 w_k1_PG_s113
                                 + 0.015500814291022 w_k1_PG_s114
                                 + 0.0155672779870915 w_k1_PG_s115
                                 + 0.0367506778078956 w_k1_PG_s116
                                 + 0.0345974729809541 w_k1_PG_s117 >= 1.12
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
