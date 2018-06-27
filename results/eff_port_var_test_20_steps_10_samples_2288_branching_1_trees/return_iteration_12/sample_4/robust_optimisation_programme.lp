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
 w_k1_ADM_s0_return_constraint:  1.08379 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.933342 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.28589 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.00206 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.23754 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.879515 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.16795 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.18492 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.928216 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.799343 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.10278 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.15549 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.11051 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.87816 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.20166 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.00865 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.33448 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.01544 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.02422 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.882282 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.11399 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.688694 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.35298 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.940924 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.962843 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.9253 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.827878 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.988546 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.14682 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.10892 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.919805 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.939978 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.793841 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.01243 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.18399 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.39237 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.12559 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.16654 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20179 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.860157 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.54951 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.929938 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.20542 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.864161 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.53962 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.02933 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.979531 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.748956 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.992891 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.29591 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.880558 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.17936 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.929623 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.850282 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.45556 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.27873 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.86503 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.928929 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.06219 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.654963 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.78935 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.02782 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.44485 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.792863 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.849551 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.15762 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.46072 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.09645 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.952565 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.03015 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.643512 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.15248 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.31724 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.68944 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.896734 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.1473 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3087 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.835628 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.55574 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.927392 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.34248 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.788263 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.90789 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.23925 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.07768 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.704447 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.827976 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.43344 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.856694 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.22035 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.40121 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.742615 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.59565 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.12322 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.837443 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.82564 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.10416 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.537783 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.8719 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.834749 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.33618 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.941677 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.706924 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.10481 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.31546 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.49673 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.890198 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.853819 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.659354 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.28982 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.37412 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.7384 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.05296 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.26101 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.16948 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.913166 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.37789 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.935504 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.21754 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.852079 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.45995 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.17979 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.954879 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.783478 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.790551 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.12116 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.986582 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.923008 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.13275 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.912777 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.39552 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.982734 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.04201 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.891112 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.14257 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.689922 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.48697 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.8431 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.08799 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.910892 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.836514 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.0423 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.20035 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.31031 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.779191 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.957996 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.751498 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.04749 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.21674 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.49132 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.09157 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.21987 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.2226 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.901418 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.20567 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.972876 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.16421 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.894464 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.26238 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.81794 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.20064 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.855772 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.39164 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.34074 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.681358 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 2.03947 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.1466 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.783547 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.18895 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.68127 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.545577 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.13272 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.88302 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.720047 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.1781 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.14269 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.94142 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.779865 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.691701 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.22182 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.14982 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.771572 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.61071 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.804988 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.82965 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.51757 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.08272 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.10727 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.793352 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.643978 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15492 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.910495 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.28705 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.909805 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.1969 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.967507 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.3399 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.12168 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.04943 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.893528 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.870037 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.00669 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.844586 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.16261 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.03789 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.967607 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.19554 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.08955 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.95118 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.828117 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.19807 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.742312 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.41598 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.933201 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.21901 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.931297 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.784975 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.988483 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.20738 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.17268 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.757887 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.797605 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.687197 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.10567 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.02942 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.32316 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.917545 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.12813 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.26613 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.877078 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.24975 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.975295 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.27239 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.900591 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.35363 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.950747 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.04266 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.811003 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.958868 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.30458 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.793373 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.41929 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.15573 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.794621 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.14888 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.26574 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.730265 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.856181 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.940569 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.642131 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.35845 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.915953 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.4977 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.822401 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.693205 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.04279 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.31245 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.07608 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.0528 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.879687 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.734139 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.19168 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.05537 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.36541 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.914947 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.91165 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.29745 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.93311 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.32513 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.947355 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.17086 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.940106 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.30966 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.708387 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.2916 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.748963 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.14812 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.43665 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.855901 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 2.08982 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.06501 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.869096 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.21103 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.86524 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.709716 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.18474 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.963846 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.706338 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.59964 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.00198 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 2.10751 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.592989 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.733569 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.1751 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.33285 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.727637 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.33359 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.687356 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.558623 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.27192 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.00461 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.27536 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.673739 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.658608 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14921 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.89172 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.31354 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.990281 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.24069 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.874577 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.32806 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.14156 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.01987 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.851676 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.852949 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.07128 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.946026 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.06015 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.13678 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.868521 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.20392 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.932157 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.01056 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.916308 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.02896 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.69761 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.50688 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.968834 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.13879 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.961023 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.751993 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.01113 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.15022 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.27144 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.862631 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.960172 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.804036 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.1527 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.22063 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.47145 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.10365 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.26083 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.1554 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.916814 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.15311 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.967183 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.12401 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.876844 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.20609 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.00919 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.940339 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.896855 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.956219 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.12772 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.974466 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.01193 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.01663 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.878114 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.13476 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.1405 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.954867 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.958031 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.00114 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.716144 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.35644 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.01652 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.1523 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.984538 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.863196 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.15519 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.13963 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.13236 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.05497 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.966087 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.811247 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.12405 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.11531 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.33157 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.00978 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.08194 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.057999896976402 z_k1_s0000
                                 - 0.0300000241947067 z_k1_s0001
                                 - 0.109999767713858 z_k1_s0002
                                 - 0.109999767713858 z_k1_s0003
                                 - 0.0620000928024027 z_k1_s0004
                                 - 0.0320000151076849 z_k1_s0005
                                 - 0.0620000928024027 z_k1_s0006
                                 - 0.0709998646607661 z_k1_s0007
                                 - 0.0629998017516316 z_k1_s0010
                                 - 0.114999812022305 z_k1_s0011
                                 - 0.167999803336703 z_k1_s0012
                                 - 0.146999205421455 z_k1_s0013
                                 - 0.0899996883309864 z_k1_s0014
                                 - 0.108999925759965 z_k1_s0015
                                 - 0.231999586132058 z_k1_s0016
                                 - 0.0719998303445535 z_k1_s0017
                                 - 0.219999496240922 z_k1_s0020
                                 - 0.269999432432031 z_k1_s0021
                                 - 0.0739999724628305 z_k1_s0022
                                 - 0.0679999355199052 z_k1_s0023
                                 - 0.117000274387122 z_k1_s0024
                                 - 0.147000459101748 z_k1_s0025
                                 - 0.0939999469392744 z_k1_s0026
                                 - 0.125000397977676 z_k1_s0027
                                 - 0.120999805325989 z_k1_s0030
                                 - 0.217000167739329 z_k1_s0031
                                 - 0.32300041483759 z_k1_s0032
                                 - 0.343999043717382 z_k1_s0033
                                 - 0.133999852536131 z_k1_s0034
                                 - 0.11199985534963 z_k1_s0035
                                 - 0.41199997989262 z_k1_s0036
                                 - 0.128999820593787 z_k1_s0037
                                 - 0.122999823408721 z_k1_s0040
                                 - 0.110000348609612 z_k1_s0041
                                 - 0.0310000156444965 z_k1_s0042
                                 - 0.0620000312889931 z_k1_s0043
                                 - 0.102000122656112 z_k1_s0044
                                 - 0.0349999988211988 z_k1_s0045
                                 - 0.117999925562873 z_k1_s0046
                                 - 0.0680002981041543 z_k1_s0047
                                 - 0.0640000073191607 z_k1_s0050
                                 - 0.0409999996341501 z_k1_s0051
                                 - 0.0779997278796169 z_k1_s0052
                                 - 0.0970001758759066 z_k1_s0053
                                 - 0.0659998195135194 z_k1_s0054
                                 - 0.045000012222847 z_k1_s0055
                                 - 0.0970001758759066 z_k1_s0056
                                 - 0.159000241600715 z_k1_s0057
                                 - 0.112000208256732 z_k1_s0060
                                 - 0.0910002191961912 z_k1_s0061
                                 - 0.239000694342972 z_k1_s0062
                                 - 0.190000853168632 z_k1_s0063
                                 - 0.18800001632316 z_k1_s0064
                                 - 0.292000876205693 z_k1_s0065
                                 - 0.118000186088279 z_k1_s0066
                                 - 0.103000174859284 z_k1_s0067
                                 - 0.0860000769826612 z_k1_s0070
                                 - 0.191999984612409 z_k1_s0071
                                 - 0.0450000215560393 z_k1_s0072
                                 - 0.102000075700362 z_k1_s0073
                                 - 0.0649999864031574 z_k1_s0074
                                 - 0.0419999798589606 z_k1_s0075
                                 - 0.0899999089120474 z_k1_s0076
                                 - 0.0489999876854567 z_k1_s0077
                                 - 0.27299991094597 z_k1_s0100
                                 - 0.381999888107549 z_k1_s0101
                                 - 0.16899996140465 z_k1_s0102
                                 - 0.106999931393473 z_k1_s0103
                                 - 0.201999240978279 z_k1_s0104
                                 - 0.0989999219210623 z_k1_s0105
                                 - 0.21999982649125 z_k1_s0106
                                 - 0.283999207870474 z_k1_s0107
                                 - 0.565000665888244 z_k1_s0110
                                 - 0.276000132734856 z_k1_s0111
                                 - 0.242998934364801 z_k1_s0112
                                 - 0.421000596635276 z_k1_s0113
                                 - 0.155999838524133 z_k1_s0114
                                 - 0.230999874593387 z_k1_s0115
                                 - 0.286999743858486 z_k1_s0116
                                 - 0.185999852951834 z_k1_s0117
                                 - 0.286000531380885 z_k1_s0120
                                 - 0.439000876660283 z_k1_s0121
                                 - 0.738001086476023 z_k1_s0122
                                 - 0.591003070694217 z_k1_s0123
                                 - 0.202000578674987 z_k1_s0124
                                 - 0.504999209683332 z_k1_s0125
                                 - 0.763001844699445 z_k1_s0126
                                 - 0.950001968440379 z_k1_s0127
                                 - 0.871999548707122 z_k1_s0130
                                 - 0.564999998105048 z_k1_s0131
                                 - 1.02900047708704 z_k1_s0132
                                 - 0.925999325347181 z_k1_s0133
                                 - 0.863002425934653 z_k1_s0134
                                 - 0.424000202655887 z_k1_s0135
                                 - 0.301000332753412 z_k1_s0136
                                 - 0.699997735704672 z_k1_s0137
                                 - 0.278000886422067 z_k1_s0140
                                 - 0.183000401526198 z_k1_s0141
                                 - 0.500000097128623 z_k1_s0142
                                 - 0.366999705315287 z_k1_s0143
                                 - 0.616000492815607 z_k1_s0144
                                 - 0.342000956262706 z_k1_s0145
                                 - 0.740002153036522 z_k1_s0146
                                 - 0.56200092723908 z_k1_s0147
                                 - 0.683001194453222 z_k1_s0150
                                 - 0.71599901962544 z_k1_s0151
                                 - 0.717999046060155 z_k1_s0152
                                 - 0.932001874574582 z_k1_s0153
                                 - 0.467999919321778 z_k1_s0154
                                 - 0.976002456138296 z_k1_s0155
                                 - 0.379000309577334 z_k1_s0156
                                 - 0.349999926273977 z_k1_s0157
                                 - 0.671003601062706 z_k1_s0160
                                 - 0.786001568230785 z_k1_s0161
                                 - 0.775002887198854 z_k1_s0162
                                 - 1.87400454520783 z_k1_s0163
                                 - 0.789001796331206 z_k1_s0164
                                 - 0.621000661523533 z_k1_s0165
                                 - 0.547001069988062 z_k1_s0166
                                 - 0.40300046678199 z_k1_s0167
                                 - 0.865000264635256 z_k1_s0170
                                 - 0.32699936652639 z_k1_s0171
                                 - 0.922998682508472 z_k1_s0172
                                 - 0.194999383899824 z_k1_s0173
                                 - 0.494998935652141 z_k1_s0174
                                 - 0.747000037936361 z_k1_s0175
                                 - 0.880997416330496 z_k1_s0176
                                 - 0.562997821529217 z_k1_s0177
                                 - 0.169999336256352 z_k1_s1000
                                 - 0.201999334351712 z_k1_s1001
                                 - 0.234000826447636 z_k1_s1002
                                 - 0.249000592117248 z_k1_s1003
                                 - 0.153000597831168 z_k1_s1004
                                 - 0.153000597831168 z_k1_s1005
                                 - 0.23899975233713 z_k1_s1006
                                 - 0.0939999757301123 z_k1_s1007
                                 - 0.369000485424474 z_k1_s1010
                                 - 1.04100277278767 z_k1_s1011
                                 - 0.657002719552947 z_k1_s1012
                                 - 0.908999554609692 z_k1_s1013
                                 - 0.872000425637524 z_k1_s1014
                                 - 0.297000414492896 z_k1_s1015
                                 - 0.24900004213815 z_k1_s1016
                                 - 0.482000447758592 z_k1_s1017
                                 - 0.228000723701281 z_k1_s1020
                                 - 0.0790000394928287 z_k1_s1021
                                 - 0.111000028391169 z_k1_s1022
                                 - 0.308999736699484 z_k1_s1023
                                 - 0.357000076847263 z_k1_s1024
                                 - 0.238000051231509 z_k1_s1025
                                 - 0.0870000367174138 z_k1_s1026
                                 - 0.375000650403016 z_k1_s1027
                                 - 0.405000466302466 z_k1_s1030
                                 - 0.75400274878468 z_k1_s1031
                                 - 1.19499979722715 z_k1_s1032
                                 - 0.34900042928091 z_k1_s1033
                                 - 0.61400265623079 z_k1_s1034
                                 - 0.556002088400948 z_k1_s1035
                                 - 0.35900030352038 z_k1_s1036
                                 - 0.401000331286547 z_k1_s1037
                                 - 0.45000063234428 z_k1_s1040
                                 - 0.720000334470003 z_k1_s1041
                                 - 0.762003034884099 z_k1_s1042
                                 - 1.18800139847656 z_k1_s1043
                                 - 0.752001854403881 z_k1_s1044
                                 - 0.669001086829346 z_k1_s1045
                                 - 0.490000839059515 z_k1_s1046
                                 - 0.613001248948581 z_k1_s1047
                                 - 0.423999885231217 z_k1_s1050
                                 - 0.672000841181499 z_k1_s1051
                                 - 1.0130009595132 z_k1_s1052
                                 - 1.15099899636308 z_k1_s1053
                                 - 0.42800017502395 z_k1_s1054
                                 - 0.368000080932745 z_k1_s1055
                                 - 0.74599955734739 z_k1_s1056
                                 - 0.513999760568038 z_k1_s1057
                                 - 0.276999613220102 z_k1_s1060
                                 - 0.263999819848781 z_k1_s1061
                                 - 0.289999406591423 z_k1_s1062
                                 - 0.143999919990246 z_k1_s1063
                                 - 0.139999983568302 z_k1_s1064
                                 - 0.268999740376212 z_k1_s1065
                                 - 0.431999157570676 z_k1_s1066
                                 - 0.191999960253669 z_k1_s1067
                                 - 0.283999701176624 z_k1_s1070
                                 - 0.114000024289215 z_k1_s1071
                                 - 0.105000046871646 z_k1_s1072
                                 - 0.193000032954548 z_k1_s1073
                                 - 0.270999216573446 z_k1_s1074
                                 - 0.261000276109529 z_k1_s1075
                                 - 0.128000030538768 z_k1_s1076
                                 - 0.506000385476704 z_k1_s1077
                                 - 0.128999755095065 z_k1_s1100
                                 - 0.158999832484435 z_k1_s1101
                                 - 0.410998849117331 z_k1_s1102
                                 - 0.289998955299645 z_k1_s1103
                                 - 0.111999850680902 z_k1_s1104
                                 - 0.104999772863462 z_k1_s1105
                                 - 0.2879989899803 z_k1_s1106
                                 - 0.497999332506156 z_k1_s1107
                                 - 0.323999303150706 z_k1_s1110
                                 - 1.21599827569931 z_k1_s1111
                                 - 0.513999001041301 z_k1_s1112
                                 - 0.465998697501417 z_k1_s1113
                                 - 0.271999105065523 z_k1_s1114
                                 - 0.871995838830761 z_k1_s1115
                                 - 1.06999794080577 z_k1_s1116
                                 - 0.49599895258869 z_k1_s1117
                                 - 1.14899732698967 z_k1_s1120
                                 - 0.631997446534607 z_k1_s1121
                                 - 0.671997676820729 z_k1_s1122
                                 - 0.35099892476766 z_k1_s1123
                                 - 0.74799811436436 z_k1_s1124
                                 - 0.52299867660076 z_k1_s1125
                                 - 1.03299665915992 z_k1_s1126
                                 - 1.08400004876778 z_k1_s1127
                                 - 0.892001437705461 z_k1_s1130
                                 - 0.502000444857014 z_k1_s1131
                                 - 0.629001228661683 z_k1_s1132
                                 - 0.735001163313324 z_k1_s1133
                                 - 0.98800386903142 z_k1_s1134
                                 - 0.349000694837424 z_k1_s1135
                                 - 0.704000907031773 z_k1_s1136
                                 - 1.5470048055868 z_k1_s1137
                                 - 0.545002200271381 z_k1_s1140
                                 - 0.321000132119257 z_k1_s1141
                                 - 0.231000837527552 z_k1_s1142
                                 - 0.190000778590061 z_k1_s1143
                                 - 0.243000701606306 z_k1_s1144
                                 - 0.487003066558102 z_k1_s1145
                                 - 0.816003945399325 z_k1_s1146
                                 - 0.307001023363157 z_k1_s1147
                                 - 0.294000279401001 z_k1_s1150
                                 - 0.236000689695445 z_k1_s1151
                                 - 0.133999833553214 z_k1_s1152
                                 - 0.420999454721692 z_k1_s1153
                                 - 0.25899912623565 z_k1_s1154
                                 - 0.10499982490071 z_k1_s1155
                                 - 0.313999105783549 z_k1_s1156
                                 - 0.375000443644022 z_k1_s1157
                                 - 0.528000659402068 z_k1_s1160
                                 - 0.333999958781036 z_k1_s1161
                                 - 0.243999859478406 z_k1_s1162
                                 - 0.296999290539094 z_k1_s1163
                                 - 0.647000316937957 z_k1_s1164
                                 - 0.216000007846405 z_k1_s1165
                                 - 0.189999857516805 z_k1_s1166
                                 - 0.424999188485154 z_k1_s1167
                                 - 0.181999868011815 z_k1_s1170
                                 - 0.558000197036835 z_k1_s1171
                                 - 0.164999852550843 z_k1_s1172
                                 - 0.336000331452116 z_k1_s1173
                                 - 0.215999898933757 z_k1_s1174
                                 - 0.389998931811666 z_k1_s1175
                                 - 0.145999848206218 z_k1_s1176
                                 - 0.205999734615664 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.17267 w_k1_ADM_s000
                                 + 0.687931 w_k1_AXP_s000
                                 + 0.702793 w_k1_DIS_s000
                                 + 0.797859 w_k1_FDX_s000
                                 + 1.04026 w_k1_IBM_s000
                                 + 0.774983 w_k1_INTC_s000
                                 + 0.97061 w_k1_KO_s000
                                 + 0.938265 w_k1_MAT_s000
                                 + 0.756624 w_k1_MSFT_s000
                                 + 0.960044 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.948897 w_k1_ADM_s000
                                 + 1.79657 w_k1_AXP_s000
                                 + 1.33625 w_k1_DIS_s000
                                 + 1.05376 w_k1_FDX_s000 + 2.3131 w_k1_IBM_s000
                                 + 1.21309 w_k1_INTC_s000
                                 + 1.56742 w_k1_KO_s000 + 2.21502 w_k1_MAT_s000
                                 + 1.11948 w_k1_MSFT_s000
                                 + 1.08297 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.27375 w_k1_ADM_s000
                                 + 0.783832 w_k1_AXP_s000
                                 + 0.781894 w_k1_DIS_s000
                                 + 0.922283 w_k1_FDX_s000
                                 + 0.643397 w_k1_IBM_s000
                                 + 0.830522 w_k1_INTC_s000
                                 + 0.811744 w_k1_KO_s000
                                 + 0.682962 w_k1_MAT_s000
                                 + 0.998586 w_k1_MSFT_s000
                                 + 0.912085 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.19673 w_k1_ADM_s000
                                 + 1.11673 w_k1_AXP_s000
                                 + 1.18234 w_k1_DIS_s000
                                 + 0.998843 w_k1_FDX_s000
                                 + 1.18367 w_k1_IBM_s000
                                 + 1.02409 w_k1_INTC_s000
                                 + 1.27934 w_k1_KO_s000 + 1.58164 w_k1_MAT_s000
                                 + 1.03142 w_k1_MSFT_s000
                                 + 1.19221 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.80882 w_k1_ADM_s000
                                 + 1.19326 w_k1_AXP_s000 + 1.8148 w_k1_DIS_s000
                                 + 1.50479 w_k1_FDX_s000
                                 + 0.963016 w_k1_IBM_s000
                                 + 1.21075 w_k1_INTC_s000
                                 + 1.33786 w_k1_KO_s000
                                 + 0.902606 w_k1_MAT_s000
                                 + 1.41349 w_k1_MSFT_s000
                                 + 1.24081 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.20804 w_k1_ADM_s000
                                 + 1.24411 w_k1_AXP_s000
                                 + 0.903968 w_k1_DIS_s000
                                 + 1.16416 w_k1_FDX_s000
                                 + 0.740733 w_k1_IBM_s000
                                 + 1.19197 w_k1_INTC_s000
                                 + 0.982757 w_k1_KO_s000
                                 + 0.890494 w_k1_MAT_s000
                                 + 1.01375 w_k1_MSFT_s000
                                 + 1.19358 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.40803 w_k1_ADM_s000
                                 + 0.734928 w_k1_AXP_s000
                                 + 1.02424 w_k1_DIS_s000
                                 + 0.915258 w_k1_FDX_s000
                                 + 1.12397 w_k1_IBM_s000
                                 + 0.777109 w_k1_INTC_s000
                                 + 1.14438 w_k1_KO_s000
                                 + 0.766102 w_k1_MAT_s000
                                 + 1.15101 w_k1_MSFT_s000
                                 + 0.992199 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.18364 w_k1_ADM_s000
                                 + 0.970492 w_k1_AXP_s000
                                 + 0.878789 w_k1_DIS_s000
                                 + 0.964848 w_k1_FDX_s000
                                 + 0.705542 w_k1_IBM_s000
                                 + 1.08746 w_k1_INTC_s000
                                 + 0.824127 w_k1_KO_s000
                                 + 0.811288 w_k1_MAT_s000
                                 + 1.07004 w_k1_MSFT_s000
                                 + 0.950029 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.735696 w_k1_ADM_s001
                                 + 0.793395 w_k1_AXP_s001
                                 + 0.973564 w_k1_DIS_s001
                                 + 0.936041 w_k1_FDX_s001
                                 + 0.759249 w_k1_IBM_s001
                                 + 1.02696 w_k1_INTC_s001
                                 + 0.792233 w_k1_KO_s001
                                 + 0.626917 w_k1_MAT_s001
                                 + 0.945366 w_k1_MSFT_s001
                                 + 0.854965 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.7926 w_k1_ADM_s001
                                 + 0.940574 w_k1_AXP_s001
                                 + 0.88424 w_k1_DIS_s001
                                 + 1.04567 w_k1_FDX_s001
                                 + 0.778627 w_k1_IBM_s001
                                 + 1.00091 w_k1_INTC_s001
                                 + 0.895298 w_k1_KO_s001
                                 + 0.791978 w_k1_MAT_s001
                                 + 0.943546 w_k1_MSFT_s001
                                 + 1.04358 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.975582 w_k1_ADM_s001
                                 + 0.866337 w_k1_AXP_s001
                                 + 1.12554 w_k1_DIS_s001
                                 + 0.982727 w_k1_FDX_s001
                                 + 1.01476 w_k1_IBM_s001
                                 + 1.13078 w_k1_INTC_s001
                                 + 1.19563 w_k1_KO_s001 + 0.89776 w_k1_MAT_s001
                                 + 1.19303 w_k1_MSFT_s001
                                 + 0.98471 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.40016 w_k1_ADM_s001
                                 + 1.46188 w_k1_AXP_s001
                                 + 0.982543 w_k1_DIS_s001
                                 + 1.17995 w_k1_FDX_s001
                                 + 1.00678 w_k1_IBM_s001
                                 + 1.12741 w_k1_INTC_s001
                                 + 1.02006 w_k1_KO_s001 + 1.19404 w_k1_MAT_s001
                                 + 1.10492 w_k1_MSFT_s001
                                 + 1.22788 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.51075 w_k1_ADM_s001
                                 + 1.4909 w_k1_AXP_s001 + 1.70802 w_k1_DIS_s001
                                 + 1.4293 w_k1_FDX_s001 + 1.08683 w_k1_IBM_s001
                                 + 1.47536 w_k1_INTC_s001 + 1.5104 w_k1_KO_s001
                                 + 1.35573 w_k1_MAT_s001
                                 + 1.43839 w_k1_MSFT_s001 + 1.5694 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.833362 w_k1_ADM_s001
                                 + 1.19988 w_k1_AXP_s001
                                 + 1.34059 w_k1_DIS_s001
                                 + 1.15887 w_k1_FDX_s001
                                 + 1.02607 w_k1_IBM_s001
                                 + 1.22455 w_k1_INTC_s001
                                 + 1.03441 w_k1_KO_s001 + 1.06566 w_k1_MAT_s001
                                 + 1.32257 w_k1_MSFT_s001
                                 + 1.14407 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.740031 w_k1_ADM_s001
                                 + 0.712685 w_k1_AXP_s001
                                 + 0.736357 w_k1_DIS_s001
                                 + 0.784112 w_k1_FDX_s001
                                 + 0.965852 w_k1_IBM_s001
                                 + 0.715703 w_k1_INTC_s001
                                 + 0.887348 w_k1_KO_s001
                                 + 1.32666 w_k1_MAT_s001
                                 + 0.88751 w_k1_MSFT_s001
                                 + 0.858754 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.954735 w_k1_ADM_s001
                                 + 0.989748 w_k1_AXP_s001
                                 + 0.748776 w_k1_DIS_s001
                                 + 0.889645 w_k1_FDX_s001
                                 + 0.580342 w_k1_IBM_s001
                                 + 0.917579 w_k1_INTC_s001
                                 + 0.833686 w_k1_KO_s001
                                 + 0.973654 w_k1_MAT_s001
                                 + 0.954328 w_k1_MSFT_s001
                                 + 0.892712 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08809 w_k1_ADM_s002
                                 + 1.03422 w_k1_AXP_s002
                                 + 1.20774 w_k1_DIS_s002
                                 + 1.08377 w_k1_FDX_s002
                                 + 1.13788 w_k1_IBM_s002
                                 + 0.896583 w_k1_INTC_s002
                                 + 1.08335 w_k1_KO_s002
                                 + 0.916624 w_k1_MAT_s002
                                 + 1.08559 w_k1_MSFT_s002
                                 + 1.10911 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.15799 w_k1_ADM_s002
                                 + 0.925844 w_k1_AXP_s002
                                 + 0.838668 w_k1_DIS_s002
                                 + 1.06143 w_k1_FDX_s002
                                 + 0.811708 w_k1_IBM_s002
                                 + 0.822158 w_k1_INTC_s002
                                 + 0.913695 w_k1_KO_s002
                                 + 0.749476 w_k1_MAT_s002
                                 + 0.934881 w_k1_MSFT_s002
                                 + 1.08951 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.37705 w_k1_ADM_s002
                                 + 1.5158 w_k1_AXP_s002 + 1.50259 w_k1_DIS_s002
                                 + 1.29074 w_k1_FDX_s002
                                 + 1.51534 w_k1_IBM_s002
                                 + 1.15861 w_k1_INTC_s002
                                 + 1.63734 w_k1_KO_s002 + 1.58659 w_k1_MAT_s002
                                 + 1.19244 w_k1_MSFT_s002
                                 + 1.31739 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.58157 w_k1_ADM_s002
                                 + 1.28217 w_k1_AXP_s002
                                 + 1.47491 w_k1_DIS_s002
                                 + 1.36241 w_k1_FDX_s002
                                 + 1.01884 w_k1_IBM_s002
                                 + 1.1704 w_k1_INTC_s002 + 1.24851 w_k1_KO_s002
                                 + 1.03321 w_k1_MAT_s002
                                 + 1.31932 w_k1_MSFT_s002
                                 + 1.26558 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.42788 w_k1_ADM_s002
                                 + 1.40181 w_k1_AXP_s002 + 1.1075 w_k1_DIS_s002
                                 + 1.24932 w_k1_FDX_s002
                                 + 1.11247 w_k1_IBM_s002
                                 + 1.16576 w_k1_INTC_s002 + 1.1416 w_k1_KO_s002
                                 + 1.20838 w_k1_MAT_s002
                                 + 1.01801 w_k1_MSFT_s002 + 1.16 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.88796 w_k1_ADM_s002
                                 + 0.697816 w_k1_AXP_s002
                                 + 0.721113 w_k1_DIS_s002
                                 + 0.836494 w_k1_FDX_s002
                                 + 0.961768 w_k1_IBM_s002
                                 + 0.845624 w_k1_INTC_s002
                                 + 0.888181 w_k1_KO_s002
                                 + 0.710337 w_k1_MAT_s002
                                 + 0.80534 w_k1_MSFT_s002
                                 + 1.03763 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.09296 w_k1_ADM_s002
                                 + 0.793739 w_k1_AXP_s002
                                 + 0.718502 w_k1_DIS_s002
                                 + 0.825731 w_k1_FDX_s002
                                 + 0.604677 w_k1_IBM_s002
                                 + 0.808602 w_k1_INTC_s002
                                 + 0.767198 w_k1_KO_s002
                                 + 0.487416 w_k1_MAT_s002
                                 + 0.812644 w_k1_MSFT_s002
                                 + 1.01445 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.18359 w_k1_ADM_s002
                                 + 1.11138 w_k1_AXP_s002
                                 + 0.927301 w_k1_DIS_s002
                                 + 0.953617 w_k1_FDX_s002
                                 + 1.34225 w_k1_IBM_s002
                                 + 1.01929 w_k1_INTC_s002
                                 + 1.05681 w_k1_KO_s002 + 1.2381 w_k1_MAT_s002
                                 + 0.996477 w_k1_MSFT_s002
                                 + 1.15054 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.86837 w_k1_ADM_s003
                                 + 1.58517 w_k1_AXP_s003
                                 + 1.57825 w_k1_DIS_s003
                                 + 1.36011 w_k1_FDX_s003
                                 + 1.29971 w_k1_IBM_s003
                                 + 0.915573 w_k1_INTC_s003
                                 + 1.27195 w_k1_KO_s003 + 1.67875 w_k1_MAT_s003
                                 + 1.08645 w_k1_MSFT_s003
                                 + 1.23925 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.1191 w_k1_ADM_s003
                                 + 1.12869 w_k1_AXP_s003
                                 + 1.51578 w_k1_DIS_s003
                                 + 1.18576 w_k1_FDX_s003
                                 + 1.35493 w_k1_IBM_s003
                                 + 1.26566 w_k1_INTC_s003
                                 + 1.69482 w_k1_KO_s003 + 1.23069 w_k1_MAT_s003
                                 + 1.18581 w_k1_MSFT_s003
                                 + 1.18544 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.10473 w_k1_ADM_s003
                                 + 0.985912 w_k1_AXP_s003
                                 + 1.08324 w_k1_DIS_s003
                                 + 0.97865 w_k1_FDX_s003
                                 + 1.35123 w_k1_IBM_s003
                                 + 1.02321 w_k1_INTC_s003
                                 + 1.13295 w_k1_KO_s003 + 1.17576 w_k1_MAT_s003
                                 + 0.926725 w_k1_MSFT_s003
                                 + 1.11538 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.17242 w_k1_ADM_s003
                                 + 0.979668 w_k1_AXP_s003
                                 + 0.944709 w_k1_DIS_s003
                                 + 1.07908 w_k1_FDX_s003
                                 + 0.639086 w_k1_IBM_s003
                                 + 0.802559 w_k1_INTC_s003
                                 + 0.770939 w_k1_KO_s003
                                 + 0.871356 w_k1_MAT_s003
                                 + 0.892536 w_k1_MSFT_s003
                                 + 0.911718 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.910774 w_k1_ADM_s003
                                 + 0.63458 w_k1_AXP_s003
                                 + 0.652837 w_k1_DIS_s003
                                 + 0.762494 w_k1_FDX_s003
                                 + 0.574424 w_k1_IBM_s003
                                 + 0.675183 w_k1_INTC_s003
                                 + 0.664022 w_k1_KO_s003
                                 + 0.485333 w_k1_MAT_s003
                                 + 0.728854 w_k1_MSFT_s003
                                 + 0.714625 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.03294 w_k1_ADM_s003
                                 + 0.694705 w_k1_AXP_s003
                                 + 0.825161 w_k1_DIS_s003
                                 + 0.781763 w_k1_FDX_s003
                                 + 1.24347 w_k1_IBM_s003
                                 + 0.834542 w_k1_INTC_s003
                                 + 0.987146 w_k1_KO_s003
                                 + 1.07404 w_k1_MAT_s003
                                 + 0.869129 w_k1_MSFT_s003
                                 + 0.967994 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.05237 w_k1_ADM_s003
                                 + 0.728447 w_k1_AXP_s003
                                 + 1.11693 w_k1_DIS_s003
                                 + 0.957484 w_k1_FDX_s003
                                 + 0.86639 w_k1_IBM_s003
                                 + 1.12672 w_k1_INTC_s003
                                 + 1.10805 w_k1_KO_s003
                                 + 0.742401 w_k1_MAT_s003
                                 + 1.11113 w_k1_MSFT_s003
                                 + 0.908906 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.936524 w_k1_ADM_s003
                                 + 0.826877 w_k1_AXP_s003
                                 + 0.66937 w_k1_DIS_s003
                                 + 0.757268 w_k1_FDX_s003
                                 + 0.895792 w_k1_IBM_s003
                                 + 0.622106 w_k1_INTC_s003
                                 + 0.727411 w_k1_KO_s003
                                 + 1.04177 w_k1_MAT_s003
                                 + 0.728726 w_k1_MSFT_s003
                                 + 0.879661 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.990815 w_k1_ADM_s004
                                 + 0.815139 w_k1_AXP_s004
                                 + 0.957668 w_k1_DIS_s004
                                 + 1.02645 w_k1_FDX_s004
                                 + 0.798052 w_k1_IBM_s004
                                 + 0.941087 w_k1_INTC_s004
                                 + 0.913324 w_k1_KO_s004
                                 + 0.626025 w_k1_MAT_s004
                                 + 0.88211 w_k1_MSFT_s004
                                 + 0.977921 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.530108 w_k1_ADM_s004
                                 + 0.64428 w_k1_AXP_s004
                                 + 0.817317 w_k1_DIS_s004
                                 + 0.770897 w_k1_FDX_s004
                                 + 1.04712 w_k1_IBM_s004
                                 + 0.952982 w_k1_INTC_s004
                                 + 1.08044 w_k1_KO_s004 + 1.0042 w_k1_MAT_s004
                                 + 0.824809 w_k1_MSFT_s004
                                 + 0.967893 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.737294 w_k1_ADM_s004
                                 + 1.47038 w_k1_AXP_s004
                                 + 1.24917 w_k1_DIS_s004
                                 + 1.02764 w_k1_FDX_s004
                                 + 1.69338 w_k1_IBM_s004
                                 + 0.886991 w_k1_INTC_s004
                                 + 1.26633 w_k1_KO_s004 + 2.16485 w_k1_MAT_s004
                                 + 0.884623 w_k1_MSFT_s004
                                 + 1.22918 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.661938 w_k1_ADM_s004
                                 + 0.596395 w_k1_AXP_s004
                                 + 0.626818 w_k1_DIS_s004
                                 + 0.762992 w_k1_FDX_s004
                                 + 0.718451 w_k1_IBM_s004
                                 + 0.769902 w_k1_INTC_s004
                                 + 0.915057 w_k1_KO_s004
                                 + 0.799607 w_k1_MAT_s004
                                 + 0.781438 w_k1_MSFT_s004
                                 + 0.944595 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.897468 w_k1_ADM_s004
                                 + 1.0951 w_k1_AXP_s004 + 1.2555 w_k1_DIS_s004
                                 + 1.26724 w_k1_FDX_s004
                                 + 0.960382 w_k1_IBM_s004
                                 + 1.08554 w_k1_INTC_s004 + 1.0869 w_k1_KO_s004
                                 + 1.15755 w_k1_MAT_s004
                                 + 1.13291 w_k1_MSFT_s004
                                 + 1.07135 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.00914 w_k1_ADM_s004
                                 + 0.696423 w_k1_AXP_s004
                                 + 1.24902 w_k1_DIS_s004
                                 + 1.07271 w_k1_FDX_s004
                                 + 1.04094 w_k1_IBM_s004
                                 + 1.1026 w_k1_INTC_s004 + 1.27216 w_k1_KO_s004
                                 + 0.905461 w_k1_MAT_s004
                                 + 1.19659 w_k1_MSFT_s004 + 1.1287 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.847754 w_k1_ADM_s004
                                 + 1.15438 w_k1_AXP_s004
                                 + 1.03825 w_k1_DIS_s004
                                 + 0.972506 w_k1_FDX_s004
                                 + 1.14021 w_k1_IBM_s004
                                 + 1.08503 w_k1_INTC_s004
                                 + 1.10265 w_k1_KO_s004 + 1.43369 w_k1_MAT_s004
                                 + 1.03616 w_k1_MSFT_s004
                                 + 0.911494 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.771476 w_k1_ADM_s004
                                 + 1.31833 w_k1_AXP_s004
                                 + 1.65902 w_k1_DIS_s004 + 1.2025 w_k1_FDX_s004
                                 + 1.37826 w_k1_IBM_s004
                                 + 1.02439 w_k1_INTC_s004
                                 + 1.30496 w_k1_KO_s004 + 1.80322 w_k1_MAT_s004
                                 + 1.10589 w_k1_MSFT_s004
                                 + 1.04897 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.0971 w_k1_ADM_s005
                                 + 1.42474 w_k1_AXP_s005
                                 + 1.43595 w_k1_DIS_s005
                                 + 1.28422 w_k1_FDX_s005
                                 + 1.51787 w_k1_IBM_s005
                                 + 1.47616 w_k1_INTC_s005
                                 + 1.58963 w_k1_KO_s005 + 1.76125 w_k1_MAT_s005
                                 + 1.44354 w_k1_MSFT_s005
                                 + 1.27555 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.743171 w_k1_ADM_s005
                                 + 0.617652 w_k1_AXP_s005
                                 + 0.636215 w_k1_DIS_s005
                                 + 0.733232 w_k1_FDX_s005
                                 + 0.657113 w_k1_IBM_s005
                                 + 0.767861 w_k1_INTC_s005
                                 + 0.627737 w_k1_KO_s005
                                 + 0.547119 w_k1_MAT_s005
                                 + 0.733459 w_k1_MSFT_s005
                                 + 0.787064 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.06922 w_k1_ADM_s005
                                 + 0.700944 w_k1_AXP_s005
                                 + 0.732816 w_k1_DIS_s005
                                 + 0.907533 w_k1_FDX_s005
                                 + 0.726415 w_k1_IBM_s005
                                 + 0.842436 w_k1_INTC_s005
                                 + 0.701549 w_k1_KO_s005
                                 + 0.473076 w_k1_MAT_s005
                                 + 0.867102 w_k1_MSFT_s005
                                 + 1.00106 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.933993 w_k1_ADM_s005
                                 + 0.998384 w_k1_AXP_s005
                                 + 1.0206 w_k1_DIS_s005
                                 + 0.966572 w_k1_FDX_s005
                                 + 1.37004 w_k1_IBM_s005
                                 + 1.12343 w_k1_INTC_s005
                                 + 1.27883 w_k1_KO_s005 + 1.16118 w_k1_MAT_s005
                                 + 1.02201 w_k1_MSFT_s005
                                 + 1.10571 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.37176 w_k1_ADM_s005
                                 + 1.22504 w_k1_AXP_s005
                                 + 1.35356 w_k1_DIS_s005
                                 + 1.39693 w_k1_FDX_s005
                                 + 0.982755 w_k1_IBM_s005
                                 + 1.50881 w_k1_INTC_s005
                                 + 1.17221 w_k1_KO_s005
                                 + 0.907467 w_k1_MAT_s005
                                 + 1.35635 w_k1_MSFT_s005
                                 + 1.17761 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.84428 w_k1_ADM_s005
                                 + 0.85583 w_k1_AXP_s005 + 1.4361 w_k1_DIS_s005
                                 + 0.962776 w_k1_FDX_s005
                                 + 1.24192 w_k1_IBM_s005
                                 + 0.948508 w_k1_INTC_s005
                                 + 1.31007 w_k1_KO_s005
                                 + 0.959572 w_k1_MAT_s005
                                 + 1.16652 w_k1_MSFT_s005
                                 + 1.05485 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.00504 w_k1_ADM_s005
                                 + 0.848307 w_k1_AXP_s005
                                 + 0.795442 w_k1_DIS_s005
                                 + 0.827669 w_k1_FDX_s005
                                 + 0.921682 w_k1_IBM_s005
                                 + 1.06831 w_k1_INTC_s005
                                 + 0.993032 w_k1_KO_s005
                                 + 1.11476 w_k1_MAT_s005
                                 + 1.11839 w_k1_MSFT_s005
                                 + 1.03201 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.23463 w_k1_ADM_s005
                                 + 1.10246 w_k1_AXP_s005
                                 + 0.961614 w_k1_DIS_s005
                                 + 1.13463 w_k1_FDX_s005
                                 + 0.863892 w_k1_IBM_s005
                                 + 0.897843 w_k1_INTC_s005
                                 + 0.908643 w_k1_KO_s005
                                 + 0.697426 w_k1_MAT_s005
                                 + 0.961619 w_k1_MSFT_s005
                                 + 1.07871 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.679433 w_k1_ADM_s006
                                 + 0.645554 w_k1_AXP_s006
                                 + 0.637756 w_k1_DIS_s006
                                 + 0.80209 w_k1_FDX_s006
                                 + 0.678774 w_k1_IBM_s006
                                 + 0.750284 w_k1_INTC_s006
                                 + 0.773294 w_k1_KO_s006
                                 + 0.571851 w_k1_MAT_s006
                                 + 0.786357 w_k1_MSFT_s006
                                 + 0.835309 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.803603 w_k1_ADM_s006
                                 + 0.905736 w_k1_AXP_s006
                                 + 0.74209 w_k1_DIS_s006
                                 + 0.766334 w_k1_FDX_s006
                                 + 1.17862 w_k1_IBM_s006
                                 + 0.93637 w_k1_INTC_s006
                                 + 0.985354 w_k1_KO_s006
                                 + 0.96024 w_k1_MAT_s006
                                 + 0.786498 w_k1_MSFT_s006
                                 + 0.941565 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.09878 w_k1_ADM_s006
                                 + 1.52786 w_k1_AXP_s006
                                 + 1.52926 w_k1_DIS_s006
                                 + 1.23558 w_k1_FDX_s006
                                 + 1.32727 w_k1_IBM_s006
                                 + 1.01908 w_k1_INTC_s006
                                 + 1.28549 w_k1_KO_s006 + 1.51201 w_k1_MAT_s006
                                 + 1.12472 w_k1_MSFT_s006
                                 + 1.23447 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.96418 w_k1_ADM_s006
                                 + 1.39494 w_k1_AXP_s006
                                 + 0.933223 w_k1_DIS_s006
                                 + 1.01217 w_k1_FDX_s006
                                 + 1.03403 w_k1_IBM_s006
                                 + 1.14033 w_k1_INTC_s006
                                 + 1.02386 w_k1_KO_s006
                                 + 0.893399 w_k1_MAT_s006
                                 + 0.942951 w_k1_MSFT_s006
                                 + 1.06147 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.03402 w_k1_ADM_s006
                                 + 1.15053 w_k1_AXP_s006
                                 + 1.06285 w_k1_DIS_s006
                                 + 0.95403 w_k1_FDX_s006
                                 + 1.56924 w_k1_IBM_s006
                                 + 1.15629 w_k1_INTC_s006
                                 + 1.25298 w_k1_KO_s006 + 1.24656 w_k1_MAT_s006
                                 + 0.945815 w_k1_MSFT_s006
                                 + 1.03864 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.899883 w_k1_ADM_s006
                                 + 0.930721 w_k1_AXP_s006
                                 + 0.892252 w_k1_DIS_s006
                                 + 0.978244 w_k1_FDX_s006
                                 + 0.635972 w_k1_IBM_s006
                                 + 1.30712 w_k1_INTC_s006
                                 + 0.849735 w_k1_KO_s006
                                 + 0.542606 w_k1_MAT_s006
                                 + 1.02584 w_k1_MSFT_s006
                                 + 0.932873 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.90173 w_k1_ADM_s006
                                 + 0.852788 w_k1_AXP_s006
                                 + 1.05127 w_k1_DIS_s006
                                 + 0.890707 w_k1_FDX_s006
                                 + 0.968221 w_k1_IBM_s006
                                 + 1.01361 w_k1_INTC_s006
                                 + 1.16655 w_k1_KO_s006
                                 + 0.731145 w_k1_MAT_s006
                                 + 1.07413 w_k1_MSFT_s006
                                 + 1.07968 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.12671 w_k1_ADM_s006
                                 + 1.31167 w_k1_AXP_s006
                                 + 1.58798 w_k1_DIS_s006
                                 + 1.38952 w_k1_FDX_s006
                                 + 0.84939 w_k1_IBM_s006
                                 + 1.57023 w_k1_INTC_s006
                                 + 1.12256 w_k1_KO_s006
                                 + 0.804962 w_k1_MAT_s006
                                 + 1.28095 w_k1_MSFT_s006
                                 + 1.08577 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.935685 w_k1_ADM_s007
                                 + 1.19735 w_k1_AXP_s007
                                 + 1.10199 w_k1_DIS_s007
                                 + 1.20963 w_k1_FDX_s007
                                 + 1.17849 w_k1_IBM_s007
                                 + 0.992332 w_k1_INTC_s007
                                 + 1.04471 w_k1_KO_s007 + 1.27669 w_k1_MAT_s007
                                 + 1.21033 w_k1_MSFT_s007
                                 + 1.33306 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.998888 w_k1_ADM_s007
                                 + 0.83838 w_k1_AXP_s007
                                 + 0.753612 w_k1_DIS_s007
                                 + 0.912992 w_k1_FDX_s007
                                 + 0.978257 w_k1_IBM_s007
                                 + 0.793927 w_k1_INTC_s007
                                 + 0.871897 w_k1_KO_s007
                                 + 1.01467 w_k1_MAT_s007
                                 + 0.91207 w_k1_MSFT_s007
                                 + 1.09821 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.21589 w_k1_ADM_s007
                                 + 1.02057 w_k1_AXP_s007
                                 + 0.881294 w_k1_DIS_s007
                                 + 0.901366 w_k1_FDX_s007
                                 + 1.45525 w_k1_IBM_s007
                                 + 0.753346 w_k1_INTC_s007
                                 + 1.13006 w_k1_KO_s007
                                 + 0.965755 w_k1_MAT_s007
                                 + 0.781714 w_k1_MSFT_s007
                                 + 1.23719 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.704633 w_k1_ADM_s007
                                 + 0.730465 w_k1_AXP_s007
                                 + 0.678972 w_k1_DIS_s007
                                 + 0.792717 w_k1_FDX_s007
                                 + 0.708299 w_k1_IBM_s007
                                 + 0.842038 w_k1_INTC_s007
                                 + 0.770392 w_k1_KO_s007
                                 + 0.732139 w_k1_MAT_s007
                                 + 0.811392 w_k1_MSFT_s007
                                 + 0.896281 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.02578 w_k1_ADM_s007
                                 + 0.959692 w_k1_AXP_s007
                                 + 1.04913 w_k1_DIS_s007
                                 + 1.09059 w_k1_FDX_s007
                                 + 0.672028 w_k1_IBM_s007
                                 + 0.797576 w_k1_INTC_s007
                                 + 0.895494 w_k1_KO_s007
                                 + 0.625953 w_k1_MAT_s007
                                 + 1.04618 w_k1_MSFT_s007 + 1.0723 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.33019 w_k1_ADM_s007
                                 + 1.50315 w_k1_AXP_s007
                                 + 1.65206 w_k1_DIS_s007
                                 + 1.41371 w_k1_FDX_s007
                                 + 0.96746 w_k1_IBM_s007
                                 + 0.974303 w_k1_INTC_s007
                                 + 1.18924 w_k1_KO_s007 + 1.23808 w_k1_MAT_s007
                                 + 1.14453 w_k1_MSFT_s007 + 1.3449 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.53616 w_k1_ADM_s007
                                 + 1.02613 w_k1_AXP_s007
                                 + 1.43514 w_k1_DIS_s007
                                 + 1.20551 w_k1_FDX_s007
                                 + 1.06798 w_k1_IBM_s007
                                 + 1.0251 w_k1_INTC_s007 + 1.2896 w_k1_KO_s007
                                 + 0.932585 w_k1_MAT_s007
                                 + 1.10065 w_k1_MSFT_s007
                                 + 1.39844 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.5906 w_k1_ADM_s007
                                 + 1.09702 w_k1_AXP_s007
                                 + 1.69096 w_k1_DIS_s007 + 1.2594 w_k1_FDX_s007
                                 + 1.73973 w_k1_IBM_s007
                                 + 0.927794 w_k1_INTC_s007
                                 + 1.51815 w_k1_KO_s007 + 1.47819 w_k1_MAT_s007
                                 + 1.15337 w_k1_MSFT_s007
                                 + 1.29893 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.945507 w_k1_ADM_s010
                                 + 0.962861 w_k1_AXP_s010
                                 + 1.02611 w_k1_DIS_s010
                                 + 0.993197 w_k1_FDX_s010
                                 + 0.988695 w_k1_IBM_s010
                                 + 1.16784 w_k1_INTC_s010
                                 + 1.06506 w_k1_KO_s010
                                 + 0.739278 w_k1_MAT_s010
                                 + 1.0287 w_k1_MSFT_s010 + 1.09776 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.890829 w_k1_ADM_s010
                                 + 1.0205 w_k1_AXP_s010
                                 + 0.956888 w_k1_DIS_s010
                                 + 0.842474 w_k1_FDX_s010
                                 + 1.22297 w_k1_IBM_s010
                                 + 1.0182 w_k1_INTC_s010 + 1.10552 w_k1_KO_s010
                                 + 1.25784 w_k1_MAT_s010
                                 + 0.962494 w_k1_MSFT_s010
                                 + 1.06634 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.832335 w_k1_ADM_s010
                                 + 1.21597 w_k1_AXP_s010
                                 + 0.843579 w_k1_DIS_s010
                                 + 1.02204 w_k1_FDX_s010
                                 + 0.885438 w_k1_IBM_s010
                                 + 1.23482 w_k1_INTC_s010
                                 + 0.888609 w_k1_KO_s010
                                 + 0.808582 w_k1_MAT_s010
                                 + 1.02901 w_k1_MSFT_s010
                                 + 1.08179 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.964635 w_k1_ADM_s010
                                 + 1.23885 w_k1_AXP_s010
                                 + 1.25114 w_k1_DIS_s010
                                 + 1.14698 w_k1_FDX_s010
                                 + 0.749369 w_k1_IBM_s010
                                 + 1.27579 w_k1_INTC_s010
                                 + 1.02245 w_k1_KO_s010
                                 + 0.652929 w_k1_MAT_s010
                                 + 1.23595 w_k1_MSFT_s010
                                 + 1.16674 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.22226 w_k1_ADM_s010
                                 + 1.31547 w_k1_AXP_s010
                                 + 1.43402 w_k1_DIS_s010 + 1.1922 w_k1_FDX_s010
                                 + 1.24841 w_k1_IBM_s010
                                 + 1.15583 w_k1_INTC_s010
                                 + 1.30986 w_k1_KO_s010 + 1.17237 w_k1_MAT_s010
                                 + 1.17257 w_k1_MSFT_s010
                                 + 1.28869 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.767646 w_k1_ADM_s010
                                 + 0.723354 w_k1_AXP_s010
                                 + 0.733558 w_k1_DIS_s010
                                 + 0.910285 w_k1_FDX_s010
                                 + 0.59018 w_k1_IBM_s010
                                 + 1.06533 w_k1_INTC_s010
                                 + 0.812482 w_k1_KO_s010
                                 + 0.643396 w_k1_MAT_s010
                                 + 1.03515 w_k1_MSFT_s010
                                 + 0.949335 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.946194 w_k1_ADM_s010
                                 + 1.60893 w_k1_AXP_s010
                                 + 1.24245 w_k1_DIS_s010
                                 + 1.11074 w_k1_FDX_s010
                                 + 1.37429 w_k1_IBM_s010
                                 + 1.09756 w_k1_INTC_s010
                                 + 1.20554 w_k1_KO_s010 + 1.47246 w_k1_MAT_s010
                                 + 1.06934 w_k1_MSFT_s010
                                 + 1.09617 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.786059 w_k1_ADM_s010
                                 + 0.820017 w_k1_AXP_s010
                                 + 0.606077 w_k1_DIS_s010
                                 + 0.749966 w_k1_FDX_s010
                                 + 0.818606 w_k1_IBM_s010
                                 + 0.852234 w_k1_INTC_s010
                                 + 0.782867 w_k1_KO_s010
                                 + 0.775784 w_k1_MAT_s010
                                 + 0.793459 w_k1_MSFT_s010
                                 + 0.889196 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.780252 w_k1_ADM_s011
                                 + 0.927881 w_k1_AXP_s011
                                 + 0.803496 w_k1_DIS_s011
                                 + 0.83228 w_k1_FDX_s011
                                 + 1.09141 w_k1_IBM_s011
                                 + 0.860764 w_k1_INTC_s011
                                 + 0.88601 w_k1_KO_s011 + 1.04688 w_k1_MAT_s011
                                 + 0.883888 w_k1_MSFT_s011
                                 + 1.01013 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.914677 w_k1_ADM_s011
                                 + 0.997448 w_k1_AXP_s011
                                 + 1.17528 w_k1_DIS_s011
                                 + 1.00711 w_k1_FDX_s011
                                 + 1.14467 w_k1_IBM_s011
                                 + 0.948132 w_k1_INTC_s011
                                 + 1.00681 w_k1_KO_s011 + 1.07046 w_k1_MAT_s011
                                 + 0.941502 w_k1_MSFT_s011
                                 + 1.01005 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.917755 w_k1_ADM_s011
                                 + 1.67465 w_k1_AXP_s011
                                 + 1.12865 w_k1_DIS_s011 + 1.2731 w_k1_FDX_s011
                                 + 1.05503 w_k1_IBM_s011
                                 + 0.960583 w_k1_INTC_s011
                                 + 1.01866 w_k1_KO_s011 + 1.34139 w_k1_MAT_s011
                                 + 0.946057 w_k1_MSFT_s011
                                 + 1.14486 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.862063 w_k1_ADM_s011
                                 + 0.968691 w_k1_AXP_s011
                                 + 0.955444 w_k1_DIS_s011
                                 + 0.982453 w_k1_FDX_s011
                                 + 0.684265 w_k1_IBM_s011
                                 + 1.08863 w_k1_INTC_s011
                                 + 0.75655 w_k1_KO_s011
                                 + 0.495642 w_k1_MAT_s011
                                 + 1.03114 w_k1_MSFT_s011
                                 + 0.940358 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.1131 w_k1_ADM_s011
                                 + 1.34779 w_k1_AXP_s011
                                 + 1.28133 w_k1_DIS_s011
                                 + 1.16181 w_k1_FDX_s011
                                 + 1.30684 w_k1_IBM_s011
                                 + 1.15663 w_k1_INTC_s011
                                 + 1.25538 w_k1_KO_s011 + 1.30013 w_k1_MAT_s011
                                 + 1.23597 w_k1_MSFT_s011
                                 + 1.15356 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.760177 w_k1_ADM_s011
                                 + 0.796226 w_k1_AXP_s011
                                 + 0.68997 w_k1_DIS_s011
                                 + 0.748663 w_k1_FDX_s011
                                 + 0.735167 w_k1_IBM_s011
                                 + 0.854987 w_k1_INTC_s011
                                 + 0.763886 w_k1_KO_s011
                                 + 0.662846 w_k1_MAT_s011
                                 + 0.926491 w_k1_MSFT_s011
                                 + 0.861876 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.10189 w_k1_ADM_s011
                                 + 1.1848 w_k1_AXP_s011 + 1.26636 w_k1_DIS_s011
                                 + 1.16695 w_k1_FDX_s011
                                 + 0.903318 w_k1_IBM_s011
                                 + 0.950972 w_k1_INTC_s011
                                 + 1.09419 w_k1_KO_s011
                                 + 0.778089 w_k1_MAT_s011
                                 + 1.09944 w_k1_MSFT_s011
                                 + 1.23212 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.884033 w_k1_ADM_s011
                                 + 1.25527 w_k1_AXP_s011
                                 + 1.47826 w_k1_DIS_s011
                                 + 1.12176 w_k1_FDX_s011
                                 + 1.50766 w_k1_IBM_s011
                                 + 0.971657 w_k1_INTC_s011
                                 + 1.44367 w_k1_KO_s011 + 1.78586 w_k1_MAT_s011
                                 + 1.17275 w_k1_MSFT_s011
                                 + 1.16877 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.937748 w_k1_ADM_s012
                                 + 1.33722 w_k1_AXP_s012
                                 + 0.857136 w_k1_DIS_s012
                                 + 0.890975 w_k1_FDX_s012
                                 + 1.01404 w_k1_IBM_s012
                                 + 0.974293 w_k1_INTC_s012
                                 + 0.930721 w_k1_KO_s012
                                 + 1.22506 w_k1_MAT_s012
                                 + 0.916845 w_k1_MSFT_s012
                                 + 1.00029 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.814708 w_k1_ADM_s012
                                 + 0.869238 w_k1_AXP_s012
                                 + 0.657993 w_k1_DIS_s012
                                 + 0.747338 w_k1_FDX_s012
                                 + 0.945704 w_k1_IBM_s012
                                 + 0.820534 w_k1_INTC_s012
                                 + 0.804478 w_k1_KO_s012
                                 + 0.924112 w_k1_MAT_s012
                                 + 0.803427 w_k1_MSFT_s012
                                 + 0.899838 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.18403 w_k1_ADM_s012
                                 + 1.13275 w_k1_AXP_s012
                                 + 1.26386 w_k1_DIS_s012
                                 + 1.18212 w_k1_FDX_s012
                                 + 0.759588 w_k1_IBM_s012
                                 + 1.24183 w_k1_INTC_s012
                                 + 1.08627 w_k1_KO_s012 + 1.05567 w_k1_MAT_s012
                                 + 1.27768 w_k1_MSFT_s012
                                 + 1.20071 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.25677 w_k1_ADM_s012
                                 + 1.32442 w_k1_AXP_s012
                                 + 1.55981 w_k1_DIS_s012
                                 + 1.16414 w_k1_FDX_s012
                                 + 1.43701 w_k1_IBM_s012
                                 + 1.2941 w_k1_INTC_s012 + 1.58647 w_k1_KO_s012
                                 + 1.80461 w_k1_MAT_s012
                                 + 1.4304 w_k1_MSFT_s012 + 1.35118 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.50685 w_k1_ADM_s012
                                 + 1.71124 w_k1_AXP_s012
                                 + 1.95314 w_k1_DIS_s012
                                 + 1.60087 w_k1_FDX_s012
                                 + 1.34941 w_k1_IBM_s012
                                 + 1.64931 w_k1_INTC_s012
                                 + 1.67873 w_k1_KO_s012 + 1.42785 w_k1_MAT_s012
                                 + 1.64933 w_k1_MSFT_s012
                                 + 1.45194 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.0665 w_k1_ADM_s012
                                 + 0.932359 w_k1_AXP_s012
                                 + 1.09672 w_k1_DIS_s012
                                 + 1.03522 w_k1_FDX_s012
                                 + 1.02609 w_k1_IBM_s012
                                 + 1.10721 w_k1_INTC_s012
                                 + 1.32595 w_k1_KO_s012
                                 + 0.946851 w_k1_MAT_s012
                                 + 1.12405 w_k1_MSFT_s012
                                 + 1.09832 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.839089 w_k1_ADM_s012
                                 + 1.05079 w_k1_AXP_s012
                                 + 0.971375 w_k1_DIS_s012
                                 + 0.931824 w_k1_FDX_s012
                                 + 1.34305 w_k1_IBM_s012
                                 + 1.06579 w_k1_INTC_s012
                                 + 1.12914 w_k1_KO_s012 + 1.45066 w_k1_MAT_s012
                                 + 1.10178 w_k1_MSFT_s012
                                 + 1.08166 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.913531 w_k1_ADM_s012
                                 + 0.861333 w_k1_AXP_s012
                                 + 0.738632 w_k1_DIS_s012
                                 + 0.971224 w_k1_FDX_s012
                                 + 0.703999 w_k1_IBM_s012
                                 + 1.01228 w_k1_INTC_s012
                                 + 0.815466 w_k1_KO_s012
                                 + 0.714607 w_k1_MAT_s012
                                 + 0.974201 w_k1_MSFT_s012
                                 + 0.940788 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.645874 w_k1_ADM_s013
                                 + 0.789521 w_k1_AXP_s013
                                 + 0.696471 w_k1_DIS_s013
                                 + 0.787266 w_k1_FDX_s013
                                 + 1.05578 w_k1_IBM_s013
                                 + 0.771962 w_k1_INTC_s013
                                 + 0.760755 w_k1_KO_s013
                                 + 0.974409 w_k1_MAT_s013
                                 + 0.898871 w_k1_MSFT_s013
                                 + 0.861113 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.13909 w_k1_ADM_s013
                                 + 1.21725 w_k1_AXP_s013
                                 + 1.35306 w_k1_DIS_s013
                                 + 1.16098 w_k1_FDX_s013
                                 + 1.49323 w_k1_IBM_s013
                                 + 1.08521 w_k1_INTC_s013
                                 + 1.22132 w_k1_KO_s013 + 1.57107 w_k1_MAT_s013
                                 + 1.2114 w_k1_MSFT_s013 + 1.1084 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.09327 w_k1_ADM_s013
                                 + 1.00445 w_k1_AXP_s013
                                 + 0.985802 w_k1_DIS_s013
                                 + 1.05314 w_k1_FDX_s013
                                 + 0.761122 w_k1_IBM_s013
                                 + 0.975344 w_k1_INTC_s013
                                 + 0.760949 w_k1_KO_s013
                                 + 0.697119 w_k1_MAT_s013
                                 + 1.01318 w_k1_MSFT_s013
                                 + 0.871466 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.15379 w_k1_ADM_s013
                                 + 1.29411 w_k1_AXP_s013 + 1.3319 w_k1_DIS_s013
                                 + 1.26887 w_k1_FDX_s013
                                 + 1.04649 w_k1_IBM_s013
                                 + 1.24071 w_k1_INTC_s013 + 1.1144 w_k1_KO_s013
                                 + 0.86876 w_k1_MAT_s013
                                 + 1.26931 w_k1_MSFT_s013
                                 + 1.08383 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.706526 w_k1_ADM_s013
                                 + 0.608928 w_k1_AXP_s013
                                 + 0.547961 w_k1_DIS_s013
                                 + 0.778858 w_k1_FDX_s013
                                 + 0.680826 w_k1_IBM_s013
                                 + 0.830516 w_k1_INTC_s013
                                 + 0.629486 w_k1_KO_s013
                                 + 0.586454 w_k1_MAT_s013
                                 + 0.85681 w_k1_MSFT_s013
                                 + 0.735195 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.942754 w_k1_ADM_s013
                                 + 0.911808 w_k1_AXP_s013
                                 + 0.910195 w_k1_DIS_s013
                                 + 0.901083 w_k1_FDX_s013
                                 + 0.989878 w_k1_IBM_s013
                                 + 0.889126 w_k1_INTC_s013
                                 + 0.89647 w_k1_KO_s013
                                 + 0.748943 w_k1_MAT_s013
                                 + 1.06157 w_k1_MSFT_s013
                                 + 0.867469 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.53539 w_k1_ADM_s013
                                 + 1.60324 w_k1_AXP_s013
                                 + 1.94382 w_k1_DIS_s013
                                 + 1.31369 w_k1_FDX_s013
                                 + 1.76786 w_k1_IBM_s013
                                 + 0.998164 w_k1_INTC_s013
                                 + 1.80182 w_k1_KO_s013 + 1.82076 w_k1_MAT_s013
                                 + 1.44709 w_k1_MSFT_s013
                                 + 1.34056 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.856366 w_k1_ADM_s013
                                 + 0.895147 w_k1_AXP_s013
                                 + 0.957524 w_k1_DIS_s013
                                 + 0.887893 w_k1_FDX_s013
                                 + 1.27427 w_k1_IBM_s013
                                 + 0.752206 w_k1_INTC_s013
                                 + 0.947211 w_k1_KO_s013
                                 + 1.19256 w_k1_MAT_s013
                                 + 0.905855 w_k1_MSFT_s013
                                 + 0.889713 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.730615 w_k1_ADM_s014
                                 + 0.649423 w_k1_AXP_s014
                                 + 0.504166 w_k1_DIS_s014
                                 + 0.690533 w_k1_FDX_s014
                                 + 0.742322 w_k1_IBM_s014
                                 + 0.808243 w_k1_INTC_s014
                                 + 0.643475 w_k1_KO_s014
                                 + 0.579299 w_k1_MAT_s014
                                 + 0.684832 w_k1_MSFT_s014
                                 + 0.708449 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.26821 w_k1_ADM_s014
                                 + 1.68195 w_k1_AXP_s014
                                 + 1.61572 w_k1_DIS_s014
                                 + 1.30761 w_k1_FDX_s014
                                 + 1.74649 w_k1_IBM_s014
                                 + 1.21676 w_k1_INTC_s014
                                 + 1.40974 w_k1_KO_s014 + 1.36919 w_k1_MAT_s014
                                 + 1.20645 w_k1_MSFT_s014 + 1.102 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.826527 w_k1_ADM_s014
                                 + 0.821034 w_k1_AXP_s014
                                 + 0.714405 w_k1_DIS_s014
                                 + 0.791915 w_k1_FDX_s014
                                 + 1.0408 w_k1_IBM_s014
                                 + 0.777043 w_k1_INTC_s014
                                 + 0.740797 w_k1_KO_s014
                                 + 0.665546 w_k1_MAT_s014
                                 + 0.679102 w_k1_MSFT_s014
                                 + 0.783614 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.960162 w_k1_ADM_s014
                                 + 1.00166 w_k1_AXP_s014
                                 + 0.801889 w_k1_DIS_s014
                                 + 0.934336 w_k1_FDX_s014
                                 + 0.725449 w_k1_IBM_s014
                                 + 1.16214 w_k1_INTC_s014
                                 + 0.689486 w_k1_KO_s014
                                 + 0.566635 w_k1_MAT_s014
                                 + 0.874797 w_k1_MSFT_s014
                                 + 0.802308 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.12227 w_k1_ADM_s014
                                 + 0.947534 w_k1_AXP_s014
                                 + 0.943934 w_k1_DIS_s014
                                 + 0.969844 w_k1_FDX_s014
                                 + 0.899593 w_k1_IBM_s014
                                 + 1.09808 w_k1_INTC_s014
                                 + 0.979951 w_k1_KO_s014
                                 + 0.594715 w_k1_MAT_s014
                                 + 0.997515 w_k1_MSFT_s014
                                 + 0.934158 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.28 w_k1_ADM_s014
                                 + 1.39272 w_k1_AXP_s014
                                 + 1.42916 w_k1_DIS_s014
                                 + 1.47198 w_k1_FDX_s014
                                 + 0.910524 w_k1_IBM_s014
                                 + 1.19416 w_k1_INTC_s014 + 1.0393 w_k1_KO_s014
                                 + 1.18253 w_k1_MAT_s014
                                 + 1.19766 w_k1_MSFT_s014
                                 + 1.14422 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.970366 w_k1_ADM_s014
                                 + 1.13383 w_k1_AXP_s014
                                 + 0.996169 w_k1_DIS_s014
                                 + 0.954728 w_k1_FDX_s014
                                 + 1.3032 w_k1_IBM_s014
                                 + 1.19687 w_k1_INTC_s014
                                 + 1.07475 w_k1_KO_s014
                                 + 0.690344 w_k1_MAT_s014
                                 + 0.877385 w_k1_MSFT_s014
                                 + 0.925798 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.00505 w_k1_ADM_s014
                                 + 1.37592 w_k1_AXP_s014
                                 + 1.47084 w_k1_DIS_s014
                                 + 1.13186 w_k1_FDX_s014
                                 + 1.17444 w_k1_IBM_s014
                                 + 1.38882 w_k1_INTC_s014
                                 + 1.16865 w_k1_KO_s014 + 1.15722 w_k1_MAT_s014
                                 + 1.0207 w_k1_MSFT_s014 + 1.00389 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.25787 w_k1_ADM_s015
                                 + 1.2104 w_k1_AXP_s015 + 1.26424 w_k1_DIS_s015
                                 + 1.15758 w_k1_FDX_s015
                                 + 0.844041 w_k1_IBM_s015
                                 + 1.2652 w_k1_INTC_s015 + 1.10717 w_k1_KO_s015
                                 + 0.689821 w_k1_MAT_s015
                                 + 1.06638 w_k1_MSFT_s015
                                 + 1.04624 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.806788 w_k1_ADM_s015
                                 + 0.853556 w_k1_AXP_s015
                                 + 0.68069 w_k1_DIS_s015
                                 + 0.801811 w_k1_FDX_s015
                                 + 0.696114 w_k1_IBM_s015
                                 + 0.961171 w_k1_INTC_s015
                                 + 0.730436 w_k1_KO_s015
                                 + 0.590822 w_k1_MAT_s015
                                 + 0.701238 w_k1_MSFT_s015
                                 + 0.822513 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.20971 w_k1_ADM_s015
                                 + 1.15664 w_k1_AXP_s015
                                 + 0.861283 w_k1_DIS_s015
                                 + 1.04376 w_k1_FDX_s015
                                 + 0.76867 w_k1_IBM_s015
                                 + 1.12931 w_k1_INTC_s015
                                 + 1.00825 w_k1_KO_s015
                                 + 0.805615 w_k1_MAT_s015
                                 + 0.949987 w_k1_MSFT_s015
                                 + 0.894544 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.985798 w_k1_ADM_s015
                                 + 0.940779 w_k1_AXP_s015
                                 + 1.13256 w_k1_DIS_s015
                                 + 0.987007 w_k1_FDX_s015
                                 + 1.07313 w_k1_IBM_s015
                                 + 1.11875 w_k1_INTC_s015
                                 + 1.11475 w_k1_KO_s015
                                 + 0.902808 w_k1_MAT_s015
                                 + 0.918467 w_k1_MSFT_s015
                                 + 1.03357 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.10461 w_k1_ADM_s015
                                 + 0.894492 w_k1_AXP_s015
                                 + 0.859927 w_k1_DIS_s015
                                 + 0.865739 w_k1_FDX_s015
                                 + 1.33001 w_k1_IBM_s015
                                 + 0.964843 w_k1_INTC_s015
                                 + 1.21101 w_k1_KO_s015 + 1.19509 w_k1_MAT_s015
                                 + 0.89603 w_k1_MSFT_s015
                                 + 0.988891 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.08047 w_k1_ADM_s015
                                 + 1.5705 w_k1_AXP_s015 + 1.50135 w_k1_DIS_s015
                                 + 1.25118 w_k1_FDX_s015 + 1.2971 w_k1_IBM_s015
                                 + 1.49503 w_k1_INTC_s015
                                 + 1.24015 w_k1_KO_s015 + 1.68074 w_k1_MAT_s015
                                 + 1.13518 w_k1_MSFT_s015
                                 + 0.994672 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.788446 w_k1_ADM_s015
                                 + 0.742093 w_k1_AXP_s015
                                 + 0.853282 w_k1_DIS_s015
                                 + 0.792005 w_k1_FDX_s015
                                 + 1.12907 w_k1_IBM_s015
                                 + 1.01294 w_k1_INTC_s015
                                 + 1.01034 w_k1_KO_s015
                                 + 0.756988 w_k1_MAT_s015
                                 + 0.808948 w_k1_MSFT_s015
                                 + 0.853734 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.740737 w_k1_ADM_s015
                                 + 0.681294 w_k1_AXP_s015
                                 + 0.73527 w_k1_DIS_s015
                                 + 0.71007 w_k1_FDX_s015
                                 + 0.906569 w_k1_IBM_s015
                                 + 0.907005 w_k1_INTC_s015
                                 + 0.846376 w_k1_KO_s015
                                 + 0.800304 w_k1_MAT_s015
                                 + 0.756229 w_k1_MSFT_s015
                                 + 0.805546 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.46452 w_k1_ADM_s016
                                 + 1.65009 w_k1_AXP_s016
                                 + 1.57517 w_k1_DIS_s016
                                 + 1.37605 w_k1_FDX_s016
                                 + 1.25498 w_k1_IBM_s016
                                 + 1.06971 w_k1_INTC_s016
                                 + 1.25691 w_k1_KO_s016 + 1.20749 w_k1_MAT_s016
                                 + 1.23738 w_k1_MSFT_s016
                                 + 1.15732 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.12131 w_k1_ADM_s016
                                 + 1.38675 w_k1_AXP_s016
                                 + 1.19657 w_k1_DIS_s016
                                 + 1.10829 w_k1_FDX_s016 + 1.2212 w_k1_IBM_s016
                                 + 0.958259 w_k1_INTC_s016
                                 + 0.981199 w_k1_KO_s016
                                 + 1.04194 w_k1_MAT_s016
                                 + 1.03058 w_k1_MSFT_s016
                                 + 1.09397 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.976854 w_k1_ADM_s016
                                 + 0.897296 w_k1_AXP_s016
                                 + 0.783566 w_k1_DIS_s016
                                 + 0.984748 w_k1_FDX_s016
                                 + 0.81383 w_k1_IBM_s016
                                 + 0.938206 w_k1_INTC_s016
                                 + 0.845749 w_k1_KO_s016
                                 + 0.652017 w_k1_MAT_s016
                                 + 0.916498 w_k1_MSFT_s016
                                 + 1.03529 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.852743 w_k1_ADM_s016
                                 + 0.874045 w_k1_AXP_s016
                                 + 0.76133 w_k1_DIS_s016
                                 + 0.780161 w_k1_FDX_s016
                                 + 0.898637 w_k1_IBM_s016
                                 + 0.818393 w_k1_INTC_s016
                                 + 0.886379 w_k1_KO_s016
                                 + 0.988215 w_k1_MAT_s016
                                 + 0.837309 w_k1_MSFT_s016
                                 + 0.857038 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.9387 w_k1_ADM_s016
                                 + 1.06541 w_k1_AXP_s016
                                 + 1.04348 w_k1_DIS_s016
                                 + 1.04996 w_k1_FDX_s016
                                 + 0.764667 w_k1_IBM_s016
                                 + 1.00027 w_k1_INTC_s016
                                 + 0.904074 w_k1_KO_s016
                                 + 0.661347 w_k1_MAT_s016
                                 + 1.18693 w_k1_MSFT_s016
                                 + 0.96203 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.02031 w_k1_ADM_s016
                                 + 0.841417 w_k1_AXP_s016
                                 + 1.03271 w_k1_DIS_s016
                                 + 0.82433 w_k1_FDX_s016 + 1.4158 w_k1_IBM_s016
                                 + 0.869028 w_k1_INTC_s016
                                 + 1.0792 w_k1_KO_s016 + 1.34711 w_k1_MAT_s016
                                 + 1.00513 w_k1_MSFT_s016
                                 + 0.99631 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.12148 w_k1_ADM_s016
                                 + 1.17335 w_k1_AXP_s016
                                 + 1.45443 w_k1_DIS_s016
                                 + 1.31448 w_k1_FDX_s016
                                 + 0.827539 w_k1_IBM_s016
                                 + 1.30909 w_k1_INTC_s016
                                 + 1.08857 w_k1_KO_s016
                                 + 0.753477 w_k1_MAT_s016
                                 + 1.36141 w_k1_MSFT_s016
                                 + 1.09606 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.36625 w_k1_ADM_s016
                                 + 1.44022 w_k1_AXP_s016
                                 + 1.02595 w_k1_DIS_s016
                                 + 1.19532 w_k1_FDX_s016
                                 + 0.809177 w_k1_IBM_s016
                                 + 0.967405 w_k1_INTC_s016
                                 + 0.840316 w_k1_KO_s016
                                 + 0.849916 w_k1_MAT_s016
                                 + 1.1243 w_k1_MSFT_s016 + 1.02636 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.02505 w_k1_ADM_s017
                                 + 1.2939 w_k1_AXP_s017 + 1.12648 w_k1_DIS_s017
                                 + 1.09215 w_k1_FDX_s017
                                 + 1.24092 w_k1_IBM_s017
                                 + 1.09047 w_k1_INTC_s017
                                 + 1.05777 w_k1_KO_s017 + 1.35551 w_k1_MAT_s017
                                 + 1.05283 w_k1_MSFT_s017 + 1.1312 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.850575 w_k1_ADM_s017
                                 + 0.675978 w_k1_AXP_s017
                                 + 0.570584 w_k1_DIS_s017
                                 + 0.740539 w_k1_FDX_s017
                                 + 0.719207 w_k1_IBM_s017
                                 + 0.760718 w_k1_INTC_s017
                                 + 0.680521 w_k1_KO_s017
                                 + 0.750469 w_k1_MAT_s017
                                 + 0.753018 w_k1_MSFT_s017
                                 + 0.764923 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.29422 w_k1_ADM_s017
                                 + 1.30089 w_k1_AXP_s017
                                 + 1.21156 w_k1_DIS_s017
                                 + 1.27387 w_k1_FDX_s017
                                 + 0.884074 w_k1_IBM_s017
                                 + 1.18142 w_k1_INTC_s017
                                 + 1.03092 w_k1_KO_s017 + 1.21284 w_k1_MAT_s017
                                 + 1.13997 w_k1_MSFT_s017
                                 + 1.13986 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.83539 w_k1_ADM_s017
                                 + 0.548882 w_k1_AXP_s017
                                 + 0.532615 w_k1_DIS_s017
                                 + 0.735177 w_k1_FDX_s017
                                 + 0.526278 w_k1_IBM_s017
                                 + 0.751825 w_k1_INTC_s017
                                 + 0.619608 w_k1_KO_s017
                                 + 0.489196 w_k1_MAT_s017
                                 + 0.696137 w_k1_MSFT_s017
                                 + 0.793265 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.909401 w_k1_ADM_s017
                                 + 0.832854 w_k1_AXP_s017
                                 + 1.00052 w_k1_DIS_s017
                                 + 1.01783 w_k1_FDX_s017
                                 + 0.590446 w_k1_IBM_s017
                                 + 1.0068 w_k1_INTC_s017
                                 + 0.821793 w_k1_KO_s017
                                 + 0.613257 w_k1_MAT_s017
                                 + 1.12132 w_k1_MSFT_s017
                                 + 1.02704 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.3006 w_k1_ADM_s017
                                 + 1.36394 w_k1_AXP_s017
                                 + 1.65778 w_k1_DIS_s017
                                 + 1.21738 w_k1_FDX_s017
                                 + 1.57571 w_k1_IBM_s017
                                 + 1.21755 w_k1_INTC_s017
                                 + 1.53192 w_k1_KO_s017 + 1.69758 w_k1_MAT_s017
                                 + 1.03308 w_k1_MSFT_s017
                                 + 1.26932 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.860063 w_k1_ADM_s017
                                 + 0.9374 w_k1_AXP_s017
                                 + 0.819633 w_k1_DIS_s017
                                 + 0.821539 w_k1_FDX_s017
                                 + 1.20926 w_k1_IBM_s017
                                 + 0.869923 w_k1_INTC_s017
                                 + 0.988569 w_k1_KO_s017
                                 + 0.931668 w_k1_MAT_s017
                                 + 0.778499 w_k1_MSFT_s017
                                 + 0.885992 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.08841 w_k1_ADM_s017
                                 + 0.749299 w_k1_AXP_s017
                                 + 0.901007 w_k1_DIS_s017
                                 + 0.926614 w_k1_FDX_s017
                                 + 0.825326 w_k1_IBM_s017
                                 + 0.875668 w_k1_INTC_s017
                                 + 0.874092 w_k1_KO_s017
                                 + 0.840133 w_k1_MAT_s017
                                 + 0.837612 w_k1_MSFT_s017
                                 + 1.01302 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.37873 w_k1_ADM_s100
                                 + 1.38169 w_k1_AXP_s100
                                 + 1.30996 w_k1_DIS_s100
                                 + 1.10431 w_k1_FDX_s100
                                 + 1.48949 w_k1_IBM_s100
                                 + 0.865058 w_k1_INTC_s100
                                 + 1.16424 w_k1_KO_s100
                                 + 0.885141 w_k1_MAT_s100
                                 + 0.931943 w_k1_MSFT_s100
                                 + 1.05991 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.0489 w_k1_ADM_s100
                                 + 0.928068 w_k1_AXP_s100
                                 + 1.15141 w_k1_DIS_s100
                                 + 0.906443 w_k1_FDX_s100
                                 + 1.31463 w_k1_IBM_s100
                                 + 0.875392 w_k1_INTC_s100
                                 + 1.15758 w_k1_KO_s100 + 1.23576 w_k1_MAT_s100
                                 + 0.944798 w_k1_MSFT_s100
                                 + 0.982446 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.724098 w_k1_ADM_s100
                                 + 0.765388 w_k1_AXP_s100
                                 + 0.650808 w_k1_DIS_s100
                                 + 0.784249 w_k1_FDX_s100
                                 + 0.856728 w_k1_IBM_s100
                                 + 0.667309 w_k1_INTC_s100
                                 + 0.68467 w_k1_KO_s100
                                 + 0.924405 w_k1_MAT_s100
                                 + 0.681244 w_k1_MSFT_s100
                                 + 0.805291 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.10171 w_k1_ADM_s100
                                 + 1.00662 w_k1_AXP_s100
                                 + 0.921423 w_k1_DIS_s100
                                 + 1.00183 w_k1_FDX_s100
                                 + 0.858207 w_k1_IBM_s100
                                 + 0.969166 w_k1_INTC_s100
                                 + 0.949554 w_k1_KO_s100
                                 + 0.693267 w_k1_MAT_s100
                                 + 0.868181 w_k1_MSFT_s100
                                 + 0.941678 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.866112 w_k1_ADM_s100
                                 + 0.71104 w_k1_AXP_s100
                                 + 0.851528 w_k1_DIS_s100
                                 + 0.940342 w_k1_FDX_s100
                                 + 0.726172 w_k1_IBM_s100
                                 + 0.794159 w_k1_INTC_s100
                                 + 0.857004 w_k1_KO_s100
                                 + 0.737725 w_k1_MAT_s100
                                 + 0.878635 w_k1_MSFT_s100
                                 + 0.905609 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.14784 w_k1_ADM_s100
                                 + 1.13118 w_k1_AXP_s100
                                 + 1.15152 w_k1_DIS_s100
                                 + 1.22414 w_k1_FDX_s100
                                 + 0.630031 w_k1_IBM_s100
                                 + 0.904556 w_k1_INTC_s100
                                 + 0.750327 w_k1_KO_s100
                                 + 0.658729 w_k1_MAT_s100
                                 + 1.00174 w_k1_MSFT_s100
                                 + 0.936245 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.39007 w_k1_ADM_s100
                                 + 1.40379 w_k1_AXP_s100
                                 + 1.60993 w_k1_DIS_s100
                                 + 1.31122 w_k1_FDX_s100
                                 + 1.08713 w_k1_IBM_s100
                                 + 1.04722 w_k1_INTC_s100
                                 + 1.20784 w_k1_KO_s100
                                 + 0.897653 w_k1_MAT_s100
                                 + 1.08387 w_k1_MSFT_s100
                                 + 1.09834 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.913411 w_k1_ADM_s100
                                 + 0.927487 w_k1_AXP_s100
                                 + 0.858219 w_k1_DIS_s100
                                 + 0.87864 w_k1_FDX_s100
                                 + 1.14932 w_k1_IBM_s100
                                 + 1.11815 w_k1_INTC_s100 + 1.0357 w_k1_KO_s100
                                 + 0.935891 w_k1_MAT_s100
                                 + 0.86227 w_k1_MSFT_s100
                                 + 0.952197 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.13702 w_k1_ADM_s101
                                 + 1.30973 w_k1_AXP_s101
                                 + 1.67969 w_k1_DIS_s101
                                 + 1.38125 w_k1_FDX_s101
                                 + 1.28296 w_k1_IBM_s101
                                 + 1.2221 w_k1_INTC_s101 + 1.2613 w_k1_KO_s101
                                 + 1.15654 w_k1_MAT_s101
                                 + 0.955317 w_k1_MSFT_s101
                                 + 1.16844 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.985202 w_k1_ADM_s101
                                 + 0.984046 w_k1_AXP_s101
                                 + 0.904838 w_k1_DIS_s101
                                 + 0.93169 w_k1_FDX_s101
                                 + 1.05046 w_k1_IBM_s101
                                 + 0.780728 w_k1_INTC_s101
                                 + 0.886666 w_k1_KO_s101
                                 + 1.12456 w_k1_MAT_s101
                                 + 0.773538 w_k1_MSFT_s101
                                 + 0.955752 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.786755 w_k1_ADM_s101
                                 + 0.972312 w_k1_AXP_s101
                                 + 0.987696 w_k1_DIS_s101
                                 + 0.889278 w_k1_FDX_s101
                                 + 1.4562 w_k1_IBM_s101
                                 + 0.952248 w_k1_INTC_s101
                                 + 1.01519 w_k1_KO_s101 + 1.2186 w_k1_MAT_s101
                                 + 0.886561 w_k1_MSFT_s101
                                 + 0.962522 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.23228 w_k1_ADM_s101
                                 + 1.22364 w_k1_AXP_s101
                                 + 1.22658 w_k1_DIS_s101
                                 + 1.27907 w_k1_FDX_s101
                                 + 0.822257 w_k1_IBM_s101
                                 + 1.0895 w_k1_INTC_s101
                                 + 0.991003 w_k1_KO_s101
                                 + 0.866485 w_k1_MAT_s101
                                 + 0.999327 w_k1_MSFT_s101
                                 + 1.16358 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.82253 w_k1_ADM_s101
                                 + 0.765283 w_k1_AXP_s101
                                 + 1.06334 w_k1_DIS_s101
                                 + 0.973031 w_k1_FDX_s101
                                 + 0.821359 w_k1_IBM_s101
                                 + 1.09136 w_k1_INTC_s101
                                 + 0.985294 w_k1_KO_s101 + 0.6202 w_k1_MAT_s101
                                 + 0.861679 w_k1_MSFT_s101
                                 + 0.903698 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.971711 w_k1_ADM_s101
                                 + 1.24969 w_k1_AXP_s101
                                 + 0.887003 w_k1_DIS_s101
                                 + 1.1104 w_k1_FDX_s101
                                 + 0.859552 w_k1_IBM_s101
                                 + 0.973316 w_k1_INTC_s101
                                 + 0.793879 w_k1_KO_s101
                                 + 0.916442 w_k1_MAT_s101
                                 + 0.770987 w_k1_MSFT_s101
                                 + 0.932128 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.968482 w_k1_ADM_s101
                                 + 1.13673 w_k1_AXP_s101
                                 + 1.23397 w_k1_DIS_s101
                                 + 1.00014 w_k1_FDX_s101
                                 + 1.83517 w_k1_IBM_s101
                                 + 0.963797 w_k1_INTC_s101
                                 + 1.17549 w_k1_KO_s101 + 2.0395 w_k1_MAT_s101
                                 + 0.952697 w_k1_MSFT_s101
                                 + 1.04184 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.863675 w_k1_ADM_s101
                                 + 0.665173 w_k1_AXP_s101
                                 + 0.680159 w_k1_DIS_s101
                                 + 0.746105 w_k1_FDX_s101
                                 + 0.830436 w_k1_IBM_s101
                                 + 0.784284 w_k1_INTC_s101
                                 + 0.806903 w_k1_KO_s101
                                 + 0.634231 w_k1_MAT_s101
                                 + 0.654875 w_k1_MSFT_s101
                                 + 0.812836 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.24558 w_k1_ADM_s102
                                 + 1.26328 w_k1_AXP_s102
                                 + 1.04029 w_k1_DIS_s102
                                 + 0.990534 w_k1_FDX_s102
                                 + 1.05718 w_k1_IBM_s102
                                 + 0.976117 w_k1_INTC_s102
                                 + 1.00184 w_k1_KO_s102 + 1.04327 w_k1_MAT_s102
                                 + 1.05164 w_k1_MSFT_s102
                                 + 1.21814 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.44263 w_k1_ADM_s102
                                 + 1.31812 w_k1_AXP_s102 + 1.6283 w_k1_DIS_s102
                                 + 1.39551 w_k1_FDX_s102
                                 + 1.03029 w_k1_IBM_s102
                                 + 1.29795 w_k1_INTC_s102
                                 + 1.19395 w_k1_KO_s102 + 1.01744 w_k1_MAT_s102
                                 + 1.3542 w_k1_MSFT_s102 + 1.34192 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.16735 w_k1_ADM_s102
                                 + 1.97078 w_k1_AXP_s102
                                 + 1.46721 w_k1_DIS_s102
                                 + 1.36744 w_k1_FDX_s102
                                 + 0.926951 w_k1_IBM_s102
                                 + 1.10966 w_k1_INTC_s102
                                 + 0.925724 w_k1_KO_s102
                                 + 1.28302 w_k1_MAT_s102
                                 + 1.05564 w_k1_MSFT_s102
                                 + 1.14476 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.916191 w_k1_ADM_s102
                                 + 0.873941 w_k1_AXP_s102
                                 + 1.03474 w_k1_DIS_s102
                                 + 1.04202 w_k1_FDX_s102
                                 + 0.827363 w_k1_IBM_s102
                                 + 0.839946 w_k1_INTC_s102
                                 + 0.929436 w_k1_KO_s102
                                 + 0.811182 w_k1_MAT_s102
                                 + 0.870649 w_k1_MSFT_s102
                                 + 1.03026 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.829179 w_k1_ADM_s102
                                 + 0.969985 w_k1_AXP_s102
                                 + 0.75299 w_k1_DIS_s102
                                 + 0.933192 w_k1_FDX_s102
                                 + 0.699946 w_k1_IBM_s102
                                 + 0.798277 w_k1_INTC_s102
                                 + 0.688029 w_k1_KO_s102
                                 + 0.696759 w_k1_MAT_s102
                                 + 0.762917 w_k1_MSFT_s102
                                 + 1.00854 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.208 w_k1_ADM_s102
                                 + 1.15428 w_k1_AXP_s102
                                 + 1.61504 w_k1_DIS_s102
                                 + 1.19367 w_k1_FDX_s102
                                 + 1.35835 w_k1_IBM_s102
                                 + 0.989222 w_k1_INTC_s102
                                 + 1.32893 w_k1_KO_s102 + 1.34396 w_k1_MAT_s102
                                 + 1.37093 w_k1_MSFT_s102
                                 + 1.26886 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.02551 w_k1_ADM_s102
                                 + 0.889427 w_k1_AXP_s102
                                 + 0.842628 w_k1_DIS_s102
                                 + 0.739423 w_k1_FDX_s102
                                 + 1.38336 w_k1_IBM_s102
                                 + 0.810465 w_k1_INTC_s102
                                 + 1.01865 w_k1_KO_s102 + 1.71595 w_k1_MAT_s102
                                 + 0.841651 w_k1_MSFT_s102
                                 + 1.00165 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.850034 w_k1_ADM_s102
                                 + 0.870561 w_k1_AXP_s102
                                 + 0.830315 w_k1_DIS_s102
                                 + 0.834424 w_k1_FDX_s102
                                 + 1.08803 w_k1_IBM_s102
                                 + 0.838669 w_k1_INTC_s102
                                 + 0.87273 w_k1_KO_s102
                                 + 0.829663 w_k1_MAT_s102
                                 + 0.762591 w_k1_MSFT_s102
                                 + 1.06376 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.895744 w_k1_ADM_s103
                                 + 1.32065 w_k1_AXP_s103
                                 + 0.807337 w_k1_DIS_s103
                                 + 0.878177 w_k1_FDX_s103
                                 + 1.21089 w_k1_IBM_s103
                                 + 0.898671 w_k1_INTC_s103
                                 + 0.948947 w_k1_KO_s103
                                 + 1.54813 w_k1_MAT_s103
                                 + 0.800573 w_k1_MSFT_s103
                                 + 0.888013 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.678398 w_k1_ADM_s103
                                 + 0.763911 w_k1_AXP_s103
                                 + 0.699986 w_k1_DIS_s103
                                 + 0.706801 w_k1_FDX_s103
                                 + 0.930195 w_k1_IBM_s103
                                 + 0.898518 w_k1_INTC_s103
                                 + 0.807865 w_k1_KO_s103
                                 + 1.08938 w_k1_MAT_s103
                                 + 0.705893 w_k1_MSFT_s103
                                 + 0.802622 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.976661 w_k1_ADM_s103
                                 + 1.2279 w_k1_AXP_s103 + 1.27704 w_k1_DIS_s103
                                 + 1.20371 w_k1_FDX_s103
                                 + 1.01033 w_k1_IBM_s103
                                 + 1.04612 w_k1_INTC_s103
                                 + 1.13385 w_k1_KO_s103 + 1.35242 w_k1_MAT_s103
                                 + 1.09214 w_k1_MSFT_s103
                                 + 1.00658 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.20979 w_k1_ADM_s103
                                 + 1.33657 w_k1_AXP_s103
                                 + 1.63368 w_k1_DIS_s103
                                 + 1.47559 w_k1_FDX_s103
                                 + 0.820097 w_k1_IBM_s103
                                 + 1.09676 w_k1_INTC_s103
                                 + 1.08957 w_k1_KO_s103
                                 + 0.849866 w_k1_MAT_s103
                                 + 1.19381 w_k1_MSFT_s103
                                 + 1.07651 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.833042 w_k1_ADM_s103
                                 + 0.820268 w_k1_AXP_s103
                                 + 0.884666 w_k1_DIS_s103
                                 + 0.860202 w_k1_FDX_s103
                                 + 0.945216 w_k1_IBM_s103
                                 + 0.900517 w_k1_INTC_s103
                                 + 0.973822 w_k1_KO_s103
                                 + 0.863858 w_k1_MAT_s103
                                 + 0.905558 w_k1_MSFT_s103
                                 + 0.873321 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.896391 w_k1_ADM_s103
                                 + 1.11435 w_k1_AXP_s103
                                 + 0.916857 w_k1_DIS_s103
                                 + 1.04224 w_k1_FDX_s103
                                 + 0.748601 w_k1_IBM_s103
                                 + 0.942302 w_k1_INTC_s103
                                 + 0.918635 w_k1_KO_s103
                                 + 1.23306 w_k1_MAT_s103
                                 + 0.935088 w_k1_MSFT_s103
                                 + 0.924303 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.857605 w_k1_ADM_s103
                                 + 0.932526 w_k1_AXP_s103
                                 + 1.13968 w_k1_DIS_s103
                                 + 0.99503 w_k1_FDX_s103
                                 + 1.35743 w_k1_IBM_s103
                                 + 0.92225 w_k1_INTC_s103
                                 + 1.28515 w_k1_KO_s103 + 1.08587 w_k1_MAT_s103
                                 + 0.976896 w_k1_MSFT_s103
                                 + 1.04958 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.929533 w_k1_ADM_s103
                                 + 1.11521 w_k1_AXP_s103
                                 + 0.924786 w_k1_DIS_s103
                                 + 0.885789 w_k1_FDX_s103
                                 + 1.67534 w_k1_IBM_s103
                                 + 0.935418 w_k1_INTC_s103
                                 + 1.22951 w_k1_KO_s103 + 2.04543 w_k1_MAT_s103
                                 + 0.914354 w_k1_MSFT_s103
                                 + 1.00149 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.737654 w_k1_ADM_s104
                                 + 0.9115 w_k1_AXP_s104 + 0.81722 w_k1_DIS_s104
                                 + 0.815359 w_k1_FDX_s104
                                 + 1.24188 w_k1_IBM_s104
                                 + 1.16291 w_k1_INTC_s104
                                 + 1.01326 w_k1_KO_s104
                                 + 0.786219 w_k1_MAT_s104
                                 + 0.992264 w_k1_MSFT_s104
                                 + 0.958394 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.20959 w_k1_ADM_s104
                                 + 1.44471 w_k1_AXP_s104
                                 + 1.54244 w_k1_DIS_s104
                                 + 1.36789 w_k1_FDX_s104
                                 + 1.09706 w_k1_IBM_s104
                                 + 1.38962 w_k1_INTC_s104
                                 + 1.39603 w_k1_KO_s104 + 1.12911 w_k1_MAT_s104
                                 + 1.48666 w_k1_MSFT_s104
                                 + 1.31712 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.07859 w_k1_ADM_s104
                                 + 1.11879 w_k1_AXP_s104
                                 + 1.19385 w_k1_DIS_s104
                                 + 1.27652 w_k1_FDX_s104
                                 + 0.728939 w_k1_IBM_s104
                                 + 1.27156 w_k1_INTC_s104
                                 + 0.834826 w_k1_KO_s104
                                 + 0.527826 w_k1_MAT_s104
                                 + 1.282 w_k1_MSFT_s104 + 1.07591 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.85407 w_k1_ADM_s104
                                 + 0.698131 w_k1_AXP_s104
                                 + 0.716957 w_k1_DIS_s104
                                 + 0.871223 w_k1_FDX_s104
                                 + 0.681557 w_k1_IBM_s104
                                 + 0.889326 w_k1_INTC_s104
                                 + 0.835097 w_k1_KO_s104
                                 + 0.741707 w_k1_MAT_s104
                                 + 0.981025 w_k1_MSFT_s104
                                 + 0.830293 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.999084 w_k1_ADM_s104
                                 + 1.10508 w_k1_AXP_s104
                                 + 1.06735 w_k1_DIS_s104
                                 + 1.05851 w_k1_FDX_s104
                                 + 0.965016 w_k1_IBM_s104
                                 + 1.22005 w_k1_INTC_s104
                                 + 1.08874 w_k1_KO_s104
                                 + 0.872412 w_k1_MAT_s104
                                 + 1.24103 w_k1_MSFT_s104
                                 + 1.06117 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.953108 w_k1_ADM_s104
                                 + 1.02149 w_k1_AXP_s104
                                 + 0.918783 w_k1_DIS_s104
                                 + 0.916264 w_k1_FDX_s104
                                 + 1.7125 w_k1_IBM_s104
                                 + 1.02913 w_k1_INTC_s104
                                 + 1.32102 w_k1_KO_s104 + 1.62564 w_k1_MAT_s104
                                 + 1.01299 w_k1_MSFT_s104
                                 + 1.09528 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.671563 w_k1_ADM_s104
                                 + 0.621933 w_k1_AXP_s104
                                 + 0.638698 w_k1_DIS_s104
                                 + 0.698317 w_k1_FDX_s104
                                 + 0.893687 w_k1_IBM_s104
                                 + 0.774336 w_k1_INTC_s104
                                 + 0.722524 w_k1_KO_s104
                                 + 0.512993 w_k1_MAT_s104
                                 + 0.764035 w_k1_MSFT_s104
                                 + 0.826613 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.930356 w_k1_ADM_s104
                                 + 0.847991 w_k1_AXP_s104
                                 + 0.849712 w_k1_DIS_s104
                                 + 0.862052 w_k1_FDX_s104
                                 + 1.0299 w_k1_IBM_s104
                                 + 0.95138 w_k1_INTC_s104
                                 + 0.887254 w_k1_KO_s104
                                 + 0.667978 w_k1_MAT_s104
                                 + 0.878807 w_k1_MSFT_s104
                                 + 0.932714 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.93311 w_k1_ADM_s105
                                 + 0.655035 w_k1_AXP_s105
                                 + 0.617211 w_k1_DIS_s105
                                 + 0.857845 w_k1_FDX_s105
                                 + 0.579177 w_k1_IBM_s105
                                 + 0.733408 w_k1_INTC_s105
                                 + 0.684122 w_k1_KO_s105
                                 + 0.558489 w_k1_MAT_s105
                                 + 0.848493 w_k1_MSFT_s105
                                 + 0.802594 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.92729 w_k1_ADM_s105
                                 + 0.996526 w_k1_AXP_s105
                                 + 0.804755 w_k1_DIS_s105
                                 + 0.817083 w_k1_FDX_s105
                                 + 1.37637 w_k1_IBM_s105
                                 + 0.951488 w_k1_INTC_s105
                                 + 1.04361 w_k1_KO_s105 + 1.56384 w_k1_MAT_s105
                                 + 0.860987 w_k1_MSFT_s105
                                 + 0.911776 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.30312 w_k1_ADM_s105
                                 + 1.20385 w_k1_AXP_s105
                                 + 1.24244 w_k1_DIS_s105
                                 + 1.23967 w_k1_FDX_s105
                                 + 0.815031 w_k1_IBM_s105
                                 + 1.21331 w_k1_INTC_s105
                                 + 1.04014 w_k1_KO_s105
                                 + 0.954773 w_k1_MAT_s105
                                 + 1.24976 w_k1_MSFT_s105
                                 + 1.10258 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.04967 w_k1_ADM_s105
                                 + 1.05966 w_k1_AXP_s105
                                 + 1.10369 w_k1_DIS_s105
                                 + 1.03375 w_k1_FDX_s105
                                 + 1.06734 w_k1_IBM_s105
                                 + 1.13486 w_k1_INTC_s105
                                 + 1.02586 w_k1_KO_s105 + 1.01347 w_k1_MAT_s105
                                 + 1.0854 w_k1_MSFT_s105 + 1.00785 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.0208 w_k1_ADM_s105
                                 + 1.12203 w_k1_AXP_s105
                                 + 0.935909 w_k1_DIS_s105
                                 + 0.998074 w_k1_FDX_s105
                                 + 0.804281 w_k1_IBM_s105
                                 + 1.17104 w_k1_INTC_s105
                                 + 0.980167 w_k1_KO_s105
                                 + 0.781215 w_k1_MAT_s105
                                 + 0.95199 w_k1_MSFT_s105
                                 + 0.905496 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.749676 w_k1_ADM_s105
                                 + 0.650676 w_k1_AXP_s105
                                 + 0.765407 w_k1_DIS_s105
                                 + 0.726466 w_k1_FDX_s105
                                 + 0.945707 w_k1_IBM_s105
                                 + 0.874738 w_k1_INTC_s105
                                 + 0.937877 w_k1_KO_s105
                                 + 0.896128 w_k1_MAT_s105
                                 + 0.887632 w_k1_MSFT_s105
                                 + 0.849907 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.951203 w_k1_ADM_s105
                                 + 0.765093 w_k1_AXP_s105
                                 + 0.885727 w_k1_DIS_s105
                                 + 0.876374 w_k1_FDX_s105
                                 + 0.758385 w_k1_IBM_s105
                                 + 0.936686 w_k1_INTC_s105
                                 + 0.888805 w_k1_KO_s105
                                 + 0.856403 w_k1_MAT_s105
                                 + 0.916133 w_k1_MSFT_s105
                                 + 0.888855 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.25494 w_k1_ADM_s105
                                 + 1.58285 w_k1_AXP_s105
                                 + 1.56559 w_k1_DIS_s105
                                 + 1.14744 w_k1_FDX_s105
                                 + 1.58586 w_k1_IBM_s105
                                 + 1.10408 w_k1_INTC_s105
                                 + 1.59842 w_k1_KO_s105 + 1.85259 w_k1_MAT_s105
                                 + 1.26446 w_k1_MSFT_s105
                                 + 1.15368 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.650979 w_k1_ADM_s106
                                 + 0.659775 w_k1_AXP_s106
                                 + 0.731057 w_k1_DIS_s106
                                 + 0.774904 w_k1_FDX_s106
                                 + 0.649833 w_k1_IBM_s106
                                 + 0.86654 w_k1_INTC_s106
                                 + 0.704707 w_k1_KO_s106
                                 + 0.649829 w_k1_MAT_s106
                                 + 0.884272 w_k1_MSFT_s106
                                 + 0.902349 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.734318 w_k1_ADM_s106
                                 + 0.940339 w_k1_AXP_s106
                                 + 0.779395 w_k1_DIS_s106
                                 + 0.917828 w_k1_FDX_s106
                                 + 0.751934 w_k1_IBM_s106
                                 + 1.0036 w_k1_INTC_s106
                                 + 0.594941 w_k1_KO_s106
                                 + 0.742217 w_k1_MAT_s106
                                 + 0.843923 w_k1_MSFT_s106
                                 + 0.822687 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.865964 w_k1_ADM_s106
                                 + 0.972686 w_k1_AXP_s106
                                 + 1.01017 w_k1_DIS_s106
                                 + 0.881609 w_k1_FDX_s106
                                 + 1.39696 w_k1_IBM_s106
                                 + 0.749538 w_k1_INTC_s106
                                 + 0.95142 w_k1_KO_s106 + 1.06062 w_k1_MAT_s106
                                 + 0.850647 w_k1_MSFT_s106
                                 + 0.90505 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.16608 w_k1_ADM_s106
                                 + 1.40102 w_k1_AXP_s106
                                 + 1.39867 w_k1_DIS_s106
                                 + 1.04525 w_k1_FDX_s106
                                 + 1.46537 w_k1_IBM_s106
                                 + 0.907527 w_k1_INTC_s106
                                 + 1.03323 w_k1_KO_s106 + 1.42007 w_k1_MAT_s106
                                 + 0.994836 w_k1_MSFT_s106
                                 + 1.25283 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.08806 w_k1_ADM_s106
                                 + 1.0122 w_k1_AXP_s106
                                 + 0.836463 w_k1_DIS_s106
                                 + 1.13405 w_k1_FDX_s106
                                 + 0.778037 w_k1_IBM_s106
                                 + 0.737723 w_k1_INTC_s106
                                 + 0.66348 w_k1_KO_s106
                                 + 0.761663 w_k1_MAT_s106
                                 + 0.883573 w_k1_MSFT_s106
                                 + 1.03833 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.01787 w_k1_ADM_s106
                                 + 1.21812 w_k1_AXP_s106
                                 + 1.52253 w_k1_DIS_s106 + 1.3307 w_k1_FDX_s106
                                 + 1.18416 w_k1_IBM_s106
                                 + 1.07004 w_k1_INTC_s106
                                 + 0.993714 w_k1_KO_s106
                                 + 1.35383 w_k1_MAT_s106
                                 + 1.27193 w_k1_MSFT_s106 + 1.1901 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.947678 w_k1_ADM_s106
                                 + 0.865866 w_k1_AXP_s106
                                 + 1.05771 w_k1_DIS_s106
                                 + 0.996886 w_k1_FDX_s106
                                 + 1.01283 w_k1_IBM_s106
                                 + 0.989344 w_k1_INTC_s106
                                 + 0.938584 w_k1_KO_s106
                                 + 1.29674 w_k1_MAT_s106
                                 + 1.18388 w_k1_MSFT_s106
                                 + 1.09542 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.05651 w_k1_ADM_s106
                                 + 1.34597 w_k1_AXP_s106
                                 + 1.24438 w_k1_DIS_s106
                                 + 1.18694 w_k1_FDX_s106
                                 + 0.744229 w_k1_IBM_s106
                                 + 1.07951 w_k1_INTC_s106
                                 + 0.822609 w_k1_KO_s106
                                 + 0.925818 w_k1_MAT_s106
                                 + 1.04652 w_k1_MSFT_s106 + 1.0129 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.08048 w_k1_ADM_s107
                                 + 1.173 w_k1_AXP_s107 + 1.20153 w_k1_DIS_s107
                                 + 1.01296 w_k1_FDX_s107
                                 + 1.12663 w_k1_IBM_s107
                                 + 1.07594 w_k1_INTC_s107
                                 + 1.24331 w_k1_KO_s107 + 1.36859 w_k1_MAT_s107
                                 + 1.00282 w_k1_MSFT_s107
                                 + 1.21456 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.01748 w_k1_ADM_s107
                                 + 1.04414 w_k1_AXP_s107 + 1.3708 w_k1_DIS_s107
                                 + 0.989885 w_k1_FDX_s107
                                 + 1.69907 w_k1_IBM_s107
                                 + 1.07662 w_k1_INTC_s107
                                 + 1.24571 w_k1_KO_s107 + 1.76719 w_k1_MAT_s107
                                 + 1.13162 w_k1_MSFT_s107
                                 + 1.07307 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.901212 w_k1_ADM_s107
                                 + 0.769538 w_k1_AXP_s107
                                 + 0.55629 w_k1_DIS_s107
                                 + 0.69571 w_k1_FDX_s107
                                 + 0.655443 w_k1_IBM_s107
                                 + 0.695044 w_k1_INTC_s107
                                 + 0.550138 w_k1_KO_s107
                                 + 0.688477 w_k1_MAT_s107
                                 + 0.707664 w_k1_MSFT_s107
                                 + 0.767546 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.95451 w_k1_ADM_s107
                                 + 1.60692 w_k1_AXP_s107
                                 + 1.46153 w_k1_DIS_s107
                                 + 1.30601 w_k1_FDX_s107
                                 + 1.29414 w_k1_IBM_s107
                                 + 1.55812 w_k1_INTC_s107
                                 + 1.24508 w_k1_KO_s107 + 1.45358 w_k1_MAT_s107
                                 + 1.17076 w_k1_MSFT_s107 + 1.1156 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.18016 w_k1_ADM_s107
                                 + 1.53269 w_k1_AXP_s107
                                 + 1.06205 w_k1_DIS_s107
                                 + 1.24457 w_k1_FDX_s107
                                 + 0.755759 w_k1_IBM_s107
                                 + 1.09368 w_k1_INTC_s107
                                 + 0.957816 w_k1_KO_s107 + 1.2283 w_k1_MAT_s107
                                 + 1.03297 w_k1_MSFT_s107
                                 + 0.957139 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.805904 w_k1_ADM_s107
                                 + 0.85786 w_k1_AXP_s107
                                 + 0.828614 w_k1_DIS_s107
                                 + 0.849749 w_k1_FDX_s107
                                 + 1.18014 w_k1_IBM_s107
                                 + 0.89586 w_k1_INTC_s107
                                 + 1.00131 w_k1_KO_s107
                                 + 0.946969 w_k1_MAT_s107
                                 + 0.947174 w_k1_MSFT_s107
                                 + 0.96774 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.825784 w_k1_ADM_s107
                                 + 0.807413 w_k1_AXP_s107
                                 + 0.894364 w_k1_DIS_s107
                                 + 0.833175 w_k1_FDX_s107
                                 + 0.89485 w_k1_IBM_s107
                                 + 0.980304 w_k1_INTC_s107
                                 + 0.948288 w_k1_KO_s107
                                 + 0.682226 w_k1_MAT_s107
                                 + 0.912196 w_k1_MSFT_s107
                                 + 0.849837 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.944164 w_k1_ADM_s107
                                 + 1.05819 w_k1_AXP_s107
                                 + 0.800674 w_k1_DIS_s107
                                 + 0.927885 w_k1_FDX_s107
                                 + 0.776523 w_k1_IBM_s107
                                 + 0.859603 w_k1_INTC_s107
                                 + 0.868057 w_k1_KO_s107 + 1.0695 w_k1_MAT_s107
                                 + 0.906517 w_k1_MSFT_s107
                                 + 0.858517 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.854475 w_k1_ADM_s110
                                 + 0.810961 w_k1_AXP_s110
                                 + 0.847182 w_k1_DIS_s110
                                 + 0.926184 w_k1_FDX_s110
                                 + 0.810913 w_k1_IBM_s110
                                 + 0.997737 w_k1_INTC_s110
                                 + 1.00216 w_k1_KO_s110
                                 + 0.925568 w_k1_MAT_s110
                                 + 0.933061 w_k1_MSFT_s110
                                 + 0.91911 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.713209 w_k1_ADM_s110
                                 + 0.643266 w_k1_AXP_s110
                                 + 0.576027 w_k1_DIS_s110
                                 + 0.826236 w_k1_FDX_s110
                                 + 0.636995 w_k1_IBM_s110
                                 + 0.863226 w_k1_INTC_s110
                                 + 0.681069 w_k1_KO_s110
                                 + 0.621349 w_k1_MAT_s110
                                 + 0.69466 w_k1_MSFT_s110
                                 + 0.733948 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.34271 w_k1_ADM_s110
                                 + 1.06283 w_k1_AXP_s110
                                 + 1.25239 w_k1_DIS_s110
                                 + 1.14861 w_k1_FDX_s110
                                 + 1.38142 w_k1_IBM_s110
                                 + 0.88946 w_k1_INTC_s110
                                 + 1.19459 w_k1_KO_s110 + 1.56612 w_k1_MAT_s110
                                 + 1.03294 w_k1_MSFT_s110
                                 + 1.06192 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.939732 w_k1_ADM_s110
                                 + 1.19695 w_k1_AXP_s110
                                 + 1.15014 w_k1_DIS_s110
                                 + 1.05067 w_k1_FDX_s110
                                 + 0.962087 w_k1_IBM_s110
                                 + 1.08885 w_k1_INTC_s110
                                 + 1.24688 w_k1_KO_s110 + 1.29373 w_k1_MAT_s110
                                 + 0.98654 w_k1_MSFT_s110
                                 + 1.07901 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.5584 w_k1_ADM_s110
                                 + 0.632859 w_k1_AXP_s110
                                 + 0.48225 w_k1_DIS_s110
                                 + 0.638385 w_k1_FDX_s110
                                 + 0.710033 w_k1_IBM_s110
                                 + 0.839835 w_k1_INTC_s110
                                 + 0.680239 w_k1_KO_s110
                                 + 0.698649 w_k1_MAT_s110
                                 + 0.770789 w_k1_MSFT_s110
                                 + 0.774545 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.981373 w_k1_ADM_s110
                                 + 1.52774 w_k1_AXP_s110
                                 + 1.24734 w_k1_DIS_s110
                                 + 1.22335 w_k1_FDX_s110
                                 + 2.02081 w_k1_IBM_s110
                                 + 1.35263 w_k1_INTC_s110
                                 + 1.53412 w_k1_KO_s110 + 2.33708 w_k1_MAT_s110
                                 + 1.14447 w_k1_MSFT_s110
                                 + 1.15826 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.90072 w_k1_ADM_s110
                                 + 0.910017 w_k1_AXP_s110
                                 + 1.03631 w_k1_DIS_s110
                                 + 1.10792 w_k1_FDX_s110
                                 + 0.681148 w_k1_IBM_s110
                                 + 1.28896 w_k1_INTC_s110
                                 + 0.897648 w_k1_KO_s110
                                 + 0.545356 w_k1_MAT_s110
                                 + 1.04615 w_k1_MSFT_s110
                                 + 0.964808 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.771977 w_k1_ADM_s110
                                 + 0.848962 w_k1_AXP_s110
                                 + 0.913843 w_k1_DIS_s110
                                 + 0.88685 w_k1_FDX_s110
                                 + 1.06307 w_k1_IBM_s110
                                 + 1.0293 w_k1_INTC_s110 + 1.04997 w_k1_KO_s110
                                 + 1.05151 w_k1_MAT_s110
                                 + 0.968028 w_k1_MSFT_s110
                                 + 0.987127 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.26138 w_k1_ADM_s111
                                 + 1.35129 w_k1_AXP_s111
                                 + 1.77607 w_k1_DIS_s111
                                 + 1.29973 w_k1_FDX_s111
                                 + 1.50472 w_k1_IBM_s111
                                 + 1.45829 w_k1_INTC_s111 + 1.3893 w_k1_KO_s111
                                 + 2.08518 w_k1_MAT_s111
                                 + 1.40589 w_k1_MSFT_s111 + 1.2127 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.01904 w_k1_ADM_s111
                                 + 1.17525 w_k1_AXP_s111
                                 + 1.40724 w_k1_DIS_s111
                                 + 1.20544 w_k1_FDX_s111
                                 + 1.02445 w_k1_IBM_s111
                                 + 1.30344 w_k1_INTC_s111
                                 + 1.08366 w_k1_KO_s111
                                 + 0.871065 w_k1_MAT_s111
                                 + 1.17128 w_k1_MSFT_s111
                                 + 1.12524 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.864576 w_k1_ADM_s111
                                 + 0.72254 w_k1_AXP_s111
                                 + 0.833798 w_k1_DIS_s111
                                 + 0.781314 w_k1_FDX_s111
                                 + 1.06472 w_k1_IBM_s111
                                 + 0.85681 w_k1_INTC_s111
                                 + 0.991982 w_k1_KO_s111
                                 + 1.37243 w_k1_MAT_s111
                                 + 0.866919 w_k1_MSFT_s111
                                 + 0.946148 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.15604 w_k1_ADM_s111
                                 + 1.02872 w_k1_AXP_s111
                                 + 0.918406 w_k1_DIS_s111
                                 + 1.07069 w_k1_FDX_s111
                                 + 0.655581 w_k1_IBM_s111
                                 + 1.06989 w_k1_INTC_s111
                                 + 0.795852 w_k1_KO_s111
                                 + 0.735561 w_k1_MAT_s111
                                 + 1.02088 w_k1_MSFT_s111
                                 + 1.12402 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.03902 w_k1_ADM_s111
                                 + 1.13069 w_k1_AXP_s111
                                 + 1.36853 w_k1_DIS_s111
                                 + 1.05369 w_k1_FDX_s111
                                 + 1.80756 w_k1_IBM_s111
                                 + 1.2926 w_k1_INTC_s111 + 1.44829 w_k1_KO_s111
                                 + 2.18013 w_k1_MAT_s111
                                 + 1.18945 w_k1_MSFT_s111
                                 + 1.17865 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.917485 w_k1_ADM_s111
                                 + 0.901842 w_k1_AXP_s111
                                 + 1.00797 w_k1_DIS_s111
                                 + 0.958571 w_k1_FDX_s111
                                 + 1.04333 w_k1_IBM_s111
                                 + 1.17927 w_k1_INTC_s111
                                 + 0.786899 w_k1_KO_s111
                                 + 0.824173 w_k1_MAT_s111
                                 + 0.976469 w_k1_MSFT_s111
                                 + 1.00937 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.82104 w_k1_ADM_s111
                                 + 0.63794 w_k1_AXP_s111 + 0.7584 w_k1_DIS_s111
                                 + 0.855324 w_k1_FDX_s111
                                 + 0.72898 w_k1_IBM_s111
                                 + 0.93325 w_k1_INTC_s111
                                 + 0.697554 w_k1_KO_s111
                                 + 0.603066 w_k1_MAT_s111
                                 + 0.843986 w_k1_MSFT_s111
                                 + 0.922806 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.09121 w_k1_ADM_s111
                                 + 0.951774 w_k1_AXP_s111
                                 + 1.01449 w_k1_DIS_s111
                                 + 0.952498 w_k1_FDX_s111
                                 + 1.47849 w_k1_IBM_s111
                                 + 1.23018 w_k1_INTC_s111
                                 + 1.24463 w_k1_KO_s111 + 1.58726 w_k1_MAT_s111
                                 + 1.06425 w_k1_MSFT_s111
                                 + 1.06981 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.845255 w_k1_ADM_s112
                                 + 1.10128 w_k1_AXP_s112
                                 + 1.06311 w_k1_DIS_s112
                                 + 0.897081 w_k1_FDX_s112
                                 + 1.34496 w_k1_IBM_s112
                                 + 0.986609 w_k1_INTC_s112
                                 + 1.21798 w_k1_KO_s112 + 1.48054 w_k1_MAT_s112
                                 + 0.92599 w_k1_MSFT_s112
                                 + 1.06943 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.14044 w_k1_ADM_s112
                                 + 1.04002 w_k1_AXP_s112
                                 + 1.64048 w_k1_DIS_s112 + 1.3198 w_k1_FDX_s112
                                 + 1.13469 w_k1_IBM_s112
                                 + 1.48577 w_k1_INTC_s112
                                 + 1.23889 w_k1_KO_s112 + 1.0124 w_k1_MAT_s112
                                 + 1.34923 w_k1_MSFT_s112
                                 + 1.18942 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.30745 w_k1_ADM_s112
                                 + 1.64336 w_k1_AXP_s112
                                 + 1.65406 w_k1_DIS_s112
                                 + 1.21475 w_k1_FDX_s112
                                 + 1.63993 w_k1_IBM_s112
                                 + 1.45816 w_k1_INTC_s112
                                 + 1.48927 w_k1_KO_s112 + 1.42669 w_k1_MAT_s112
                                 + 1.28617 w_k1_MSFT_s112
                                 + 1.28637 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.945599 w_k1_ADM_s112
                                 + 0.809942 w_k1_AXP_s112
                                 + 1.11047 w_k1_DIS_s112
                                 + 1.10555 w_k1_FDX_s112
                                 + 0.583401 w_k1_IBM_s112
                                 + 1.20142 w_k1_INTC_s112
                                 + 0.877653 w_k1_KO_s112
                                 + 0.663376 w_k1_MAT_s112
                                 + 1.18306 w_k1_MSFT_s112
                                 + 1.00866 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.28302 w_k1_ADM_s112
                                 + 1.29859 w_k1_AXP_s112
                                 + 1.16259 w_k1_DIS_s112
                                 + 1.10401 w_k1_FDX_s112
                                 + 0.85648 w_k1_IBM_s112
                                 + 1.06197 w_k1_INTC_s112
                                 + 0.949167 w_k1_KO_s112
                                 + 0.782426 w_k1_MAT_s112
                                 + 0.984988 w_k1_MSFT_s112
                                 + 1.13321 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.22084 w_k1_ADM_s112
                                 + 1.65101 w_k1_AXP_s112
                                 + 1.16577 w_k1_DIS_s112 + 1.1404 w_k1_FDX_s112
                                 + 1.04207 w_k1_IBM_s112
                                 + 1.36716 w_k1_INTC_s112 + 1.2042 w_k1_KO_s112
                                 + 1.46069 w_k1_MAT_s112
                                 + 1.11817 w_k1_MSFT_s112
                                 + 1.23966 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.02173 w_k1_ADM_s112
                                 + 0.951457 w_k1_AXP_s112
                                 + 0.745844 w_k1_DIS_s112
                                 + 0.898608 w_k1_FDX_s112
                                 + 0.698062 w_k1_IBM_s112
                                 + 1.01215 w_k1_INTC_s112
                                 + 0.747093 w_k1_KO_s112
                                 + 0.835722 w_k1_MAT_s112
                                 + 0.925182 w_k1_MSFT_s112
                                 + 0.978858 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.777108 w_k1_ADM_s112
                                 + 0.746314 w_k1_AXP_s112
                                 + 0.721587 w_k1_DIS_s112
                                 + 0.764734 w_k1_FDX_s112
                                 + 0.788181 w_k1_IBM_s112
                                 + 0.909677 w_k1_INTC_s112
                                 + 0.821177 w_k1_KO_s112
                                 + 0.796291 w_k1_MAT_s112
                                 + 0.903789 w_k1_MSFT_s112
                                 + 0.947167 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.25354 w_k1_ADM_s113
                                 + 1.00417 w_k1_AXP_s113
                                 + 0.923938 w_k1_DIS_s113
                                 + 1.01933 w_k1_FDX_s113
                                 + 0.952961 w_k1_IBM_s113
                                 + 1.01891 w_k1_INTC_s113
                                 + 1.14161 w_k1_KO_s113 + 1.173 w_k1_MAT_s113
                                 + 1.0724 w_k1_MSFT_s113 + 1.12136 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.04358 w_k1_ADM_s113
                                 + 0.920717 w_k1_AXP_s113
                                 + 0.722974 w_k1_DIS_s113
                                 + 0.858881 w_k1_FDX_s113
                                 + 1.05465 w_k1_IBM_s113
                                 + 0.697645 w_k1_INTC_s113
                                 + 0.889175 w_k1_KO_s113
                                 + 1.01977 w_k1_MAT_s113
                                 + 0.666603 w_k1_MSFT_s113
                                 + 0.950421 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.94975 w_k1_ADM_s113
                                 + 0.756275 w_k1_AXP_s113
                                 + 0.61614 w_k1_DIS_s113
                                 + 0.844209 w_k1_FDX_s113
                                 + 0.615193 w_k1_IBM_s113
                                 + 0.788684 w_k1_INTC_s113
                                 + 0.726755 w_k1_KO_s113
                                 + 0.720684 w_k1_MAT_s113
                                 + 0.821153 w_k1_MSFT_s113
                                 + 0.922248 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.818827 w_k1_ADM_s113
                                 + 0.738252 w_k1_AXP_s113
                                 + 0.590673 w_k1_DIS_s113
                                 + 0.746533 w_k1_FDX_s113
                                 + 0.81543 w_k1_IBM_s113
                                 + 0.681725 w_k1_INTC_s113
                                 + 0.825017 w_k1_KO_s113
                                 + 0.838432 w_k1_MAT_s113
                                 + 0.723605 w_k1_MSFT_s113
                                 + 0.908277 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.085 w_k1_ADM_s113
                                 + 0.912798 w_k1_AXP_s113
                                 + 0.879387 w_k1_DIS_s113
                                 + 1.02171 w_k1_FDX_s113
                                 + 0.736416 w_k1_IBM_s113
                                 + 0.809451 w_k1_INTC_s113
                                 + 0.913116 w_k1_KO_s113
                                 + 0.75422 w_k1_MAT_s113
                                 + 1.02692 w_k1_MSFT_s113
                                 + 0.981811 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.819443 w_k1_ADM_s113
                                 + 0.817853 w_k1_AXP_s113
                                 + 0.922813 w_k1_DIS_s113
                                 + 0.799036 w_k1_FDX_s113
                                 + 1.07011 w_k1_IBM_s113
                                 + 0.821922 w_k1_INTC_s113
                                 + 1.04672 w_k1_KO_s113
                                 + 0.673396 w_k1_MAT_s113
                                 + 0.850417 w_k1_MSFT_s113
                                 + 0.9479 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.832811 w_k1_ADM_s113
                                 + 1.18668 w_k1_AXP_s113
                                 + 1.10464 w_k1_DIS_s113
                                 + 1.00156 w_k1_FDX_s113
                                 + 1.57201 w_k1_IBM_s113
                                 + 0.960038 w_k1_INTC_s113
                                 + 1.1441 w_k1_KO_s113 + 1.51486 w_k1_MAT_s113
                                 + 1.03404 w_k1_MSFT_s113
                                 + 1.06747 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.38873 w_k1_ADM_s113
                                 + 1.40423 w_k1_AXP_s113
                                 + 1.35209 w_k1_DIS_s113
                                 + 1.28743 w_k1_FDX_s113
                                 + 1.18846 w_k1_IBM_s113
                                 + 1.12888 w_k1_INTC_s113
                                 + 1.25689 w_k1_KO_s113 + 1.08515 w_k1_MAT_s113
                                 + 1.30412 w_k1_MSFT_s113
                                 + 1.17081 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.11642 w_k1_ADM_s114
                                 + 1.1877 w_k1_AXP_s114
                                 + 0.943715 w_k1_DIS_s114
                                 + 1.06166 w_k1_FDX_s114
                                 + 0.902183 w_k1_IBM_s114
                                 + 0.914269 w_k1_INTC_s114
                                 + 1.14988 w_k1_KO_s114
                                 + 0.885934 w_k1_MAT_s114
                                 + 1.00848 w_k1_MSFT_s114
                                 + 1.07208 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.988296 w_k1_ADM_s114
                                 + 0.894129 w_k1_AXP_s114
                                 + 1.03309 w_k1_DIS_s114
                                 + 0.946707 w_k1_FDX_s114
                                 + 1.12135 w_k1_IBM_s114
                                 + 1.02211 w_k1_INTC_s114
                                 + 1.30649 w_k1_KO_s114
                                 + 0.937174 w_k1_MAT_s114
                                 + 0.921938 w_k1_MSFT_s114
                                 + 1.04241 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.08921 w_k1_ADM_s114
                                 + 1.256 w_k1_AXP_s114 + 0.939502 w_k1_DIS_s114
                                 + 0.965238 w_k1_FDX_s114
                                 + 1.49027 w_k1_IBM_s114
                                 + 0.942281 w_k1_INTC_s114 + 1.069 w_k1_KO_s114
                                 + 1.58986 w_k1_MAT_s114
                                 + 0.805626 w_k1_MSFT_s114
                                 + 1.14575 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.841185 w_k1_ADM_s114
                                 + 1.26305 w_k1_AXP_s114
                                 + 1.11357 w_k1_DIS_s114
                                 + 1.27537 w_k1_FDX_s114
                                 + 0.785391 w_k1_IBM_s114
                                 + 1.23105 w_k1_INTC_s114
                                 + 1.01364 w_k1_KO_s114
                                 + 0.895636 w_k1_MAT_s114
                                 + 1.17024 w_k1_MSFT_s114
                                 + 1.14506 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.981681 w_k1_ADM_s114
                                 + 0.775513 w_k1_AXP_s114
                                 + 1.04336 w_k1_DIS_s114
                                 + 0.965476 w_k1_FDX_s114
                                 + 0.724037 w_k1_IBM_s114
                                 + 0.90668 w_k1_INTC_s114
                                 + 0.941837 w_k1_KO_s114
                                 + 0.760405 w_k1_MAT_s114
                                 + 0.887877 w_k1_MSFT_s114
                                 + 0.953904 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.07913 w_k1_ADM_s114
                                 + 1.39644 w_k1_AXP_s114
                                 + 1.47467 w_k1_DIS_s114
                                 + 1.23713 w_k1_FDX_s114
                                 + 1.39113 w_k1_IBM_s114
                                 + 1.33158 w_k1_INTC_s114
                                 + 1.61374 w_k1_KO_s114 + 1.14183 w_k1_MAT_s114
                                 + 1.15791 w_k1_MSFT_s114 + 1.2603 w_k1_PG_s114
                                 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.820813 w_k1_ADM_s114
                                 + 0.824983 w_k1_AXP_s114
                                 + 0.820123 w_k1_DIS_s114
                                 + 0.797371 w_k1_FDX_s114
                                 + 0.896772 w_k1_IBM_s114
                                 + 0.872849 w_k1_INTC_s114
                                 + 0.82355 w_k1_KO_s114
                                 + 0.717995 w_k1_MAT_s114
                                 + 0.812135 w_k1_MSFT_s114
                                 + 0.920509 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.01984 w_k1_ADM_s114
                                 + 0.97848 w_k1_AXP_s114
                                 + 1.28913 w_k1_DIS_s114
                                 + 1.10614 w_k1_FDX_s114
                                 + 1.02996 w_k1_IBM_s114
                                 + 1.04594 w_k1_INTC_s114
                                 + 1.32389 w_k1_KO_s114
                                 + 0.904978 w_k1_MAT_s114
                                 + 1.22018 w_k1_MSFT_s114
                                 + 1.19187 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.797615 w_k1_ADM_s115
                                 + 0.775386 w_k1_AXP_s115
                                 + 0.673214 w_k1_DIS_s115
                                 + 0.880415 w_k1_FDX_s115
                                 + 0.797807 w_k1_IBM_s115
                                 + 0.752443 w_k1_INTC_s115
                                 + 0.769564 w_k1_KO_s115
                                 + 0.634119 w_k1_MAT_s115
                                 + 0.802822 w_k1_MSFT_s115
                                 + 0.858624 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.29977 w_k1_ADM_s115
                                 + 1.4099 w_k1_AXP_s115 + 1.5489 w_k1_DIS_s115
                                 + 1.33119 w_k1_FDX_s115
                                 + 1.74823 w_k1_IBM_s115
                                 + 1.12857 w_k1_INTC_s115 + 1.4875 w_k1_KO_s115
                                 + 1.80741 w_k1_MAT_s115
                                 + 1.23097 w_k1_MSFT_s115
                                 + 1.23489 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.807718 w_k1_ADM_s115
                                 + 0.64293 w_k1_AXP_s115
                                 + 0.762424 w_k1_DIS_s115
                                 + 0.887349 w_k1_FDX_s115
                                 + 0.54075 w_k1_IBM_s115
                                 + 0.729897 w_k1_INTC_s115
                                 + 0.702849 w_k1_KO_s115
                                 + 0.498115 w_k1_MAT_s115
                                 + 0.936998 w_k1_MSFT_s115
                                 + 0.797563 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.1443 w_k1_ADM_s115
                                 + 1.02882 w_k1_AXP_s115
                                 + 1.26378 w_k1_DIS_s115
                                 + 1.23859 w_k1_FDX_s115
                                 + 0.84199 w_k1_IBM_s115
                                 + 1.14765 w_k1_INTC_s115
                                 + 0.95865 w_k1_KO_s115
                                 + 0.529969 w_k1_MAT_s115
                                 + 1.07796 w_k1_MSFT_s115
                                 + 1.06505 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.908481 w_k1_ADM_s115
                                 + 0.835874 w_k1_AXP_s115
                                 + 0.878885 w_k1_DIS_s115
                                 + 0.879672 w_k1_FDX_s115
                                 + 1.06313 w_k1_IBM_s115
                                 + 0.965268 w_k1_INTC_s115
                                 + 0.880724 w_k1_KO_s115
                                 + 0.87335 w_k1_MAT_s115
                                 + 0.921376 w_k1_MSFT_s115
                                 + 0.857875 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.708751 w_k1_ADM_s115
                                 + 0.461793 w_k1_AXP_s115
                                 + 0.68863 w_k1_DIS_s115
                                 + 0.692855 w_k1_FDX_s115
                                 + 0.742005 w_k1_IBM_s115
                                 + 0.798086 w_k1_INTC_s115
                                 + 0.684358 w_k1_KO_s115
                                 + 0.324086 w_k1_MAT_s115
                                 + 0.801125 w_k1_MSFT_s115
                                 + 0.748421 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.920911 w_k1_ADM_s115
                                 + 0.941081 w_k1_AXP_s115
                                 + 1.13108 w_k1_DIS_s115
                                 + 0.933858 w_k1_FDX_s115
                                 + 1.4558 w_k1_IBM_s115
                                 + 0.887383 w_k1_INTC_s115
                                 + 1.30707 w_k1_KO_s115 + 1.3894 w_k1_MAT_s115
                                 + 0.957 w_k1_MSFT_s115 + 1.14581 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.960084 w_k1_ADM_s115
                                 + 0.933687 w_k1_AXP_s115
                                 + 1.07553 w_k1_DIS_s115
                                 + 1.00658 w_k1_FDX_s115
                                 + 0.813798 w_k1_IBM_s115
                                 + 1.10248 w_k1_INTC_s115
                                 + 0.919192 w_k1_KO_s115
                                 + 0.78308 w_k1_MAT_s115
                                 + 1.05811 w_k1_MSFT_s115
                                 + 0.920444 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.857718 w_k1_ADM_s116
                                 + 0.810554 w_k1_AXP_s116
                                 + 0.747814 w_k1_DIS_s116
                                 + 0.790465 w_k1_FDX_s116
                                 + 0.8871 w_k1_IBM_s116
                                 + 0.988792 w_k1_INTC_s116
                                 + 0.886999 w_k1_KO_s116
                                 + 0.783472 w_k1_MAT_s116
                                 + 0.786187 w_k1_MSFT_s116
                                 + 0.896655 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.2597 w_k1_ADM_s116
                                 + 1.30561 w_k1_AXP_s116
                                 + 1.76632 w_k1_DIS_s116
                                 + 1.23465 w_k1_FDX_s116
                                 + 1.43165 w_k1_IBM_s116
                                 + 1.49007 w_k1_INTC_s116
                                 + 1.44101 w_k1_KO_s116 + 1.11783 w_k1_MAT_s116
                                 + 1.32459 w_k1_MSFT_s116
                                 + 1.31502 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.842309 w_k1_ADM_s116
                                 + 0.847663 w_k1_AXP_s116
                                 + 0.934207 w_k1_DIS_s116
                                 + 0.894848 w_k1_FDX_s116
                                 + 0.665538 w_k1_IBM_s116
                                 + 0.979212 w_k1_INTC_s116
                                 + 0.805289 w_k1_KO_s116
                                 + 0.653249 w_k1_MAT_s116
                                 + 0.954137 w_k1_MSFT_s116
                                 + 0.89073 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.03099 w_k1_ADM_s116
                                 + 1.14143 w_k1_AXP_s116
                                 + 1.20652 w_k1_DIS_s116
                                 + 1.16669 w_k1_FDX_s116
                                 + 0.758945 w_k1_IBM_s116
                                 + 1.20089 w_k1_INTC_s116
                                 + 0.931398 w_k1_KO_s116
                                 + 0.790988 w_k1_MAT_s116
                                 + 1.0989 w_k1_MSFT_s116 + 1.05023 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.05363 w_k1_ADM_s116
                                 + 1.21187 w_k1_AXP_s116
                                 + 1.29559 w_k1_DIS_s116
                                 + 1.11871 w_k1_FDX_s116
                                 + 1.11849 w_k1_IBM_s116
                                 + 1.0467 w_k1_INTC_s116 + 1.01752 w_k1_KO_s116
                                 + 0.989801 w_k1_MAT_s116
                                 + 1.0646 w_k1_MSFT_s116 + 1.06772 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.912303 w_k1_ADM_s116
                                 + 0.77092 w_k1_AXP_s116
                                 + 0.793889 w_k1_DIS_s116
                                 + 0.978326 w_k1_FDX_s116
                                 + 0.623626 w_k1_IBM_s116
                                 + 1.06781 w_k1_INTC_s116
                                 + 0.889008 w_k1_KO_s116
                                 + 0.632302 w_k1_MAT_s116
                                 + 1.03156 w_k1_MSFT_s116
                                 + 0.924054 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.886902 w_k1_ADM_s116
                                 + 0.79044 w_k1_AXP_s116 + 1.0883 w_k1_DIS_s116
                                 + 0.939993 w_k1_FDX_s116
                                 + 0.827442 w_k1_IBM_s116
                                 + 0.993494 w_k1_INTC_s116
                                 + 1.09371 w_k1_KO_s116
                                 + 0.739332 w_k1_MAT_s116
                                 + 1.0608 w_k1_MSFT_s116 + 1.06452 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.741029 w_k1_ADM_s116
                                 + 1.00654 w_k1_AXP_s116
                                 + 1.06434 w_k1_DIS_s116
                                 + 0.876033 w_k1_FDX_s116
                                 + 1.4193 w_k1_IBM_s116
                                 + 1.08475 w_k1_INTC_s116
                                 + 1.25649 w_k1_KO_s116 + 1.23983 w_k1_MAT_s116
                                 + 1.06071 w_k1_MSFT_s116
                                 + 1.06027 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.675797 w_k1_ADM_s117
                                 + 0.6799 w_k1_AXP_s117 + 0.80312 w_k1_DIS_s117
                                 + 0.872033 w_k1_FDX_s117
                                 + 0.858938 w_k1_IBM_s117
                                 + 0.831874 w_k1_INTC_s117
                                 + 0.888901 w_k1_KO_s117
                                 + 0.822103 w_k1_MAT_s117
                                 + 0.790945 w_k1_MSFT_s117
                                 + 0.853716 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.898267 w_k1_ADM_s117
                                 + 1.01357 w_k1_AXP_s117
                                 + 0.941609 w_k1_DIS_s117
                                 + 1.04287 w_k1_FDX_s117 + 1.1298 w_k1_IBM_s117
                                 + 0.89429 w_k1_INTC_s117
                                 + 0.919228 w_k1_KO_s117
                                 + 1.01496 w_k1_MAT_s117
                                 + 0.950156 w_k1_MSFT_s117
                                 + 1.0251 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.926164 w_k1_ADM_s117
                                 + 1.03237 w_k1_AXP_s117
                                 + 0.819449 w_k1_DIS_s117
                                 + 0.942763 w_k1_FDX_s117
                                 + 1.46753 w_k1_IBM_s117
                                 + 0.815154 w_k1_INTC_s117
                                 + 1.24768 w_k1_KO_s117 + 1.4415 w_k1_MAT_s117
                                 + 0.871352 w_k1_MSFT_s117
                                 + 1.09069 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.765586 w_k1_ADM_s117
                                 + 0.921714 w_k1_AXP_s117
                                 + 0.537208 w_k1_DIS_s117
                                 + 0.816158 w_k1_FDX_s117
                                 + 0.916914 w_k1_IBM_s117
                                 + 0.695392 w_k1_INTC_s117
                                 + 0.793603 w_k1_KO_s117
                                 + 0.889483 w_k1_MAT_s117
                                 + 0.737534 w_k1_MSFT_s117
                                 + 0.85101 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.877306 w_k1_ADM_s117
                                 + 0.836487 w_k1_AXP_s117
                                 + 1.0121 w_k1_DIS_s117
                                 + 0.954859 w_k1_FDX_s117
                                 + 1.02385 w_k1_IBM_s117
                                 + 0.813664 w_k1_INTC_s117
                                 + 0.967532 w_k1_KO_s117
                                 + 0.779404 w_k1_MAT_s117
                                 + 0.963186 w_k1_MSFT_s117
                                 + 0.867375 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.26945 w_k1_ADM_s117
                                 + 1.05896 w_k1_AXP_s117
                                 + 1.44111 w_k1_DIS_s117
                                 + 1.24468 w_k1_FDX_s117
                                 + 1.38956 w_k1_IBM_s117
                                 + 1.13032 w_k1_INTC_s117
                                 + 1.39741 w_k1_KO_s117 + 1.42485 w_k1_MAT_s117
                                 + 1.23331 w_k1_MSFT_s117
                                 + 1.20533 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.25147 w_k1_ADM_s117
                                 + 1.18369 w_k1_AXP_s117
                                 + 1.44984 w_k1_DIS_s117
                                 + 1.44302 w_k1_FDX_s117
                                 + 1.04847 w_k1_IBM_s117
                                 + 0.994438 w_k1_INTC_s117
                                 + 1.24326 w_k1_KO_s117 + 1.07262 w_k1_MAT_s117
                                 + 1.2137 w_k1_MSFT_s117 + 1.13978 w_k1_PG_s117
                                 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.21143 w_k1_ADM_s117
                                 + 1.23257 w_k1_AXP_s117
                                 + 0.96775 w_k1_DIS_s117
                                 + 1.22898 w_k1_FDX_s117
                                 + 0.771407 w_k1_IBM_s117
                                 + 1.20972 w_k1_INTC_s117
                                 + 0.887609 w_k1_KO_s117
                                 + 0.764742 w_k1_MAT_s117
                                 + 1.04218 w_k1_MSFT_s117
                                 + 1.02556 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.00690374474876501 w_k1_ADM_s000
                                 + 0.00974448613786194 w_k1_ADM_s001
                                 + 0.0130976449723193 w_k1_ADM_s002
                                 + 0.0202117746021263 w_k1_ADM_s003
                                 + 0.00523435688843611 w_k1_ADM_s004
                                 + 0.00697004820853455 w_k1_ADM_s005
                                 + 0.0127464692835969 w_k1_ADM_s006
                                 + 0.00737580437959828 w_k1_ADM_s007
                                 + 0.0159659550232488 w_k1_ADM_s010
                                 + 0.021091527650101 w_k1_ADM_s011
                                 + 0.0459347324030209 w_k1_ADM_s012
                                 + 0.054712404984217 w_k1_ADM_s013
                                 + 0.0361282936643068 w_k1_ADM_s014
                                 + 0.0535370027761294 w_k1_ADM_s015
                                 + 0.0675745785945342 w_k1_ADM_s016
                                 + 0.0531445641387284 w_k1_ADM_s017
                                 + 0.0161624911701497 w_k1_ADM_s100
                                 + 0.0474548996686449 w_k1_ADM_s101
                                 + 0.018021433684368 w_k1_ADM_s102
                                 + 0.0415412638577509 w_k1_ADM_s103
                                 + 0.053276943108345 w_k1_ADM_s104
                                 + 0.0561442840361312 w_k1_ADM_s105
                                 + 0.0183160634727182 w_k1_ADM_s106
                                 + 0.0181530783265859 w_k1_ADM_s107
                                 + 0.0185743587990094 w_k1_ADM_s110
                                 + 0.0513334719716899 w_k1_ADM_s111
                                 + 0.0639848152986555 w_k1_ADM_s112
                                 + 0.0693391187826633 w_k1_ADM_s113
                                 + 0.0308409609125174 w_k1_ADM_s114
                                 + 0.020901410754375 w_k1_ADM_s115
                                 + 0.0274754637021677 w_k1_ADM_s116
                                 + 0.0215113304084986 w_k1_ADM_s117
                                 + 0.00531122623510601 w_k1_AXP_s000
                                 + 0.0102016088432855 w_k1_AXP_s001
                                 + 0.0115698766230282 w_k1_AXP_s002
                                 + 0.0166181364082719 w_k1_AXP_s003
                                 + 0.00615630014890399 w_k1_AXP_s004
                                 + 0.00644966080961883 w_k1_AXP_s005
                                 + 0.0150872613628761 w_k1_AXP_s006
                                 + 0.00655992106269288 w_k1_AXP_s007
                                 + 0.0191493203147099 w_k1_AXP_s010
                                 + 0.0258200727810122 w_k1_AXP_s011
                                 + 0.0481928367446145 w_k1_AXP_s012
                                 + 0.0562942248413249 w_k1_AXP_s013
                                 + 0.0393876163933182 w_k1_AXP_s014
                                 + 0.0561626448086226 w_k1_AXP_s015
                                 + 0.071159392553954 w_k1_AXP_s016
                                 + 0.0532684538615826 w_k1_AXP_s017
                                 + 0.015566558808952 w_k1_AXP_s100
                                 + 0.049009305150396 w_k1_AXP_s101
                                 + 0.0190580900268643 w_k1_AXP_s102
                                 + 0.0494986296895385 w_k1_AXP_s103
                                 + 0.054712366026977 w_k1_AXP_s104
                                 + 0.0549059129406556 w_k1_AXP_s105
                                 + 0.0201669339330023 w_k1_AXP_s106
                                 + 0.0212115369654462 w_k1_AXP_s107
                                 + 0.0190698806567235 w_k1_AXP_s110
                                 + 0.0496630854302431 w_k1_AXP_s111
                                 + 0.0693795586503383 w_k1_AXP_s112
                                 + 0.0657128898573725 w_k1_AXP_s113
                                 + 0.0330653668532373 w_k1_AXP_s114
                                 + 0.0199059669651639 w_k1_AXP_s115
                                 + 0.0293844145644016 w_k1_AXP_s116
                                 + 0.0218975310676296 w_k1_AXP_s117
                                 + 0.00564256370853543 w_k1_DIS_s000
                                 + 0.0102113705406429 w_k1_DIS_s001
                                 + 0.0112266136897834 w_k1_DIS_s002
                                 + 0.0192118014079415 w_k1_DIS_s003
                                 + 0.00692389304762694 w_k1_DIS_s004
                                 + 0.00658157558841633 w_k1_DIS_s005
                                 + 0.0142973298195523 w_k1_DIS_s006
                                 + 0.00697977572547059 w_k1_DIS_s007
                                 + 0.0172985306876244 w_k1_DIS_s010
                                 + 0.0245252784718813 w_k1_DIS_s011
                                 + 0.047798243224803 w_k1_DIS_s012
                                 + 0.0573370016033646 w_k1_DIS_s013
                                 + 0.0372135014418891 w_k1_DIS_s014
                                 + 0.0547330486029449 w_k1_DIS_s015
                                 + 0.0670510671421317 w_k1_DIS_s016
                                 + 0.053460915768028 w_k1_DIS_s017
                                 + 0.0160891384447322 w_k1_DIS_s100
                                 + 0.0515139363986911 w_k1_DIS_s101
                                 + 0.0188629179895827 w_k1_DIS_s102
                                 + 0.0478392743086 w_k1_DIS_s103
                                 + 0.0549092315251879 w_k1_DIS_s104
                                 + 0.0547913631337334 w_k1_DIS_s105
                                 + 0.0212513543494713 w_k1_DIS_s106
                                 + 0.0186169438119269 w_k1_DIS_s107
                                 + 0.0198767649611813 w_k1_DIS_s110
                                 + 0.0570905119133006 w_k1_DIS_s111
                                 + 0.0679155397114032 w_k1_DIS_s112
                                 + 0.0606904287924652 w_k1_DIS_s113
                                 + 0.0331124755094361 w_k1_DIS_s114
                                 + 0.022561007645562 w_k1_DIS_s115
                                 + 0.0323992670510784 w_k1_DIS_s116
                                 + 0.0217897429319255 w_k1_DIS_s117
                                 + 0.00545012394638406 w_k1_FDX_s000
                                 + 0.0101869315308447 w_k1_FDX_s001
                                 + 0.0117912982775327 w_k1_FDX_s002
                                 + 0.0179109225956218 w_k1_FDX_s003
                                 + 0.0065354396968006 w_k1_FDX_s004
                                 + 0.00673010124504999 w_k1_FDX_s005
                                 + 0.0136041860592247 w_k1_FDX_s006
                                 + 0.00701211907892092 w_k1_FDX_s007
                                 + 0.0167671316437592 w_k1_FDX_s010
                                 + 0.0236892126804136 w_k1_FDX_s011
                                 + 0.0462312761063519 w_k1_FDX_s012
                                 + 0.0567226355565643 w_k1_FDX_s013
                                 + 0.0361357064934936 w_k1_FDX_s014
                                 + 0.0520905059419503 w_k1_FDX_s015
                                 + 0.0656072627227539 w_k1_FDX_s016
                                 + 0.0516466828932487 w_k1_FDX_s017
                                 + 0.0153094446092038 w_k1_FDX_s100
                                 + 0.0501343647125472 w_k1_FDX_s101
                                 + 0.0180434010615053 w_k1_FDX_s102
                                 + 0.0466207842467469 w_k1_FDX_s103
                                 + 0.0563911754289677 w_k1_FDX_s104
                                 + 0.0529651265165003 w_k1_FDX_s105
                                 + 0.0203840781905209 w_k1_FDX_s106
                                 + 0.0186085561871224 w_k1_FDX_s107
                                 + 0.0198830053103642 w_k1_FDX_s110
                                 + 0.0529756492597534 w_k1_FDX_s111
                                 + 0.0624866578444813 w_k1_FDX_s112
                                 + 0.0640519068880615 w_k1_FDX_s113
                                 + 0.0317513363046717 w_k1_FDX_s114
                                 + 0.0218463680357943 w_k1_FDX_s115
                                 + 0.0288062397726787 w_k1_FDX_s116
                                 + 0.0232594024364327 w_k1_FDX_s117
                                 + 0.00533895153771666 w_k1_IBM_s000
                                 + 0.00931367540483714 w_k1_IBM_s001
                                 + 0.0114707092155783 w_k1_IBM_s002
                                 + 0.0179632837894279 w_k1_IBM_s003
                                 + 0.00673041377639496 w_k1_IBM_s004
                                 + 0.00661150037778219 w_k1_IBM_s005
                                 + 0.0137942164504638 w_k1_IBM_s006
                                 + 0.00692588355330994 w_k1_IBM_s007
                                 + 0.0181234213298196 w_k1_IBM_s010
                                 + 0.0239039047438043 w_k1_IBM_s011
                                 + 0.0459934125903249 w_k1_IBM_s012
                                 + 0.0594793409099025 w_k1_IBM_s013
                                 + 0.0380257120044231 w_k1_IBM_s014
                                 + 0.0526059577236322 w_k1_IBM_s015
                                 + 0.063780314369124 w_k1_IBM_s016
                                 + 0.0522653663185256 w_k1_IBM_s017
                                 + 0.0150829553012408 w_k1_IBM_s100
                                 + 0.0509984387151536 w_k1_IBM_s101
                                 + 0.0178250976607436 w_k1_IBM_s102
                                 + 0.0484105748284741 w_k1_IBM_s103
                                 + 0.0565447030048965 w_k1_IBM_s104
                                 + 0.0529776642901715 w_k1_IBM_s105
                                 + 0.0200254220894526 w_k1_IBM_s106
                                 + 0.0185253594040056 w_k1_IBM_s107
                                 + 0.0206994219916501 w_k1_IBM_s110
                                 + 0.0550079328465344 w_k1_IBM_s111
                                 + 0.0633040951361385 w_k1_IBM_s112
                                 + 0.0641207729059016 w_k1_IBM_s113
                                 + 0.0324651807695584 w_k1_IBM_s114
                                 + 0.0218963135291603 w_k1_IBM_s115
                                 + 0.0295313954668574 w_k1_IBM_s116
                                 + 0.0241204537711056 w_k1_IBM_s117
                                 + 0.00523948566018149 w_k1_INTC_s000
                                 + 0.0103386806144595 w_k1_INTC_s001
                                 + 0.0104867579118193 w_k1_INTC_s002
                                 + 0.0172041255550847 w_k1_INTC_s003
                                 + 0.00642821066453683 w_k1_INTC_s004
                                 + 0.00689286808946756 w_k1_INTC_s005
                                 + 0.0150986919735578 w_k1_INTC_s006
                                 + 0.00588047062889862 w_k1_INTC_s007
                                 + 0.0187540763688853 w_k1_INTC_s010
                                 + 0.0227134407845416 w_k1_INTC_s011
                                 + 0.0498732190209713 w_k1_INTC_s012
                                 + 0.0535953734772822 w_k1_INTC_s013
                                 + 0.0401341263976831 w_k1_INTC_s014
                                 + 0.0601790685967314 w_k1_INTC_s015
                                 + 0.0616657698639421 w_k1_INTC_s016
                                 + 0.0509633957436499 w_k1_INTC_s017
                                 + 0.013366574565258 w_k1_INTC_s100
                                 + 0.047384324192052 w_k1_INTC_s101
                                 + 0.0161324061027894 w_k1_INTC_s102
                                 + 0.0445736278227686 w_k1_INTC_s103
                                 + 0.0611788157977732 w_k1_INTC_s104
                                 + 0.0557504573617778 w_k1_INTC_s105
                                 + 0.0187881419635786 w_k1_INTC_s106
                                 + 0.0189264273039096 w_k1_INTC_s107
                                 + 0.0206719014529791 w_k1_INTC_s110
                                 + 0.0598508071263933 w_k1_INTC_s111
                                 + 0.0701520311969361 w_k1_INTC_s112
                                 + 0.0576508163493965 w_k1_INTC_s113
                                 + 0.0318010451796409 w_k1_INTC_s114
                                 + 0.0209439826776038 w_k1_INTC_s115
                                 + 0.0317300100327791 w_k1_INTC_s116
                                 + 0.0202953191708707 w_k1_INTC_s117
                                 + 0.00577196926866001 w_k1_KO_s000
                                 + 0.0101825899976694 w_k1_KO_s001
                                 + 0.0115944406996133 w_k1_KO_s002
                                 + 0.0190271905093472 w_k1_KO_s003
                                 + 0.00701417098560227 w_k1_KO_s004
                                 + 0.00683358464038633 w_k1_KO_s005
                                 + 0.0141500441349227 w_k1_KO_s006
                                 + 0.00685290485395965 w_k1_KO_s007
                                 + 0.0180522534705613 w_k1_KO_s010
                                 + 0.0229936853442045 w_k1_KO_s011
                                 + 0.0500356867123064 w_k1_KO_s012
                                 + 0.0529711972813129 w_k1_KO_s013
                                 + 0.0347150092758252 w_k1_KO_s014
                                 + 0.0549835425889027 w_k1_KO_s015
                                 + 0.0624874789951434 w_k1_KO_s016
                                 + 0.0512403183098457 w_k1_KO_s017
                                 + 0.0146035511043228 w_k1_KO_s100
                                 + 0.047328297995483 w_k1_KO_s101
                                 + 0.0169050193073142 w_k1_KO_s102
                                 + 0.047917627565244 w_k1_KO_s103
                                 + 0.0568976954818531 w_k1_KO_s104
                                 + 0.0547508718909964 w_k1_KO_s105
                                 + 0.0170056854572462 w_k1_KO_s106
                                 + 0.0187470327721219 w_k1_KO_s107
                                 + 0.0210881225850945 w_k1_KO_s110
                                 + 0.0509242743318317 w_k1_KO_s111
                                 + 0.0649294293405032 w_k1_KO_s112
                                 + 0.0654553270554773 w_k1_KO_s113
                                 + 0.0357882189166765 w_k1_KO_s114
                                 + 0.0213015626883563 w_k1_KO_s115
                                 + 0.0301492175751172 w_k1_KO_s116
                                 + 0.0226556536811083 w_k1_KO_s117
                                 + 0.0055953283140698 w_k1_MAT_s000
                                 + 0.0107297903462411 w_k1_MAT_s001
                                 + 0.0103806147010648 w_k1_MAT_s002
                                 + 0.0177528917335585 w_k1_MAT_s003
                                 + 0.00745705544150936 w_k1_MAT_s004
                                 + 0.00606782583681894 w_k1_MAT_s005
                                 + 0.0124452869424115 w_k1_MAT_s006
                                 + 0.00671799420313012 w_k1_MAT_s007
                                 + 0.0173360824232274 w_k1_MAT_s010
                                 + 0.0233297537150432 w_k1_MAT_s011
                                 + 0.0515398929311332 w_k1_MAT_s012
                                 + 0.0546564758678697 w_k1_MAT_s013
                                 + 0.0288432178118359 w_k1_MAT_s014
                                 + 0.0508073950865998 w_k1_MAT_s015
                                 + 0.0609946051347499 w_k1_MAT_s016
                                 + 0.0549820607336711 w_k1_MAT_s017
                                 + 0.0130520238019387 w_k1_MAT_s100
                                 + 0.0481222542246436 w_k1_MAT_s101
                                 + 0.017403318390569 w_k1_MAT_s102
                                 + 0.057871749199021 w_k1_MAT_s103
                                 + 0.048545613092644 w_k1_MAT_s104
                                 + 0.0567663239393672 w_k1_MAT_s105
                                 + 0.0209677700573589 w_k1_MAT_s106
                                 + 0.021514898284151 w_k1_MAT_s107
                                 + 0.0224140100640299 w_k1_MAT_s110
                                 + 0.0552723603295683 w_k1_MAT_s111
                                 + 0.0660821697200637 w_k1_MAT_s112
                                 + 0.0635318532572657 w_k1_MAT_s113
                                 + 0.0292566907088992 w_k1_MAT_s114
                                 + 0.0189299672198079 w_k1_MAT_s115
                                 + 0.0262572296176058 w_k1_MAT_s116
                                 + 0.0229086455720251 w_k1_MAT_s117
                                 + 0.00568180678081886 w_k1_MSFT_s000
                                 + 0.0107914474093808 w_k1_MSFT_s001
                                 + 0.0110764210812733 w_k1_MSFT_s002
                                 + 0.0174194500904775 w_k1_MSFT_s003
                                 + 0.00630006419142726 w_k1_MSFT_s004
                                 + 0.00692621208569024 w_k1_MSFT_s005
                                 + 0.0134365902175369 w_k1_MSFT_s006
                                 + 0.00668790529677242 w_k1_MSFT_s007
                                 + 0.0175459118529341 w_k1_MSFT_s010
                                 + 0.0236375413477323 w_k1_MSFT_s011
                                 + 0.0507018635119625 w_k1_MSFT_s012
                                 + 0.0594540503518123 w_k1_MSFT_s013
                                 + 0.0331873777698585 w_k1_MSFT_s014
                                 + 0.0486707933231675 w_k1_MSFT_s015
                                 + 0.0667819584845765 w_k1_MSFT_s016
                                 + 0.0482906407642732 w_k1_MSFT_s017
                                 + 0.0135266402038825 w_k1_MSFT_s100
                                 + 0.0418186797525225 w_k1_MSFT_s101
                                 + 0.0169080154304056 w_k1_MSFT_s102
                                 + 0.0437151309731127 w_k1_MSFT_s103
                                 + 0.0618329793933123 w_k1_MSFT_s104
                                 + 0.0552111479685828 w_k1_MSFT_s105
                                 + 0.0203589795789735 w_k1_MSFT_s106
                                 + 0.0181667256313751 w_k1_MSFT_s107
                                 + 0.0193131296886165 w_k1_MSFT_s110
                                 + 0.0540704249858544 w_k1_MSFT_s111
                                 + 0.0645321382161607 w_k1_MSFT_s112
                                 + 0.0639641616909322 w_k1_MSFT_s113
                                 + 0.0307097515822919 w_k1_MSFT_s114
                                 + 0.0212596023687228 w_k1_MSFT_s115
                                 + 0.0298066831768977 w_k1_MSFT_s116
                                 + 0.0214665022123412 w_k1_MSFT_s117
                                 + 0.00563736814181022 w_k1_PG_s000
                                 + 0.0104925734109805 w_k1_PG_s001
                                 + 0.0124914551375147 w_k1_PG_s002
                                 + 0.0177320820794219 w_k1_PG_s003
                                 + 0.00651091023440245 w_k1_PG_s004
                                 + 0.00696051917580314 w_k1_PG_s005
                                 + 0.0138285780721994 w_k1_PG_s006
                                 + 0.00788286157197069 w_k1_PG_s007
                                 + 0.0186268037330635 w_k1_PG_s010
                                 + 0.0247364668993056 w_k1_PG_s011
                                 + 0.0493279243539391 w_k1_PG_s012
                                 + 0.053060891156773 w_k1_PG_s013
                                 + 0.0330091551481565 w_k1_PG_s014
                                 + 0.0494818230165382 w_k1_PG_s015
                                 + 0.0643579598874776 w_k1_PG_s016
                                 + 0.0524284361139085 w_k1_PG_s017
                                 + 0.0143536881963258 w_k1_PG_s100
                                 + 0.0483224164553275 w_k1_PG_s101
                                 + 0.0197725887695864 w_k1_PG_s102
                                 + 0.0437192339374229 w_k1_PG_s103
                                 + 0.0569338190568909 w_k1_PG_s104
                                 + 0.0518635791086791 w_k1_PG_s105
                                 + 0.0204321097736565 w_k1_PG_s106
                                 + 0.0181832130711075 w_k1_PG_s107
                                 + 0.0196244052245645 w_k1_PG_s110
                                 + 0.0549009818173973 w_k1_PG_s111
                                 + 0.0673282444184472 w_k1_PG_s112
                                 + 0.0658864564368065 w_k1_PG_s113
                                 + 0.0336374479925858 w_k1_PG_s114
                                 + 0.021048525392167 w_k1_PG_s115
                                 + 0.0298519000433892 w_k1_PG_s116
                                 + 0.0222838865218893 w_k1_PG_s117 >= 1.15
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
