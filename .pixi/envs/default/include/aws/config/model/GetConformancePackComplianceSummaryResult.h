﻿/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#pragma once
#include <aws/config/ConfigService_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/config/model/ConformancePackComplianceSummary.h>
#include <utility>

namespace Aws
{
template<typename RESULT_TYPE>
class AmazonWebServiceResult;

namespace Utils
{
namespace Json
{
  class JsonValue;
} // namespace Json
} // namespace Utils
namespace ConfigService
{
namespace Model
{
  class GetConformancePackComplianceSummaryResult
  {
  public:
    AWS_CONFIGSERVICE_API GetConformancePackComplianceSummaryResult();
    AWS_CONFIGSERVICE_API GetConformancePackComplianceSummaryResult(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);
    AWS_CONFIGSERVICE_API GetConformancePackComplianceSummaryResult& operator=(const Aws::AmazonWebServiceResult<Aws::Utils::Json::JsonValue>& result);


    ///@{
    /**
     * <p>A list of <code>ConformancePackComplianceSummary</code> objects. </p>
     */
    inline const Aws::Vector<ConformancePackComplianceSummary>& GetConformancePackComplianceSummaryList() const{ return m_conformancePackComplianceSummaryList; }
    inline void SetConformancePackComplianceSummaryList(const Aws::Vector<ConformancePackComplianceSummary>& value) { m_conformancePackComplianceSummaryList = value; }
    inline void SetConformancePackComplianceSummaryList(Aws::Vector<ConformancePackComplianceSummary>&& value) { m_conformancePackComplianceSummaryList = std::move(value); }
    inline GetConformancePackComplianceSummaryResult& WithConformancePackComplianceSummaryList(const Aws::Vector<ConformancePackComplianceSummary>& value) { SetConformancePackComplianceSummaryList(value); return *this;}
    inline GetConformancePackComplianceSummaryResult& WithConformancePackComplianceSummaryList(Aws::Vector<ConformancePackComplianceSummary>&& value) { SetConformancePackComplianceSummaryList(std::move(value)); return *this;}
    inline GetConformancePackComplianceSummaryResult& AddConformancePackComplianceSummaryList(const ConformancePackComplianceSummary& value) { m_conformancePackComplianceSummaryList.push_back(value); return *this; }
    inline GetConformancePackComplianceSummaryResult& AddConformancePackComplianceSummaryList(ConformancePackComplianceSummary&& value) { m_conformancePackComplianceSummaryList.push_back(std::move(value)); return *this; }
    ///@}

    ///@{
    /**
     * <p>The nextToken string returned on a previous page that you use to get the next
     * page of results in a paginated response.</p>
     */
    inline const Aws::String& GetNextToken() const{ return m_nextToken; }
    inline void SetNextToken(const Aws::String& value) { m_nextToken = value; }
    inline void SetNextToken(Aws::String&& value) { m_nextToken = std::move(value); }
    inline void SetNextToken(const char* value) { m_nextToken.assign(value); }
    inline GetConformancePackComplianceSummaryResult& WithNextToken(const Aws::String& value) { SetNextToken(value); return *this;}
    inline GetConformancePackComplianceSummaryResult& WithNextToken(Aws::String&& value) { SetNextToken(std::move(value)); return *this;}
    inline GetConformancePackComplianceSummaryResult& WithNextToken(const char* value) { SetNextToken(value); return *this;}
    ///@}

    ///@{
    
    inline const Aws::String& GetRequestId() const{ return m_requestId; }
    inline void SetRequestId(const Aws::String& value) { m_requestId = value; }
    inline void SetRequestId(Aws::String&& value) { m_requestId = std::move(value); }
    inline void SetRequestId(const char* value) { m_requestId.assign(value); }
    inline GetConformancePackComplianceSummaryResult& WithRequestId(const Aws::String& value) { SetRequestId(value); return *this;}
    inline GetConformancePackComplianceSummaryResult& WithRequestId(Aws::String&& value) { SetRequestId(std::move(value)); return *this;}
    inline GetConformancePackComplianceSummaryResult& WithRequestId(const char* value) { SetRequestId(value); return *this;}
    ///@}
  private:

    Aws::Vector<ConformancePackComplianceSummary> m_conformancePackComplianceSummaryList;

    Aws::String m_nextToken;

    Aws::String m_requestId;
  };

} // namespace Model
} // namespace ConfigService
} // namespace Aws
